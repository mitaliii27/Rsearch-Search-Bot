import pandas as pd
import re
from tqdm import tqdm

# Configuration 
DATA_FILE = 'arXiv_scientific dataset.csv'
SAMPLE_SIZE = None

# Function to generate arXiv links
def get_arxiv_link(paper_id):
    """
    Generates a valid arXiv URL from various dataset ID formats, removing any version numbers.
    """
    base_url = "https://arxiv.org/abs/"
    if isinstance(paper_id, str) and paper_id.lower().startswith('abs-'):
        cleaned_id = paper_id[4:]
    else:
        cleaned_id = str(paper_id)
    cleaned_id = re.split(r'v\d+$', cleaned_id)[0]
    if cleaned_id and cleaned_id[0].isdigit():
        return f"{base_url}{cleaned_id}"
    else:
        match = re.match(r'(.*)-(\d{7})$', cleaned_id)
        if match:
            category = match.group(1)
            number_part = match.group(2)
            return f"{base_url}{category}/{number_part}"
        else:
            return f"Could not parse ID: {paper_id}"

# Pattern for LaTeX equations (e.g., $...$ or $$...$$)
latex_pattern = re.compile(r'\$.*?\$|\$\$.*?\$\$', re.DOTALL)

# Pattern for citations (e.g., [1], [cs.AI], (Smith et al., 2023))
citation_pattern = re.compile(r'\[\d+\]|\[[a-zA-Z\.\-]+\]|\s*\([A-Za-z\s]+(et al\.)?,\s*\d{4}\)')

# Pattern for newlines
newline_pattern = re.compile(r'\n')

# Pattern for excess whitespace (2 or more spaces)
whitespace_pattern = re.compile(r'\s{2,}')

print(f"Loading data from '{DATA_FILE}'...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' was not found. Make sure it's in the same directory.")
    exit()

df.dropna(subset=['summary'], inplace=True)
df['summary'] = df['summary'].astype(str)

if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    sample_df = df.head(SAMPLE_SIZE)
else:
    sample_df = df

print(f"\nAnalyzing {len(sample_df)} summaries for potential noise...")

# Initialising counters and example lists
counters = {
    "latex_equations": 0,
    "citations": 0,
    "newlines": 0,
    "extra_whitespace": 0
}

examples = {
    "latex_equations": [],
    "citations": [],
    "newlines": [],
    "extra_whitespace": []
}
MAX_EXAMPLES = 3 

# Finding total number of papers need to be pre-processed
for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Analyzing Summaries"):
    summary = row.summary
    paper_id = row.id

    if latex_pattern.search(summary):
        counters["latex_equations"] += 1
        if len(examples["latex_equations"]) < MAX_EXAMPLES:
            examples["latex_equations"].append((summary[:300] + "...", paper_id))

    if citation_pattern.search(summary):
        counters["citations"] += 1
        if len(examples["citations"]) < MAX_EXAMPLES:
            examples["citations"].append((summary[:300] + "...", paper_id))

    if newline_pattern.search(summary):
        counters["newlines"] += 1
        if len(examples["newlines"]) < MAX_EXAMPLES:
            snippet = summary[:300].replace('\n', ' [\\n] ') + "..."
            examples["newlines"].append((snippet, paper_id))

    if whitespace_pattern.search(summary):
        counters["extra_whitespace"] += 1
        if len(examples["extra_whitespace"]) < MAX_EXAMPLES:
            examples["extra_whitespace"].append((summary[:300] + "...", paper_id))


print("\n--- Summary Analysis Report ---")
print(f"Analyzed {len(sample_df)} paper summaries.\n")

for noise_type, count in counters.items():
    percentage = (count / len(sample_df)) * 100
    print(f"Noise Type: {noise_type.replace('_', ' ').title()}")
    print(f"  - Found in: {count} summaries ({percentage:.2f}%)")
    if count > 0:
        print("  - Examples found:")
        for i, (ex_summary, ex_paper_id) in enumerate(examples[noise_type]):
            link = get_arxiv_link(ex_paper_id)
            print(f"    {i+1}. Link: {link}")
            print(f"       Summary: \"{ex_summary}\"")
    print("-" * 20)

print("\nConclusion: The presence of this noise justifies a pre-processing step to sanitize the text before embedding.")


