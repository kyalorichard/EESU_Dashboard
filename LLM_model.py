import pandas as pd
import openai
from tqdm import tqdm
import json
import time
import os
from datetime import datetime
from langdetect import detect  # Language detection library

# ---------------- CONFIG ----------------
openai.api_key = "YOUR_API_KEY"  # Set your OpenAI API key here
INPUT_CSV = "your_data.csv"      # Input CSV file path
SUMMARY_COL = "summary"          # Column in CSV containing text summaries

# ---------------- OUTPUT FOLDER ----------------
OUTPUT_FOLDER = "data"   # Folder to save outputs
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
INTERMEDIATE_PARQUET = os.path.join(OUTPUT_FOLDER, "intermediate_output.parquet")
LOG_FILE = os.path.join(OUTPUT_FOLDER, "batch_processing_log.csv")

# ---------------- BATCH & TOKEN CONFIG ----------------
MAX_BATCH_TOKENS = 2000          # Approximate max tokens per batch including prompts
MIN_BATCH_SIZE = 1               # Minimum number of summaries per batch
MAX_BATCH_SIZE = 50              # Maximum number of summaries per batch
DEFAULT_BATCH_SIZE = 5           # Starting batch size

# API and retries
MAX_RETRIES = 5                  # Max retries on API failure
MODEL = "gpt-5-mini"             # Model to use

# Dynamic batch adjustment
LATENCY_TARGET = 10              # Desired API latency per batch (seconds)
LATENCY_TOLERANCE = 2            # Acceptable deviation
ADJUST_FACTOR = 0.2              # Fractional increase/decrease of batch size

# ---------------- LOAD THEMES FROM JSON ----------------
THEMES_FILE = "data/themes.json"
with open(THEMES_FILE, "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

# ---------------- LOAD CSV ----------------
df = pd.read_csv(INPUT_CSV)

# Initialize output columns if missing
for col in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
    if col not in df.columns:
        df[col] = ""

# ---------------- HELPER FUNCTIONS ----------------
def estimate_tokens(text):
    """Estimate token count for a string (rough approximation)."""
    return len(text) // 4 + 1

def format_theme_options(theme_list, lang):
    """Format theme options with labels and definitions."""
    return ", ".join([f'{t["label"]} ({t["definition"]})' for t in theme_list[lang]])

def build_prompt(batch_summaries):
    """Build the prompt for the batch with language detection and theme options."""
    numbered_texts = []
    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary)
        except:
            lang = "en"
        if lang not in ACTOR_THEMES:
            lang = "en"

        numbered_texts.append(
            f"{idx+1}. Summary: {summary}\n"
            f"Language: {lang}\n"
            f"Actor of repression options: {format_theme_options(ACTOR_THEMES, lang)}\n"
            f"Subject of repression options: {format_theme_options(SUBJECT_THEMES, lang)}\n"
            f"Mechanism of repression options: {format_theme_options(MECHANISM_THEMES, lang)}\n"
            f"Type of event options: {format_theme_options(TYPE_THEMES, lang)}"
        )

    numbered_text = "\n\n".join(numbered_texts)
    prompt = f"""
Extract repression info from each text below. For each summary, select the themes in the provided language.
Return a JSON array of objects in the same order as the texts. Format each object as:

{{
    "Actor of repression": "...",
    "Subject of repression": "...",
    "Mechanism of repression": "...",
    "Type of event": "..."
}}

Texts:
{numbered_text}
"""
    return prompt

def extract_batch(batch_summaries):
    """Send batch to OpenAI API and return JSON results with retries."""
    prompt = build_prompt(batch_summaries)
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            duration = time.time() - start_time
            content = response['choices'][0]['message']['content'].strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fix common JSON formatting issues
                content_fixed = content.replace("\n", "").replace("'", '"')
                data = json.loads(content_fixed)
            return data, duration
        except Exception as e:
            print(f"Error: {e}. Retrying ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(2 ** attempt)
    duration = None
    return [{"Actor of repression": "Error",
             "Subject of repression": "Error",
             "Mechanism of repression": "Error",
             "Type of event": "Error"} for _ in batch_summaries], duration

# ---------------- RESUME AND LOGGING ----------------
if os.path.exists(INTERMEDIATE_PARQUET):
    df_out = pd.read_parquet(INTERMEDIATE_PARQUET)
    start_idx = df_out[df_out["Actor of repression"].isna() | (df_out["Actor of repression"] == "")].index.min()
    if pd.isna(start_idx):
        print("All rows already processed.")
        exit()
else:
    start_idx = 0
    df_out = df.copy()

# Initialize log file if not exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Batch Start Index", "Batch Size", "Estimated Tokens", "API Duration(s)", "Timestamp"]).to_csv(LOG_FILE, index=False)

# ---------------- PROCESS WITH DYNAMIC BATCHING ----------------
i = start_idx
batch_size = DEFAULT_BATCH_SIZE
pbar = tqdm(total=len(df) - start_idx)

while i < len(df):
    batch_summaries = []
    batch_tokens = 0
    batch_start_idx = i

    # Accumulate summaries until token limit or batch size reached
    while i < len(df) and len(batch_summaries) < batch_size:
        summary = str(df[SUMMARY_COL].iloc[i])
        tokens = estimate_tokens(summary) + 50
        if batch_tokens + tokens > MAX_BATCH_TOKENS:
            break
        batch_summaries.append(summary)
        batch_tokens += tokens
        i += 1

    # Extract information
    results, duration = extract_batch(batch_summaries)

    # Write results to dataframe
    for j, res in enumerate(results):
        idx = batch_start_idx + j
        df_out.at[idx, "Actor of repression"] = res.get("Actor of repression", "Unknown")
        df_out.at[idx, "Subject of repression"] = res.get("Subject of repression", "Unknown")
        df_out.at[idx, "Mechanism of repression"] = res.get("Mechanism of repression", "Unknown")
        df_out.at[idx, "Type of event"] = res.get("Type of event", "Unknown")

    # Save intermediate Parquet
    df_out.to_parquet(INTERMEDIATE_PARQUET, index=False, engine="pyarrow")

    # Log batch info
    log_entry = {
        "Batch Start Index": batch_start_idx,
        "Batch Size": len(batch_summaries),
        "Estimated Tokens": batch_tokens,
        "API Duration(s)": duration,
        "Timestamp": datetime.now().isoformat()
    }
    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)

    # Adjust batch size dynamically
    if duration:
        if duration < LATENCY_TARGET - LATENCY_TOLERANCE:
            batch_size = min(int(batch_size * (1 + ADJUST_FACTOR)), MAX_BATCH_SIZE)
        elif duration > LATENCY_TARGET + LATENCY_TOLERANCE:
            batch_size = max(int(batch_size * (1 - ADJUST_FACTOR)), MIN_BATCH_SIZE)

    pbar.update(len(batch_summaries))

pbar.close()

# ---------------- SAVE FINAL OUTPUT ----------------
df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print("Processing complete! Final Parquet saved at:", OUTPUT_PARQUET)
print("Intermediate Parquet saved at:", INTERMEDIATE_PARQUET)
print("Batch log saved at:", LOG_FILE)
