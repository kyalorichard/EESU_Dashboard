import pandas as pd
import openai
from tqdm import tqdm
import json
import time
import os
from datetime import datetime
from langdetect import detect
from dotenv import load_dotenv

# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()  # Load .env file from project root

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env")

# ---------------- CONFIG ----------------
INPUT_CSV = "data/raw_data.csv"             # Input CSV
SUMMARY_COL = "summary"                 # Column containing text summaries

OUTPUT_FOLDER = "data/output"          # Folder to save outputs
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
INTERMEDIATE_PARQUET = os.path.join(OUTPUT_FOLDER, "intermediate_output.parquet")
LOG_FILE = os.path.join(OUTPUT_FOLDER, "batch_processing_log.csv")

MAX_BATCH_TOKENS = 2000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 50
DEFAULT_BATCH_SIZE = 5

MAX_RETRIES = 5
MODEL = "gpt-5-mini"
LATENCY_TARGET = 10
LATENCY_TOLERANCE = 2
ADJUST_FACTOR = 0.2

# ---------------- LOAD THEMES ----------------
THEMES_FILE = "data/themes.json"
with open(THEMES_FILE, "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

# ---------------- LOAD CSV ----------------
df = pd.read_csv(INPUT_CSV)

# Initialize output columns
for col in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
    if col not in df.columns:
        df[col] = ""

# ---------------- HELPER FUNCTIONS ----------------
def estimate_tokens(text):
    return len(text) // 4 + 1

def format_theme_options(theme_list, lang):
    return ", ".join([f'{t["label"]} ({t["definition"]})' for t in theme_list[lang]])

def build_prompt(batch_summaries):
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

    while i < len(df) and len(batch_summaries) < batch_size:
        summary = str(df[SUMMARY_COL].iloc[i])
        tokens = estimate_tokens(summary) + 50
        if batch_tokens + tokens > MAX_BATCH_TOKENS:
            break
        batch_summaries.append(summary)
        batch_tokens += tokens
        i += 1

    results, duration = extract_batch(batch_summaries)

    for j, res in enumerate(results):
        idx = batch_start_idx + j
        df_out.at[idx, "Actor of repression"] = res.get("Actor of repression", "Unknown")
        df_out.at[idx, "Subject of repression"] = res.get("Subject of repression", "Unknown")
        df_out.at[idx, "Mechanism of repression"] = res.get("Mechanism of repression", "Unknown")
        df_out.at[idx, "Type of event"] = res.get("Type of event", "Unknown")

    df_out.to_parquet(INTERMEDIATE_PARQUET, index=False, engine="pyarrow")

    log_entry = {
        "Batch Start Index": batch_start_idx,
        "Batch Size": len(batch_summaries),
        "Estimated Tokens": batch_tokens,
        "API Duration(s)": duration,
        "Timestamp": datetime.now().isoformat()
    }
    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=False, index=False)

    if duration:
        if duration < LATENCY_TARGET - LATENCY_TOLERANCE:
            batch_size = min(int(batch_size * (1 + ADJUST_FACTOR)), MAX_BATCH_SIZE)
        elif duration > LATENCY_TARGET + LATENCY_TOLERANCE:
            batch_size = max(int(batch_size * (1 - ADJUST_FACTOR)), MIN_BATCH_SIZE)

    pbar.update(len(batch_summaries))

pbar.close()

df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print("Processing complete!")
print("Final Parquet:", OUTPUT_PARQUET)
print("Intermediate Parquet:", INTERMEDIATE_PARQUET)
print("Batch log:", LOG_FILE)
