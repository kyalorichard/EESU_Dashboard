import pandas as pd
import openai
import asyncio
import json
import os
from datetime import datetime
from langdetect import detect
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib
from tqdm.asyncio import tqdm_asyncio
import random
import argparse

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="mock", choices=["mock", "live"])
args = parser.parse_args()
MOCK_MODE = args.mode == "mock"

# ---------------- LOAD ENV ----------------
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
MODEL = "gpt-5-mini"

# Email
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)

# ---------------- FILE PATHS ----------------
INPUT_CSV = "data/raw_data.csv"
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
INTERMEDIATE_PARQUET = os.path.join(OUTPUT_FOLDER, "intermediate_output.parquet")
LOG_FILE = os.path.join(OUTPUT_FOLDER, "batch_processing_log.csv")
FAILED_BATCHES_FILE = os.path.join(OUTPUT_FOLDER, "failed_batches.json")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

# ---------------- BATCH / PROCESS CONFIG ----------------
MAX_BATCH_TOKENS = 10000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 2
DEFAULT_BATCH_SIZE = 2
SAVE_EVERY_N_BATCHES = 2
MAX_RETRIES = 2
LATENCY_TARGET = 5
LATENCY_TOLERANCE = 2
ADJUST_FACTOR = 0.2
CONCURRENT_BATCHES = 3
COST_PER_1K_TOKENS = 0.005
TEST_ROWS = int(os.getenv("TEST_ROWS", 3))

# ---------------- MOCK MODE ----------------
MOCK_CONFIG = {
    "quota_error_batches": 1,
    "malformed_json_prob": 0.3,
    "missing_keys_prob": 0.2,
    "latency_range": (0.5, 2.0),
    "fixed_keys": ["Actor of repression", "Subject of repression",
                   "Mechanism of repression", "Type of event"]
}

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
if TEST_ROWS:
    df = df.head(TEST_ROWS)

for col in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
    if col not in df.columns:
        df[col] = ""

# ---------------- TOKEN ESTIMATION ----------------
try:
    import tiktoken
    encoding = tiktoken.encoding_for_model(MODEL)
    def estimate_tokens(text):
        return len(encoding.encode(text)) + 50
    print("tiktoken detected: Using precise token estimation.")
except ImportError:
    print("tiktoken not found. Using rough token estimation (approx. 1 token per 4 chars).")
    def estimate_tokens(text):
        return max(1, len(text)//4 + 50)

# ---------------- HELPER FUNCTIONS ----------------
def format_theme_options(theme_list, lang):
    return ", ".join([f'{t["label"]} ({t["definition"]})' for t in theme_list.get(lang, theme_list["en"])])

def build_prompt(batch_summaries):
    numbered_texts = []
    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary) if summary.strip() else "en"
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
Extract repression info from each text below. Return a JSON array of objects in the same order:

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

# ---------------- MOCK EXTRACT_BATCH ----------------
async def mock_extract_batch(batch_summaries):
    if not hasattr(mock_extract_batch, "call_count"):
        mock_extract_batch.call_count = 0
        mock_extract_batch.quota_error_batches = 0
        mock_extract_batch.malformed_json_batches = 0
        mock_extract_batch.missing_keys_batches = 0

    if mock_extract_batch.call_count < MOCK_CONFIG["quota_error_batches"]:
        mock_extract_batch.call_count += 1
        mock_extract_batch.quota_error_batches += 1
        raise Exception("You exceeded your current quota (simulated).")

    batch_data = []
    for summary in batch_summaries:
        r = {k: f"{k} Value" for k in MOCK_CONFIG["fixed_keys"]}
        batch_data.append(r)

    mock_extract_batch.call_count += 1
    await asyncio.sleep(random.uniform(*MOCK_CONFIG["latency_range"]))
    return batch_data, random.uniform(*MOCK_CONFIG["latency_range"])

# ---------------- EXTRACT BATCH ----------------
async def extract_batch(batch_summaries):
    if MOCK_MODE:
        return await mock_extract_batch(batch_summaries)

    prompt = build_prompt(batch_summaries)
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data, None
        except Exception as e:
            print(f"Error: {e}. Retrying ({attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(2 ** attempt)

    return [{"Actor of repression": "Error",
             "Subject of repression": "Error",
             "Mechanism of repression": "Error",
             "Type of event": "Error"} for _ in batch_summaries], None

# ---------------- EMAIL FUNCTION ----------------
def send_email_notification(subject, body, to_email):
    if not subject or not body or not to_email:
        print("Cannot send email: missing subject, body, or recipient.")
        return
    if not SMTP_USER or not SMTP_PASS or not SMTP_HOST or not SMTP_PORT:
        print("Cannot send email: missing SMTP credentials.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = str(subject)
    msg['From'] = str(SMTP_USER)
    msg['To'] = str(to_email)


    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ---------------- MAIN PROCESSING ----------------
async def process_all_incremental_resume():
    df_out = pd.read_parquet(INTERMEDIATE_PARQUET) if os.path.exists(INTERMEDIATE_PARQUET) else df.copy()
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    permanently_failed = []

    normal_batches = []
    i = 0
    while i < len(df_out):
        if pd.notna(df_out.at[i,"Actor of repression"]) and df_out.at[i,"Actor of repression"] != "":
            i += 1
            continue
        batch_summaries = []
        batch_start_idx = i
        while i < len(df_out) and len(batch_summaries) < DEFAULT_BATCH_SIZE:
            summary = str(df_out.at[i,"summary"])
            batch_summaries.append(summary)
            i += 1
        if batch_summaries:
            normal_batches.append([batch_start_idx, batch_summaries, 0])

    async def process_batch(batch_start_idx, batch_summaries, retries):
        async with semaphore:
            results, duration = await extract_batch(batch_summaries)
            for j,res in enumerate(results):
                idx = batch_start_idx+j
                for key in ["Actor of repression","Subject of repression","Mechanism of repression","Type of event"]:
                    df_out.at[idx,key] = res.get(key,"Unknown")
            df_out.to_parquet(INTERMEDIATE_PARQUET,index=False,engine="pyarrow")

    tasks = [process_batch(*b, 0) for b in normal_batches]
    await asyncio.gather(*tasks)
    return df_out

# ---------------- RUN ----------------
df_out = asyncio.run(process_all_incremental_resume())
df_out.to_parquet(OUTPUT_PARQUET,index=False,engine="pyarrow")
df_out.to_csv(OUTPUT_CSV,index=False)

# ---------------- EMAIL SUMMARY ----------------
summary_message = f"Processing complete! Mode: {'MOCK' if MOCK_MODE else 'LIVE'}\nOutput: {OUTPUT_PARQUET}, {OUTPUT_CSV}"
if NOTIFY_EMAIL:
    send_email_notification("WordPress Summary Extraction Completed", summary_message, NOTIFY_EMAIL)

print(summary_message)
