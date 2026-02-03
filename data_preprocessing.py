import pandas as pd
import openai
import asyncio
import json
import os
import re
import random
import smtplib
import hashlib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# ===================== ENV =====================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")

# ===================== PATHS =====================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_PATTERN = re.compile(r"raw_data_(\d{4}-\d{2}-\d{2})\.csv")
HASH_FILE = DATA_DIR / "processed_hashes.json"

OUTPUT_PARQUET = DATA_DIR / "output_final.parquet"
OUTPUT_CSV = DATA_DIR / "output_final.csv"

# ===================== CONFIG =====================
MAX_BATCH_TOKENS = 10_000
CONCURRENT_BATCHES = 3
MOCK_MODE = True  # Set False for real OpenAI processing

FIELDS = [
    "Actor of repression",
    "Subject of repression",
    "Mechanism of repression",
    "Type of event",
]

# ===================== LOAD THEMES =====================
with open(DATA_DIR / "themes.json", "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

# ===================== EMAIL =====================
def send_html_email(subject, text_body, html_body):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        print("Email skipped: missing configuration.")
        return

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_EMAIL
    msg["Subject"] = subject
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("Notification email sent.")
    except Exception as e:
        print(f"Email failed: {e}")

# ===================== CSV DISCOVERY =====================
def find_latest_csv():
    candidates = []
    for f in DATA_DIR.glob("raw_data_*.csv"):
        match = RAW_PATTERN.match(f.name)
        if match:
            candidates.append((datetime.fromisoformat(match.group(1)), f))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

# ===================== TEXT COLUMN DETECTION =====================
def detect_text_column(df):
    env_col = os.getenv("TEXT_COLUMN")
    if env_col and env_col in df.columns:
        print(f"Using text column from ENV: {env_col}")
        return env_col

    normalized = {col.lower(): col for col in df.columns}
    preferred = ["summary", "event_summary", "description", "text", "content"]
    for key in preferred:
        if key in normalized:
            actual = normalized[key]
            print(f"Using text column: {actual}")
            return actual

    raise RuntimeError(
        f"No valid text column found. Expected one of {preferred}. "
        f"Found columns: {list(df.columns)}"
    )

# ===================== TOKEN ESTIMATION =====================
def estimate_tokens(text):
    return max(1, len(str(text)) // 4 + 50)

# ===================== HASH HELPERS =====================
def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_processed_hashes():
    if HASH_FILE.exists():
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_hashes(hashes):
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, indent=2)

# ===================== MOCK EXTRACTOR =====================
async def mock_extract(batch):
    await asyncio.sleep(random.uniform(0.1, 0.3))
    return [{
        FIELDS[0]: ", ".join(t["label"] for t in random.sample(ACTOR_THEMES["en"], 2)),
        FIELDS[1]: ", ".join(t["label"] for t in random.sample(SUBJECT_THEMES["en"], 2)),
        FIELDS[2]: ", ".join(t["label"] for t in random.sample(MECHANISM_THEMES["en"], 2)),
        FIELDS[3]: ", ".join(t["label"] for t in random.sample(TYPE_THEMES["en"], 2)),
    } for _ in batch]

# ===================== BATCH BUILDER =====================
def build_batches(df, text_col):
    batches = []
    i = 0
    while i < len(df):
        batch = []
        tokens = 0
        start = i
        while i < len(df):
            text = str(df.iloc[i][text_col])
            t = estimate_tokens(text)
            if tokens + t > MAX_BATCH_TOKENS and batch:
                break
            batch.append(text)
            tokens += t
            i += 1
        batches.append((start, batch))
    return batches

# ===================== MAIN PROCESS =====================
async def process():
    latest_csv = find_latest_csv()
    if not latest_csv:
        print("No CSV file found.")
        return

    df_raw = pd.read_csv(latest_csv)
    text_col = detect_text_column(df_raw)

    processed_hashes = load_processed_hashes()

    # Compute hash for each row
    df_raw["__hash"] = df_raw[text_col].astype(str).apply(compute_hash)
    df_new = df_raw[~df_raw["__hash"].isin(processed_hashes)].copy()

    if df_new.empty:
        print("No new rows to process. Exiting without sending email.")
        return

    for field in FIELDS:
        if field not in df_new.columns:
            df_new[field] = ""

    batches = build_batches(df_new, text_col)
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    async def process_batch(start, texts):
        async with semaphore:
            results = await mock_extract(texts) if MOCK_MODE else []
            for i, res in enumerate(results):
                for f in FIELDS:
                    df_new.loc[df_new.index[start + i], f] = res[f]

    await tqdm_asyncio.gather(*(process_batch(s, b) for s, b in batches))

    # Merge old and new
    if OUTPUT_PARQUET.exists():
        df_existing = pd.read_parquet(OUTPUT_PARQUET)
        df_final = pd.concat([df_existing, df_new.drop(columns="__hash")], ignore_index=True)
    else:
        df_final = df_new.drop(columns="__hash")

    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    df_final.to_csv(OUTPUT_CSV, index=False)

    # Save processed hashes
    processed_hashes.update(df_new["__hash"])
    save_processed_hashes(processed_hashes)

    # Send professional HTML email
    subject = "EESU Data Processing Update"
    text_body = f"""
New data processed successfully.

Source file: {latest_csv.name}
New records: {len(df_new)}
Total records: {len(df_final)}
Processed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;background:#f4f6f8;padding:20px">
<div style="background:#fff;padding:20px;border-radius:6px;max-width:600px">
<h2 style="color:#2e7d32;">✅ EESU Data Processing Complete</h2>
<p>The latest dataset has been processed successfully.</p>
<table cellpadding="6" cellspacing="0">
<tr><td><b>Source file</b></td><td>{latest_csv.name}</td></tr>
<tr><td><b>New records</b></td><td>{len(df_new)}</td></tr>
<tr><td><b>Total records</b></td><td>{len(df_final)}</td></tr>
<tr><td><b>Processed at</b></td><td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</td></tr>
</table>
<p>No action is required.</p>
<hr>
<p style="font-size:12px;color:#777">Automated notification – EESU Data Pipeline</p>
</div>
</body>
</html>
"""
    send_html_email(subject, text_body.strip(), html_body)

# ===================== RUN =====================
if __name__ == "__main__":
    asyncio.run(process())
