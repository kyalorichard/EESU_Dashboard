import pandas as pd
import openai
import asyncio
import json
import os
from datetime import datetime
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib
import random

# ---------------- LOAD ENV ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
######

# ---------------- FILE PATHS -----------------
INPUT_CSV = "data/raw_data.csv"
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

# ---------------- CONFIG ----------------
MAX_BATCH_TOKENS = 10000
MAX_BATCH_SIZE = None  # Max rows per batch
CONCURRENT_BATCHES = 3
MAX_RETRIES = 2
SAVE_EVERY_N_BATCHES = 2
TEST_ROWS = None  # Use for partial test runs

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
    encoding = tiktoken.encoding_for_model("gpt-5-mini")
    def estimate_tokens(text):
        return len(encoding.encode(text)) + 50
except ImportError:
    def estimate_tokens(text):
        return max(1, len(text)//4 + 50)

# ---------------- HELPERS ----------------
def format_theme_options(theme_list, lang):
    return ", ".join([f'{t["label"]} ({t["definition"]})' for t in theme_list.get(lang, theme_list["en"])])

def build_prompt(batch_summaries):
    numbered_texts = []
    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary) if summary.strip() else "en"
        except LangDetectException:
            lang = "en"
        if lang not in ACTOR_THEMES:
            lang = "en"
        numbered_texts.append(
            f"{idx+1}. Summary: {summary}\n"
            f"Language: {lang}\n"
            f"Actor options: {format_theme_options(ACTOR_THEMES, lang)}\n"
            f"Subject options: {format_theme_options(SUBJECT_THEMES, lang)}\n"
            f"Mechanism options: {format_theme_options(MECHANISM_THEMES, lang)}\n"
            f"Type options: {format_theme_options(TYPE_THEMES, lang)}"
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

# ---------------- MOCK EXTRACT ----------------
async def mock_extract_batch(batch_summaries):
    await asyncio.sleep(random.uniform(0.1, 0.5))
    return [{k: f"{k} Value" for k in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]} for _ in batch_summaries], None

# ---------------- LIVE EXTRACT ----------------
async def extract_batch(batch_summaries, mock_mode=False):
    if mock_mode:
        return await mock_extract_batch(batch_summaries)

    prompt = build_prompt(batch_summaries)
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data, None
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    # fallback
    return [{"Actor of repression": "Error",
             "Subject of repression": "Error",
             "Mechanism of repression": "Error",
             "Type of event": "Error"} for _ in batch_summaries], None

# ---------------- BATCHING ----------------
def build_batches(df_input, max_tokens=MAX_BATCH_TOKENS, max_rows=None):
    batches = []
    i = 0
    while i < len(df_input):
        # skip already processed
        if all(df_input.at[i, col] not in [None, ""] for col in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]):
            i += 1
            continue

        batch_summaries = []
        batch_start_idx = i
        batch_tokens = 0
        while i < len(df_input):
            summary = str(df_input.at[i, "summary"])
            est_tokens = estimate_tokens(summary)
            if batch_tokens + est_tokens > max_tokens and batch_summaries:
                break
            batch_summaries.append(summary)
            batch_tokens += est_tokens
            i += 1
            if max_rows and len(batch_summaries) >= max_rows:
                break
        if batch_summaries:
            batches.append((batch_start_idx, batch_summaries))
    return batches

# ---------------- EMAIL ----------------
def send_email(subject, body, to_email):
    if not all([subject, body, to_email, SMTP_USER, SMTP_PASS, SMTP_HOST]):
        print("Email not sent: Missing credentials or recipient.")
        return
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = str(subject)
        msg['From'] = str(SMTP_USER)
        msg['To'] = str(to_email)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"Email successfully sent to {to_email}")
    except Exception as e:
        print(f"Email failed: {e}")

# ---------------- MAIN PROCESS ----------------
async def process_all(mock_mode=False):
    # Load previous output if exists
    if os.path.exists(OUTPUT_PARQUET):
        df_out = pd.read_parquet(OUTPUT_PARQUET)
        print(f"Loaded previous output: {OUTPUT_PARQUET}")
    else:
        df_out = df.copy()

    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    batches = build_batches(df_out, max_rows=MAX_BATCH_SIZE)
    print(f"Total batches to process: {len(batches)}")

    async def process_batch(batch_start_idx, batch_summaries):
        async with semaphore:
            retries = 0
            last_exception = None
            while retries <= MAX_RETRIES:
                try:
                    results, _ = await extract_batch(batch_summaries, mock_mode)
                    break
                except Exception as exc:
                    retries += 1
                    last_exception = exc
                    print(f"Batch starting at {batch_start_idx} failed (attempt {retries}): {exc}")
                    await asyncio.sleep(2 ** retries)
            else:
                permanently_failed.append({"start_idx": batch_start_idx, "error": str(last_exception)})
                results = [{"Actor of repression": "Error",
                            "Subject of repression": "Error",
                            "Mechanism of repression": "Error",
                            "Type of event": "Error"} for _ in batch_summaries]

            for j, res in enumerate(results):
                idx = batch_start_idx + j
                for key in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
                    df_out.at[idx, key] = res.get(key, "Unknown")
            print(f"Processed batch starting at row {batch_start_idx}, {len(results)} rows")

    for i in range(0, len(batches), CONCURRENT_BATCHES):
        tasks = [process_batch(*b) for b in batches[i:i+CONCURRENT_BATCHES]]
        await asyncio.gather(*tasks)

        if (i // CONCURRENT_BATCHES + 1) % SAVE_EVERY_N_BATCHES == 0:
            df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
            print(f"Intermediate save after batch group {i // CONCURRENT_BATCHES + 1}")

    df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    df_out.to_csv(OUTPUT_CSV, index=False)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(permanently_failed, f, indent=2)

    return df_out

# ---------------- RUN ----------------
if __name__ == "__main__":
    mock_mode = True  # Change to False to use live GPT API
    df_out = asyncio.run(process_all(mock_mode=mock_mode))

    summary_message = f"Processing complete! Total rows: {len(df_out)} | Mock mode: {mock_mode}"
    print(summary_message)
    if NOTIFY_EMAIL:
        send_email("Extraction Completed", summary_message, NOTIFY_EMAIL)
