#!/usr/bin/env python3
"""
data_preprocessing_google_drive.py
Fully merged script: performs batch extraction (mock or live), saves outputs,
and uploads results to Google Drive using a service account.

Usage:
 - Ensure env vars are set (via .env or CI):
    OPENAI_API_KEY
    NOTIFY_EMAIL (optional)
    SMTP_USER / SMTP_PASS / SMTP_HOST / SMTP_PORT (optional)
    SERVICE_ACCOUNT_FILE (optional)  OR GDRIVE_SERVICE_ACCOUNT_JSON (content or path)
    GDRIVE_FOLDER_ID (optional)
 - Run: python data_preprocessing_google_drive.py
"""

import asyncio
import json
import os
import random
import smtplib
from datetime import datetime
from email.message import EmailMessage

import pandas as pd
from dotenv import load_dotenv
from langdetect import LangDetectException, detect

# Optional token estimator
try:
    import tiktoken
except Exception:
    tiktoken = None

# OpenAI client
import openai

# Google Drive client libs
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ---------------- LOAD ENV ----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)

# Drive configuration:
# Two supported ways to provide credentials:
# 1) Point SERVICE_ACCOUNT_FILE to an existing JSON file path on the runner
# 2) Provide GDRIVE_SERVICE_ACCOUNT_JSON env var which contains either:
#    - a path to the JSON file, OR
#    - the raw JSON content (e.g. via GitHub secret). The script will write it to service-account.json

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service-account.json")
GDRIVE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # path or raw JSON content
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# ---------------- FILE PATHS -----------------
INPUT_CSV = os.getenv("INPUT_CSV", "data/raw_data.csv")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "data")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final2.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

# ---------------- CONFIG ----------------
MAX_BATCH_TOKENS = int(os.getenv("MAX_BATCH_TOKENS", 10000))
MAX_BATCH_SIZE = None
CONCURRENT_BATCHES = int(os.getenv("CONCURRENT_BATCHES", 3))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
SAVE_EVERY_N_BATCHES = int(os.getenv("SAVE_EVERY_N_BATCHES", 2))
TEST_ROWS = int(os.getenv("TEST_ROWS")) if os.getenv("TEST_ROWS") else None

# ---------------- LOAD THEMES ----------------
THEMES_FILE = os.getenv("THEMES_FILE", "data/themes.json")
if not os.path.exists(THEMES_FILE):
    raise FileNotFoundError(f"Required themes file not found: {THEMES_FILE}")
with open(THEMES_FILE, "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

# ---------------- LOAD CSV ----------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
if TEST_ROWS:
    df = df.head(TEST_ROWS)

for col in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
    if col not in df.columns:
        df[col] = ""

# ---------------- TOKEN ESTIMATION ----------------
if tiktoken:
    try:
        encoding = tiktoken.encoding_for_model("gpt-5-mini")
        def estimate_tokens(text: str) -> int:
            return len(encoding.encode(text)) + 50
    except Exception:
        def estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4 + 50)
else:
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4 + 50)

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
            print(f"[extract_batch] Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    # Fallback: return error placeholders
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
        print("[send_email] Email not sent: Missing credentials or recipient.")
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
        print(f"[send_email] Email successfully sent to {to_email}")
    except Exception as e:
        print(f"[send_email] Email failed: {e}")

# ---------------- GOOGLE DRIVE CREDENTIALS HANDLING ----------------
def ensure_service_account_file():
    """
    Ensure SERVICE_ACCOUNT_FILE exists. Accepts either:
      - GDRIVE_SERVICE_ACCOUNT_JSON env var containing a path to JSON file
      - GDRIVE_SERVICE_ACCOUNT_JSON env var containing raw JSON content
      - SERVICE_ACCOUNT_FILE already present on disk
    Returns the path to the credentials file, or None if not available.
    """
    # If SERVICE_ACCOUNT_FILE already exists, use it
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        return SERVICE_ACCOUNT_FILE

    # If GDRIVE_SERVICE_ACCOUNT_JSON is set, interpret it
    if GDRIVE_SERVICE_ACCOUNT_JSON:
        # If it points to an existing file path, use that
        if os.path.exists(GDRIVE_SERVICE_ACCOUNT_JSON):
            return GDRIVE_SERVICE_ACCOUNT_JSON

        # Otherwise treat it as raw JSON content and write to SERVICE_ACCOUNT_FILE
        try:
            # Validate JSON by parsing
            parsed = json.loads(GDRIVE_SERVICE_ACCOUNT_JSON)
            with open(SERVICE_ACCOUNT_FILE, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            os.chmod(SERVICE_ACCOUNT_FILE, 0o600)
            return SERVICE_ACCOUNT_FILE
        except json.JSONDecodeError:
            print("[ensure_service_account_file] GDRIVE_SERVICE_ACCOUNT_JSON does not contain valid JSON.")
            return None
        except Exception as exc:
            print(f"[ensure_service_account_file] Failed to write service account file: {exc}")
            return None

    # No credential info available
    print("[ensure_service_account_file] No service account credentials provided (SERVICE_ACCOUNT_FILE or GDRIVE_SERVICE_ACCOUNT_JSON).")
    return None

def get_drive_service():
    cred_path = ensure_service_account_file()
    if not cred_path or not os.path.exists(cred_path):
        raise FileNotFoundError(f"Service account file not found: {cred_path}")

    creds = service_account.Credentials.from_service_account_file(
        cred_path,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return build('drive', 'v3', credentials=creds)

def safe_get_drive_service():
    try:
        return get_drive_service()
    except Exception as e:
        print(f"[get_drive_service] Google Drive service unavailable: {e}")
        return None

# ---------------- GOOGLE DRIVE UPLOAD ----------------
def upload_to_drive(local_path, drive_folder_id=GDRIVE_FOLDER_ID):
    """
    Uploads local_path to Google Drive. If credentials are missing the function logs and returns None.
    Returns the uploaded file ID on success, None otherwise.
    """
    if not os.path.exists(local_path):
        print(f"[upload_to_drive] Local file not found, skipping upload: {local_path}")
        return None

    service = safe_get_drive_service()
    if service is None:
        print("[upload_to_drive] Skipping upload because Drive service is unavailable.")
        return None

    file_name = os.path.basename(local_path)
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {"name": file_name}
    if drive_folder_id:
        file_metadata["parents"] = [drive_folder_id]

    try:
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        fid = uploaded_file.get("id")
        print(f"[upload_to_drive] Uploaded {file_name} to Google Drive with ID: {fid}")
        return fid
    except Exception as exc:
        print(f"[upload_to_drive] Upload failed for {local_path}: {exc}")
        return None

# ---------------- MAIN PROCESS ----------------
async def process_all(mock_mode=False):
    # Load previous output if exists
    if os.path.exists(OUTPUT_PARQUET):
        df_out = pd.read_parquet(OUTPUT_PARQUET)
        print(f"[process_all] Loaded previous output: {OUTPUT_PARQUET}")
    else:
        df_out = df.copy()

    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    batches = build_batches(df_out, max_rows=MAX_BATCH_SIZE)
    print(f"[process_all] Total batches to process: {len(batches)}")

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
                    print(f"[process_batch] Batch starting at {batch_start_idx} failed (attempt {retries}): {exc}")
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
            print(f"[process_batch] Processed batch starting at row {batch_start_idx}, {len(results)} rows")

    for i in range(0, len(batches), CONCURRENT_BATCHES):
        tasks = [process_batch(*b) for b in batches[i:i+CONCURRENT_BATCHES]]
        await asyncio.gather(*tasks)

        if (i // CONCURRENT_BATCHES + 1) % SAVE_EVERY_N_BATCHES == 0:
            df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
            print(f"[process_all] Intermediate save after batch group {i // CONCURRENT_BATCHES + 1}")
            # Try upload but don't fail the whole job if upload fails
            upload_to_drive(OUTPUT_PARQUET)

    # Save final outputs
    df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    df_out.to_csv(OUTPUT_CSV, index=False)

    # Upload final outputs (best-effort)
    upload_to_drive(OUTPUT_PARQUET)
    upload_to_drive(OUTPUT_CSV)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(permanently_failed, f, indent=2)
        upload_to_drive(PERMANENTLY_FAILED_FILE)

    return df_out

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Choose mock_mode via env var for CI convenience
    mock_mode_env = os.getenv("MOCK_MODE")
    if mock_mode_env is not None:
        mock_mode = mock_mode_env.lower() in ("1", "true", "yes", "y")
    else:
        mock_mode = True  # default to True to avoid accidental API usage in CI; override in env when needed

    # Run processing
    df_out = asyncio.run(process_all(mock_mode=mock_mode))

    summary_message = f"Processing complete! Total rows: {len(df_out)} | Mock mode: {mock_mode} | Time: {datetime.utcnow().isoformat()}Z"
    print(summary_message)
    if NOTIFY_EMAIL:
        send_email("Extraction Completed", summary_message, NOTIFY_EMAIL)
