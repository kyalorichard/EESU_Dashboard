#!/usr/bin/env python3
"""
data_preprocessing_google_drive.py
Batch extraction with optional mock/live mode, saves outputs, uploads to Google Drive.

Environment variables expected (via .env or CI):
  OPENAI_API_KEY
  NOTIFY_EMAIL (optional)
  SMTP_USER / SMTP_PASS / SMTP_HOST / SMTP_PORT (optional)
  GDRIVE_JSON           (content or path)
  GDRIVE_FOLDER_ID      (optional)
  MOCK_MODE             (optional, default True in CI)
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

try:
    import tiktoken
except ImportError:
    tiktoken = None

import openai
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

GDRIVE_JSON = os.getenv("GDRIVE_JSON")  # path or raw JSON content
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

INPUT_CSV = os.getenv("INPUT_CSV", "data/raw_data.csv")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "data")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final2.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

MAX_BATCH_TOKENS = int(os.getenv("MAX_BATCH_TOKENS", 10000))
CONCURRENT_BATCHES = int(os.getenv("CONCURRENT_BATCHES", 3))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
SAVE_EVERY_N_BATCHES = int(os.getenv("SAVE_EVERY_N_BATCHES", 2))
TEST_ROWS = int(os.getenv("TEST_ROWS")) if os.getenv("TEST_ROWS") else None
MAX_BATCH_SIZE = None

THEMES_FILE = os.getenv("THEMES_FILE", "data/themes.json")
if not os.path.exists(THEMES_FILE):
    raise FileNotFoundError(f"Themes file missing: {THEMES_FILE}")
with open(THEMES_FILE, "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV missing: {INPUT_CSV}")
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
            return max(1, len(text)//4 + 50)
else:
    def estimate_tokens(text: str) -> int:
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
    return f"""
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

async def mock_extract_batch(batch_summaries):
    await asyncio.sleep(random.uniform(0.1,0.5))
    return [{k: f"{k} Value" for k in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]} for _ in batch_summaries], None

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
            await asyncio.sleep(2**attempt)
    return [{"Actor of repression":"Error","Subject of repression":"Error","Mechanism of repression":"Error","Type of event":"Error"} for _ in batch_summaries], None

def build_batches(df_input, max_tokens=MAX_BATCH_TOKENS, max_rows=None):
    batches = []
    i = 0
    while i < len(df_input):
        if all(df_input.at[i, col] not in [None,""] for col in ["Actor of repression","Subject of repression","Mechanism of repression","Type of event"]):
            i+=1
            continue
        batch_summaries = []
        batch_start_idx = i
        batch_tokens = 0
        while i < len(df_input):
            summary = str(df_input.at[i,"summary"])
            est_tokens = estimate_tokens(summary)
            if batch_tokens + est_tokens > max_tokens and batch_summaries:
                break
            batch_summaries.append(summary)
            batch_tokens += est_tokens
            i += 1
            if max_rows and len(batch_summaries) >= max_rows:
                break
        if batch_summaries:
            batches.append((batch_start_idx,batch_summaries))
    return batches

def send_email(subject, body, to_email):
    if not all([subject, body, to_email, SMTP_USER, SMTP_PASS, SMTP_HOST]):
        print("[send_email] Missing credentials, skipping email")
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
        print(f"[send_email] Email sent to {to_email}")
    except Exception as e:
        print(f"[send_email] Failed: {e}")

# ---------------- GOOGLE DRIVE ----------------
def ensure_service_account_file():
    if GDRIVE_JSON:
        if os.path.exists(GDRIVE_JSON):
            return GDRIVE_JSON
        try:
            parsed = json.loads(GDRIVE_JSON)
            path = os.path.join(OUTPUT_FOLDER, "service-account.json")
            with open(path,"w",encoding="utf-8") as f:
                json.dump(parsed,f,indent=2)
            os.chmod(path,0o600)
            return path
        except Exception as e:
            print(f"[ensure_service_account_file] Failed to write service account: {e}")
            return None
    print("[ensure_service_account_file] No credentials provided")
    return None

def get_drive_service():
    cred_path = ensure_service_account_file()
    if not cred_path or not os.path.exists(cred_path):
        raise FileNotFoundError(f"Service account file missing: {cred_path}")
    creds = service_account.Credentials.from_service_account_file(
        cred_path,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return build("drive","v3",credentials=creds)

def safe_get_drive_service():
    try:
        return get_drive_service()
    except Exception as e:
        print(f"[get_drive_service] Drive service unavailable: {e}")
        return None

def upload_to_drive(local_path, drive_folder_id=GDRIVE_FOLDER_ID):
    if not os.path.exists(local_path):
        print(f"[upload_to_drive] File missing: {local_path}")
        return None
    service = safe_get_drive_service()
    if service is None:
        print("[upload_to_drive] Skipping upload")
        return None
    file_name = os.path.basename(local_path)
    media = MediaFileUpload(local_path,resumable=True)
    meta = {"name":file_name}
    if drive_folder_id:
        meta["parents"] = [drive_folder_id]
    try:
        fid = service.files().create(body=meta,media_body=media,fields="id").execute().get("id")
        print(f"[upload_to_drive] Uploaded {file_name} ID: {fid}")
        return fid
    except Exception as e:
        print(f"[upload_to_drive] Upload failed: {e}")
        return None

# ---------------- MAIN ----------------
async def process_all(mock_mode=False):
    df_out = pd.read_parquet(OUTPUT_PARQUET) if os.path.exists(OUTPUT_PARQUET) else df.copy()
    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    batches = build_batches(df_out,max_rows=MAX_BATCH_SIZE)
    print(f"[process_all] Total batches: {len(batches)}")

    async def process_batch(start_idx, batch_summaries):
        async with semaphore:
            retries = 0
            last_exc = None
            while retries <= MAX_RETRIES:
                try:
                    results,_ = await extract_batch(batch_summaries,mock_mode)
                    break
                except Exception as e:
                    retries+=1
                    last_exc=e
                    print(f"[process_batch] Batch {start_idx} failed attempt {retries}: {e}")
                    await asyncio.sleep(2**retries)
            else:
                permanently_failed.append({"start_idx":start_idx,"error":str(last_exc)})
                results=[{"Actor of repression":"Error","Subject of repression":"Error","Mechanism of repression":"Error","Type of event":"Error"} for _ in batch_summaries]
            for j,res in enumerate(results):
                idx=start_idx+j
                for k in ["Actor of repression","Subject of repression","Mechanism of repression","Type of event"]:
                    df_out.at[idx,k]=res.get(k,"Unknown")
            print(f"[process_batch] Processed batch starting at row {start_idx}, {len(results)} rows")

    for i in range(0,len(batches),CONCURRENT_BATCHES):
        tasks=[process_batch(*b) for b in batches[i:i+CONCURRENT_BATCHES]]
        await asyncio.gather(*tasks)
        if (i//CONCURRENT_BATCHES+1) % SAVE_EVERY_N_BATCHES==0:
            df_out.to_parquet(OUTPUT_PARQUET,index=False,engine="pyarrow")
            print(f"[process_all] Intermediate save after batch group {i//CONCURRENT_BATCHES+1}")
            upload_to_drive(OUTPUT_PARQUET)

    df_out.to_parquet(OUTPUT_PARQUET,index=False,engine="pyarrow")
    df_out.to_csv(OUTPUT_CSV,index=False)
    upload_to_drive(OUTPUT_PARQUET)
    upload_to_drive(OUTPUT_CSV)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE,"w",encoding="utf-8") as f:
            json.dump(permanently_failed,f,indent=2)
        upload_to_drive(PERMANENTLY_FAILED_FILE)

    return df_out

if __name__=="__main__":
    mock_mode_env=os.getenv("MOCK_MODE")
    mock_mode = True if mock_mode_env is None else mock_mode_env.lower() in ("1","true","yes","y")
    df_out = asyncio.run(process_all(mock_mode=mock_mode))
    summary=f"Processing complete! Rows: {len(df_out)} | Mock mode: {mock_mode} | Time: {datetime.utcnow().isoformat()}Z"
    print(summary)
    if NOTIFY_EMAIL:
        send_email("Extraction Completed",summary,NOTIFY_EMAIL)
