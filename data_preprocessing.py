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

# ---------------- CONFIG ----------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_CSV = "data/raw_data.csv"
SUMMARY_COL = "summary"
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
INTERMEDIATE_PARQUET = os.path.join(OUTPUT_FOLDER, "intermediate_output.parquet")
LOG_FILE = os.path.join(OUTPUT_FOLDER, "batch_processing_log.csv")
FAILED_BATCHES_FILE = os.path.join(OUTPUT_FOLDER, "failed_batches.json")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

MAX_BATCH_TOKENS = 10000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 2
DEFAULT_BATCH_SIZE = 2
SAVE_EVERY_N_BATCHES = 2
MAX_RETRIES = 2
MODEL = "gpt-5-mini"
LATENCY_TARGET = 5
LATENCY_TOLERANCE = 2
ADJUST_FACTOR = 0.2
CONCURRENT_BATCHES = 3
COST_PER_1K_TOKENS = 0.003

TEST_ROWS = 3  # set None for full dataset

# ---------------- MOCK MODE CONFIG ----------------
MOCK_MODE = True
MOCK_CONFIG = {
    "quota_error_batches": 1,        # First N batches simulate quota errors
    "malformed_json_prob": 0.3,      # Probability a batch returns malformed JSON
    "missing_keys_prob": 0.2,        # Probability some keys are missing in a batch
    "latency_range": (0.5, 2.0),     # Simulated API response time in seconds
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

    # Simulate quota error for first N batches
    if mock_extract_batch.call_count < MOCK_CONFIG["quota_error_batches"]:
        mock_extract_batch.call_count += 1
        mock_extract_batch.quota_error_batches += 1
        raise Exception("You exceeded your current quota (simulated).")

    batch_data = []
    missing_keys_in_batch = False
    malformed_in_batch = False

    for summary in batch_summaries:
        r = {}
        # Possibly omit some keys
        if random.random() < MOCK_CONFIG["missing_keys_prob"]:
            keys_to_include = random.sample(MOCK_CONFIG["fixed_keys"], k=random.randint(1, len(MOCK_CONFIG["fixed_keys"])))
            missing_keys_in_batch = True
        else:
            keys_to_include = MOCK_CONFIG["fixed_keys"]

        for key in keys_to_include:
            r[key] = f"{key} Value"

        # Possibly make malformed
        if random.random() < MOCK_CONFIG["malformed_json_prob"]:
            batch_data.append(json.dumps(r)[:-1])  # remove last char to simulate malformed
            malformed_in_batch = True
        else:
            batch_data.append(r)

    # Update issue counters
    if malformed_in_batch:
        mock_extract_batch.malformed_json_batches += 1
    if missing_keys_in_batch:
        mock_extract_batch.missing_keys_batches += 1

    mock_extract_batch.call_count += 1

    # Combine into simulated API string
    response_content_list = []
    for r in batch_data:
        if isinstance(r, dict):
            response_content_list.append(json.dumps(r))
        else:
            response_content_list.append(r)
    response_content = "[" + ",".join(response_content_list) + "]"

    # Simulate latency
    await asyncio.sleep(random.uniform(*MOCK_CONFIG["latency_range"]))

    # Attempt to parse; fallback will handle malformed JSON
    try:
        parsed_data = json.loads(response_content)
    except:
        parsed_data = response_content

    return parsed_data, random.uniform(*MOCK_CONFIG["latency_range"])

# ---------------- EXTRACT BATCH ----------------
async def extract_batch(batch_summaries):
    if MOCK_MODE:
        return await mock_extract_batch(batch_summaries)

    prompt = build_prompt(batch_summaries)
    for attempt in range(MAX_RETRIES):
        try:
            start_time = datetime.now()
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            duration = (datetime.now() - start_time).total_seconds()
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise

            for item in data:
                for key in ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]:
                    if key not in item:
                        item[key] = "Unknown"
            return data, duration

        except Exception as e:
            print(f"Error: {e}. Retrying ({attempt+1}/{MAX_RETRIES})...")
            await asyncio.sleep(2 ** attempt)

    return [{"Actor of repression": "Error",
             "Subject of repression": "Error",
             "Mechanism of repression": "Error",
             "Type of event": "Error"} for _ in batch_summaries], None

# ---------------- EMAIL FUNCTION ----------------
def send_email_notification(subject, body, to_email):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ---------------- MAIN PROCESSING ----------------
async def process_all_incremental_resume():
    if os.path.exists(INTERMEDIATE_PARQUET):
        df_out = pd.read_parquet(INTERMEDIATE_PARQUET)
        if len(df) > len(df_out):
            new_rows = df.iloc[len(df_out):].copy()
            df_out = pd.concat([df_out, new_rows], ignore_index=True)
    else:
        df_out = df.copy()

    failed_batches = []
    if os.path.exists(FAILED_BATCHES_FILE):
        with open(FAILED_BATCHES_FILE, "r", encoding="utf-8") as f:
            failed_batches = json.load(f)

    batch_size = DEFAULT_BATCH_SIZE
    total_tokens_processed = 0
    total_cost = 0.0
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    permanently_failed = []

    # Prepare normal batches
    i = 0
    normal_batches = []
    while i < len(df_out):
        if df_out.at[i,"Actor of repression"]:
            i += 1
            continue
        batch_summaries = []
        batch_tokens = 0
        batch_start_idx = i
        while i < len(df_out) and len(batch_summaries) < batch_size:
            if df_out.at[i,"Actor of repression"]:
                i += 1
                continue
            summary = str(df_out.at[i,SUMMARY_COL])
            tokens = estimate_tokens(summary)
            if batch_tokens + tokens > MAX_BATCH_TOKENS:
                break
            batch_summaries.append(summary)
            batch_tokens += tokens
            i += 1
        if batch_summaries:
            normal_batches.append([batch_start_idx, batch_summaries, batch_tokens, 0])

    for fb in failed_batches:
        fb.setdefault("retries",0)
    batches_to_process = failed_batches + normal_batches

    async def process_batch(batch_start_idx, batch_summaries, batch_tokens, retries):
        nonlocal batch_size, total_tokens_processed, total_cost, df_out, permanently_failed
        async with semaphore:
            try:
                results, duration = await extract_batch(batch_summaries)
            except Exception as e:
                print(f"Batch {batch_start_idx} error: {e}. Retrying ({retries+1}/{MAX_RETRIES})...")
                retries += 1
                if retries > MAX_RETRIES:
                    permanently_failed.append({
                        "start_idx": batch_start_idx,
                        "summaries": batch_summaries,
                        "tokens": batch_tokens,
                        "retries": retries
                    })
                    return 0
                else:
                    return (batch_start_idx, batch_summaries, batch_tokens, retries)

        # --- PATCH: Convert any string or malformed results into dicts ---
        safe_results = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    safe_results.append(r)
                else:
                    safe_results.append({
                        "Actor of repression": "Error",
                        "Subject of repression": "Error",
                        "Mechanism of repression": "Error",
                        "Type of event": "Error"
                    })
        else:
            # entire batch is malformed string
            safe_results = [{
                "Actor of repression": "Error",
                "Subject of repression": "Error",
                "Mechanism of repression": "Error",
                "Type of event": "Error"
            } for _ in batch_summaries]

        if all(r.get("Actor of repression")=="Error" for r in safe_results):
            retries += 1
            if retries > MAX_RETRIES:
                permanently_failed.append({
                    "start_idx": batch_start_idx,
                    "summaries": batch_summaries,
                    "tokens": batch_tokens,
                    "retries": retries
                })
                return 0
            else:
                return (batch_start_idx, batch_summaries, batch_tokens, retries)
        else:
            for j,res in enumerate(safe_results):
                idx = batch_start_idx+j
                df_out.at[idx,"Actor of repression"] = res.get("Actor of repression","Unknown")
                df_out.at[idx,"Subject of repression"] = res.get("Subject of repression","Unknown")
                df_out.at[idx,"Mechanism of repression"] = res.get("Mechanism of repression","Unknown")
                df_out.at[idx,"Type of event"] = res.get("Type of event","Unknown")

        log_entry = {"Batch Start Index": batch_start_idx,"Batch Size":len(batch_summaries),
                     "Estimated Tokens":batch_tokens,"API Duration(s)":duration,"Timestamp":datetime.now().isoformat()}
        pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
        df_out.to_parquet(INTERMEDIATE_PARQUET,index=False,engine="pyarrow")

        if duration:
            if duration < LATENCY_TARGET - LATENCY_TOLERANCE:
                batch_size = min(int(batch_size*(1+ADJUST_FACTOR)), MAX_BATCH_SIZE)
            elif duration > LATENCY_TARGET + LATENCY_TOLERANCE:
                batch_size = max(int(batch_size*(1-ADJUST_FACTOR)), MIN_BATCH_SIZE)

        total_tokens_processed += batch_tokens
        total_cost = (total_tokens_processed/1000)*COST_PER_1K_TOKENS
        return 0

    # ---------------- MAIN LOOP ----------------
    pending_batches = batches_to_process
    pbar = tqdm_asyncio(total=len(pending_batches), desc="Processing batches")
    while pending_batches:
        chunk = pending_batches[:CONCURRENT_BATCHES]
        tasks = [process_batch(*b) for b in chunk]
        results = await asyncio.gather(*tasks)
        pbar.update(len(chunk))
        pending_batches = [r for r in results if r] + pending_batches[CONCURRENT_BATCHES:]
    pbar.close()

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE,"w",encoding="utf-8") as f:
            json.dump(permanently_failed,f,ensure_ascii=False,indent=2)

    remaining_failed = [b for b in pending_batches if b]
    with open(FAILED_BATCHES_FILE,"w",encoding="utf-8") as f:
        json.dump(remaining_failed,f,ensure_ascii=False,indent=2)

    # ---------------- MOCK STATS ----------------
    mock_stats = None
    if MOCK_MODE:
        mock_stats = {
            "total_batches": getattr(mock_extract_batch, "call_count", 0),
            "quota_errors": getattr(mock_extract_batch, "quota_error_batches", 0),
            "malformed_json": getattr(mock_extract_batch, "malformed_json_batches", 0),
            "missing_keys": getattr(mock_extract_batch, "missing_keys_batches", 0)
        }

    return df_out, total_tokens_processed, total_cost, len(remaining_failed), len(permanently_failed), mock_stats

# ---------------- RUN ----------------
df_out, total_tokens, total_cost, remaining_failed_count, permanently_failed_count, mock_stats = asyncio.run(process_all_incremental_resume())

df_out.to_parquet(OUTPUT_PARQUET,index=False,engine="pyarrow")
df_out.to_csv(OUTPUT_CSV,index=False)

summary_message = (
    f"✅ Processing Complete!\n"
    f"Final Parquet: {OUTPUT_PARQUET}\n"
    f"Final CSV: {OUTPUT_CSV}\n"
    f"Intermediate Parquet: {INTERMEDIATE_PARQUET}\n"
    f"Batch log: {LOG_FILE}\n"
    f"Remaining failed batches: {remaining_failed_count}\n"
    f"Permanently failed batches: {permanently_failed_count}\n"
    f"Total estimated tokens processed: {total_tokens}\n"
    f"Total estimated OpenAI cost: ${total_cost:.4f}\n"
)

if MOCK_MODE and mock_stats:
    summary_message += (
        f"\n--- MOCK RUN SUMMARY ---\n"
        f"Total batches processed: {mock_stats['total_batches']}\n"
        f"Simulated quota errors: {mock_stats['quota_errors']}\n"
        f"Batches with malformed JSON: {mock_stats['malformed_json']}\n"
        f"Batches with missing keys: {mock_stats['missing_keys']}\n"
        f"------------------------\n"
    )

recipient_email = os.getenv("NOTIFY_EMAIL")
if recipient_email:
    send_email_notification(
        subject="WordPress Summary Extraction Completed",
        body=summary_message,
        to_email=recipient_email
    )

print(summary_message)
