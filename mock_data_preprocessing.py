import pandas as pd
import openai
import asyncio
import json
import os
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib
import random
from tqdm.asyncio import tqdm_asyncio

# ---------------- LOAD ENVIRONMENT VARIABLES ----------------
load_dotenv()
OPENAI_API_KEY = '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'  #os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)

# ---------------- FILE PATHS -----------------
INPUT_CSV = "data/output_final.csv"
OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PARQUET = os.path.join(OUTPUT_FOLDER, "output_final.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "output_final.csv")
PERMANENTLY_FAILED_FILE = os.path.join(OUTPUT_FOLDER, "permanently_failed_batches.json")

# ---------------- CONFIGURATION ----------------
MAX_BATCH_TOKENS = 4000
MAX_BATCH_SIZE = 100
CONCURRENT_BATCHES = 5
MAX_RETRIES = 2
TEST_ROWS = 5#None

# ---------------- LOAD THEMES ----------------
THEMES_FILE = "data/themes.json"
with open(THEMES_FILE, "r", encoding="utf-8") as f:
    themes = json.load(f)

ACTOR_THEMES = themes["ACTOR_THEMES"]
SUBJECT_THEMES = themes["SUBJECT_THEMES"]
MECHANISM_THEMES = themes["MECHANISM_THEMES"]
TYPE_THEMES = themes["TYPE_THEMES"]

FIELDS = ["Actor of repression", "Subject of repression", "Mechanism of repression", "Type of event"]

# ---------------- LOAD CSV ----------------
df = pd.read_csv(INPUT_CSV)
if TEST_ROWS:
    df = df.head(TEST_ROWS)

for col in FIELDS:
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
    options = theme_list.get(lang, theme_list["en"])
    return ", ".join([t["label"] for t in options])

def to_comma_separated(item):
    if isinstance(item, list):
        return ", ".join(str(i) for i in item[:3])
    elif isinstance(item, str) and item.strip():
        return item
    else:
        return ""

def pick_random_themes(theme_map, lang, n=3):
    if lang not in theme_map:
        lang = "en"
    options = theme_map[lang]
    return [c["label"] for c in (random.sample(options, n) if len(options) >= n else options)]

def build_prompt(batch_summaries):
    numbered_texts = []
    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary) if summary.strip() else "en"
        except LangDetectException:
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
Extract repression info from each text below. Return a JSON array of objects in the same order.
- Return only valid JSON, do not include explanations or extra text.
- Each object must contain:
{json.dumps({field: "" for field in FIELDS}, indent=4)}
- Use only the provided options for each field, based on the language detected.

Texts:
{numbered_text}
"""
    return prompt

# ---------------- BATCH BUILDER (preserve indices) ----------------
def build_batches(df_input, max_tokens=MAX_BATCH_TOKENS, max_rows=None):
    batches = []
    i = 0
    while i < len(df_input):
        # Skip rows that are fully filled
        if all(df_input.iloc[i][col] not in [None, ""] for col in FIELDS):
            i += 1
            continue

        batch_summaries = []
        batch_indices = []
        batch_tokens = 0

        while i < len(df_input):
            summary = str(df_input.iloc[i]["summary"])
            est_tokens = estimate_tokens(summary)
            if batch_tokens + est_tokens > max_tokens and batch_summaries:
                break
            batch_summaries.append(summary)
            batch_indices.append(df_input.index[i])
            batch_tokens += est_tokens
            i += 1
            if max_rows and len(batch_summaries) >= max_rows:
                break

        if batch_summaries:
            batches.append((batch_indices, batch_summaries))

    return batches

# ---------------- EMAIL NOTIFIER ----------------
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

# ---------------- MOCK EXTRACTOR ----------------
async def mock_extract_batch(batch_summaries, batch_indices=None):
    await asyncio.sleep(random.uniform(0.1, 0.4))
    if batch_indices is not None:
        print(f"[MOCK] Processing rows with indices: {batch_indices}")
    else:
        print(f"[MOCK] Processing batch of size: {len(batch_summaries)}")
    results = []
    for summary in batch_summaries:
        lang = "en"
        result = {
            "Actor of repression": pick_random_themes(ACTOR_THEMES, lang, n=2),
            "Subject of repression": pick_random_themes(SUBJECT_THEMES, lang, n=2),
            "Mechanism of repression": pick_random_themes(MECHANISM_THEMES, lang, n=2),
            "Type of event": pick_random_themes(TYPE_THEMES, lang, n=2),
        }
        for key in FIELDS:
            result[key] = to_comma_separated(result[key])
        results.append(result)
    return results, None

# ---------------- OPENAI EXTRACTOR ----------------
async def extract_batch(batch_summaries, mock_mode=False, batch_indices=None):
    if mock_mode:
        return await mock_extract_batch(batch_summaries, batch_indices=batch_indices)
    if batch_indices is not None:
        print(f"[OPENAI] Processing rows with indices: {batch_indices}")
    else:
        print(f"[OPENAI] Processing batch of size: {len(batch_summaries)}")
    prompt = build_prompt(batch_summaries)
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("[")
                end = content.rfind("]")+1
                data = json.loads(content[start:end])
            for res in data:
                for key in FIELDS:
                    res[key] = to_comma_separated(res.get(key, ""))
            return data, None
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    return [{k: "Error" for k in FIELDS} for _ in batch_summaries], None

# ---------------- MAIN PROCESS ----------------
async def process_all(mock_mode=False):
    if os.path.exists(OUTPUT_PARQUET):
        df_out = pd.read_parquet(OUTPUT_PARQUET)
        print(f"Loaded previous output: {OUTPUT_PARQUET}")
    else:
        df_out = df.copy()

    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    # Only fully blank rows
    rows_to_process = df_out[
        df_out[FIELDS].isna().all(axis=1) | (df_out[FIELDS] == "").all(axis=1)
    ]
    if rows_to_process.empty:
        print("No fully blank rows to process. Nothing to do.")
        return df_out

    batches = build_batches(rows_to_process, max_rows=MAX_BATCH_SIZE)
    print(f"Total batches to process (fully blank rows): {len(batches)}")

    async def process_batch(batch_indices, batch_summaries):
        async with semaphore:
            retries = 0
            last_exception = None
            while retries <= MAX_RETRIES:
                try:
                    results, _ = await extract_batch(batch_summaries, mock_mode, batch_indices=batch_indices)
                    break
                except Exception as exc:
                    retries += 1
                    last_exception = exc
                    print(f"Batch starting at {batch_indices[0]} failed (attempt {retries}): {exc}")
                    await asyncio.sleep(2 ** retries)
            else:
                permanently_failed.append({"start_idx": batch_indices[0], "error": str(last_exception)})
                results = [{k: "Error" for k in FIELDS} for _ in batch_summaries]

            # Map results back and log per row
            for j, res in enumerate(results):
                idx = batch_indices[j]
                row_status = []
                for key in FIELDS:
                    value = res.get(key, "")
                    df_out.loc[idx, key] = value
                    if value == "Error" or not value:
                        row_status.append(f"{key}=ERROR")
                if row_status:
                    print(f"[ROW {idx}] Failed fields: {', '.join(row_status)}")
                else:
                    print(f"[ROW {idx}] Successfully filled all fields.")

    for i in range(0, len(batches), CONCURRENT_BATCHES):
        tasks = [process_batch(*b) for b in batches[i:i + CONCURRENT_BATCHES]]
        await tqdm_asyncio.gather(*tasks, desc="Processing batches", total=len(batches[i:i + CONCURRENT_BATCHES]))
        # Save intermediate results
        df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
        df_out.to_csv(OUTPUT_CSV, index=False)

    # Final save
    df_out.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    df_out.to_csv(OUTPUT_CSV, index=False)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(permanently_failed, f, indent=2)

    print(f"Processing complete! Fully blank rows processed: {len(rows_to_process)}")
    return df_out

# ---------------- RUN SCRIPT ----------------
if __name__ == "__main__":
    mock_mode = False  # Set True for testing

    # Pre-run summary
    if os.path.exists(OUTPUT_PARQUET):
        df_prev = pd.read_parquet(OUTPUT_PARQUET)
    else:
        df_prev = pd.read_csv(INPUT_CSV)
    total_rows = len(df_prev)
    blank_rows = df_prev[df_prev[FIELDS].isna().all(axis=1) | (df_prev[FIELDS] == "").all(axis=1)]
    skipped_rows = total_rows - len(blank_rows)
    print(f"Total rows: {total_rows} | Fully blank rows to process: {len(blank_rows)} | Skipped rows: {skipped_rows}")

    # Run extraction
    df_out = asyncio.run(process_all(mock_mode=mock_mode))

    summary_message = (
        f"Processing complete! Total rows: {len(df_out)} | "
        f"Fully blank rows processed: {len(blank_rows)} | "
        f"Skipped rows: {skipped_rows} | Mock mode: {mock_mode}"
    )
    print(summary_message)

    if NOTIFY_EMAIL:
        send_email("Extraction Completed", summary_message, NOTIFY_EMAIL)
