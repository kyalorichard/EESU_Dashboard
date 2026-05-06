#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import posixpath
import random
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import openai
import pandas as pd
import paramiko
from dotenv import load_dotenv
from langdetect import LangDetectException, detect
from tqdm.asyncio import tqdm_asyncio

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

# --- SFTP CONFIG ---
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT") or 22)
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR", "exports")

# --- SMTP / NOTIFICATIONS ---
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# --- LOCAL PATHS ---
OUTPUT_FOLDER = BASE_DIR / "exports"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_CSV = OUTPUT_FOLDER / "output_final.csv"
THEMES_FILE = OUTPUT_FOLDER / "themes.json"
OUTPUT_CSV = OUTPUT_FOLDER / "output_final.csv"
OUTPUT_PARQUET = OUTPUT_FOLDER / "output_final.parquet"
PERMANENTLY_FAILED_FILE = OUTPUT_FOLDER / "permanently_failed_batches.json"

# --- PROCESSING CONFIG ---
MAX_BATCH_TOKENS = 4000
MAX_BATCH_SIZE = 100
CONCURRENT_BATCHES = 5
MAX_RETRIES = 2
TEST_ROWS = None

FIELDS = [
    "Actor of repression",
    "Subject of repression",
    "Mechanism of repression",
    "Type of event",
]

CORE_TEXT_COLUMNS = [
    "post_title",
    "summary",
    "creation_date",
    "alert-country",
    "alert-impact",
    "alert-type",
    "enabling-principle",
] + FIELDS


# ==========================================================
# TOKEN ESTIMATION
# ==========================================================
try:
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-5-mini")

    def estimate_tokens(text: str) -> int:
        return len(encoding.encode(text or "")) + 50

except ImportError:
    def estimate_tokens(text: str) -> int:
        text = text or ""
        return max(1, len(text) // 4 + 50)


# ==========================================================
# DATAFRAME SAFETY HELPERS
# ==========================================================
def safe_str(value) -> str:
    """
    Convert any scalar value to a clean string.
    Prevents pandas dtype failures when assigning labels into previously numeric columns.
    """
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "null", "n/a", "na"}:
        return ""
    if value == "Error":
        return ""
    return value


def normalize_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make output dataframe safe for text classification updates.
    Critical fix:
    - classification columns are forced to object/string columns
    - blank-like values are normalized to empty strings
    - fully blank rows are removed
    """
    df = df.copy()

    for col in FIELDS:
        if col not in df.columns:
            df[col] = ""

    # Force target columns to object before async assignment
    for col in FIELDS:
        df[col] = df[col].astype(object).map(safe_str)

    # Keep key text columns safe too
    for col in CORE_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object).map(safe_str)

    # Remove completely blank rows
    mask = df.apply(lambda row: any(safe_str(v) for v in row), axis=1)
    removed = int((~mask).sum())
    if removed:
        print(f"Removed fully blank rows before processing: {removed}")

    return df.loc[mask].copy().reset_index(drop=True)


# ==========================================================
# OUTPUT WRITER
# ==========================================================
def write_outputs(df: pd.DataFrame) -> None:
    """
    Write local CSV and parquet outputs and log clearly.
    """
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    df = normalize_output_dataframe(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote local CSV: {OUTPUT_CSV}")

    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    print(f"Wrote local parquet: {OUTPUT_PARQUET}")


# ==========================================================
# SFTP HELPERS
# ==========================================================
def sftp_enabled() -> bool:
    return all([SFTP_HOST, SFTP_USERNAME, SFTP_PASSWORD, REMOTE_DIR])


def create_sftp_client():
    if not sftp_enabled():
        raise ValueError("Missing SFTP configuration.")

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    remote_directory = remote_directory.strip("/")
    if not remote_directory:
        return

    parts = remote_directory.split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def download_file_from_sftp(
    remote_filename: str,
    local_path: Path,
    required: bool = True,
) -> bool:
    transport = None
    sftp = None
    try:
        transport, sftp = create_sftp_client()
        remote_path = posixpath.join(REMOTE_DIR, remote_filename)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {remote_path} -> {local_path}")
        sftp.get(remote_path, str(local_path))
        print(f"Downloaded: {remote_filename}")
        return True

    except FileNotFoundError:
        msg = f"Remote file not found: {remote_filename}"
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return False

    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def upload_file_to_sftp(local_path: Path, remote_filename: str) -> None:
    transport = None
    sftp = None
    try:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        transport, sftp = create_sftp_client()
        ensure_remote_dir(sftp, REMOTE_DIR)

        remote_path = posixpath.join(REMOTE_DIR, remote_filename)

        print(f"Uploading {local_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)
        print(f"Uploaded and overwrote remote file: {remote_filename}")

    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def verify_remote_file_matches(local_path: Path, remote_filename: str) -> bool:
    transport = None
    sftp = None
    try:
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Cannot verify missing local file: {local_path}")
            return False

        transport, sftp = create_sftp_client()
        remote_path = posixpath.join(REMOTE_DIR, remote_filename)

        local_size = local_path.stat().st_size
        remote_size = sftp.stat(remote_path).st_size

        if local_size == remote_size:
            print(f"Verified {remote_filename}: local and remote sizes match ({local_size} bytes)")
            return True

        print(
            f"Verification failed for {remote_filename}: "
            f"local={local_size} bytes, remote={remote_size} bytes"
        )
        return False

    except FileNotFoundError:
        print(f"Verification failed: remote file not found: {remote_filename}")
        return False

    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


def fetch_required_input_files() -> None:
    if not sftp_enabled():
        raise RuntimeError("SFTP is required because remote files are the source of truth.")

    download_file_from_sftp("output_final.csv", INPUT_CSV, required=True)
    download_file_from_sftp("themes.json", THEMES_FILE, required=True)
    download_file_from_sftp("output_final.parquet", OUTPUT_PARQUET, required=False)


def upload_output_files(verify: bool = True) -> None:
    if not sftp_enabled():
        print("SFTP not configured. Skipping remote upload.")
        return

    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Local CSV output is missing: {OUTPUT_CSV}")

    if not OUTPUT_PARQUET.exists():
        raise FileNotFoundError(f"Local parquet output is missing: {OUTPUT_PARQUET}")

    uploads = [
        (OUTPUT_CSV, "output_final.csv"),
        (OUTPUT_PARQUET, "output_final.parquet"),
    ]

    if PERMANENTLY_FAILED_FILE.exists():
        uploads.append((PERMANENTLY_FAILED_FILE, "permanently_failed_batches.json"))

    for local_path, remote_name in uploads:
        upload_file_to_sftp(local_path, remote_name)

        if verify:
            ok = verify_remote_file_matches(local_path, remote_name)
            if not ok:
                raise RuntimeError(f"Upload verification failed for {remote_name}")


# ==========================================================
# EMAIL NOTIFIER
# ==========================================================
def send_summary_update_email(
    to_email,
    total_rows,
    processed_rows,
    skipped_rows,
    output_csv,
    output_parquet,
    permanently_failed_count=0,
    mock_mode=False,
    smtp_host=SMTP_HOST,
    smtp_port=SMTP_PORT,
    smtp_user=SMTP_USER,
    smtp_pass=SMTP_PASS,
):
    if not all([to_email, smtp_host, smtp_port, smtp_user, smtp_pass]):
        print("Email not sent: missing SMTP configuration.")
        return

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)

    csv_status = "Created" if output_csv.exists() else "Not found"
    parquet_status = "Created" if output_parquet.exists() else "Not found"

    subject = f"Dataset Summary Update Completed | {run_time}"

    plain_text = f"""
Dataset Summary Update Completed

Run time: {run_time}
Mock mode: {mock_mode}

Processing results:
- Total rows in dataset: {total_rows}
- Fully blank rows processed: {processed_rows}
- Rows skipped (already filled): {skipped_rows}
- Permanently failed batches: {permanently_failed_count}

Output files:
- CSV: {output_csv.name} ({csv_status})
- Parquet: {output_parquet.name} ({parquet_status})
"""

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg.set_content(plain_text)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"Summary update email sent to {to_email}")

    except Exception as e:
        print(f"Failed to send summary update email: {e}")


# ==========================================================
# LOAD HELPERS
# ==========================================================
def load_themes(themes_path: Path) -> dict:
    if not themes_path.exists():
        raise FileNotFoundError(f"Themes file not found: {themes_path}")

    with open(themes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_input_dataframe(input_csv: Path, test_rows: int | None = None) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    df = normalize_output_dataframe(df)

    if test_rows:
        df = df.head(test_rows).copy()

    for col in FIELDS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object).map(safe_str)

    return df


# ==========================================================
# PROMPT HELPERS
# ==========================================================
def format_theme_options(theme_list: dict, lang: str) -> str:
    options = theme_list.get(lang, theme_list["en"])
    return ", ".join([t["label"] for t in options])


def to_comma_separated(item) -> str:
    if isinstance(item, list):
        return ", ".join(safe_str(i) for i in item[:2] if safe_str(i))
    if isinstance(item, str) and item.strip():
        return safe_str(item)
    return ""


def pick_random_themes(theme_map: dict, lang: str, n: int = 2) -> list[str]:
    if lang not in theme_map:
        lang = "en"
    options = theme_map[lang]
    selected = random.sample(options, n) if len(options) >= n else options
    return [c["label"] for c in selected]


def build_prompt(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
) -> str:
    numbered_texts = []

    for idx, summary in enumerate(batch_summaries):
        try:
            lang = detect(summary) if str(summary).strip() else "en"
        except LangDetectException:
            lang = "en"

        numbered_texts.append(
            f"{idx + 1}. Summary: {summary}\n"
            f"Language: {lang}\n"
            f"Actor options: {format_theme_options(actor_themes, lang)}\n"
            f"Subject options: {format_theme_options(subject_themes, lang)}\n"
            f"Mechanism options: {format_theme_options(mechanism_themes, lang)}\n"
            f"Type options: {format_theme_options(type_themes, lang)}"
        )

    numbered_text = "\n\n".join(numbered_texts)

    prompt = f"""
Extract repression info from each text below. Return a JSON array of objects in the same order.
Return only valid JSON, no explanations.

Each object must contain:
{json.dumps({field: "" for field in FIELDS}, indent=4)}

Rules:
- Use ONLY the provided options for each field.
- Do NOT invent labels.
- Never return more than TWO labels in any field.
- Return multiple labels as comma-separated strings.
- Do NOT assign labels based on weak implication or speculation.

Texts:
{numbered_text}
"""
    return prompt


# ==========================================================
# BATCH BUILDER
# ==========================================================
def is_field_blank(value) -> bool:
    value = safe_str(value)
    return value == ""


def is_row_filled(row, fields=FIELDS) -> bool:
    for col in fields:
        if not is_field_blank(row.get(col, "")):
            return True
    return False


def is_row_fully_blank(row, fields=FIELDS) -> bool:
    return not is_row_filled(row, fields)


def build_batches(
    df_input: pd.DataFrame,
    max_tokens: int = MAX_BATCH_TOKENS,
    max_rows: int | None = None,
) -> list[tuple[list[int], list[str]]]:
    batches = []
    i = 0

    while i < len(df_input):
        if is_row_filled(df_input.iloc[i]):
            i += 1
            continue

        batch_summaries = []
        batch_indices = []
        batch_tokens = 0

        while i < len(df_input):
            row = df_input.iloc[i]

            if is_row_filled(row):
                i += 1
                continue

            summary = safe_str(row.get("summary", ""))
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


# ==========================================================
# MOCK EXTRACTOR
# ==========================================================
async def mock_extract_batch(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    batch_indices: list[int] | None = None,
):
    await asyncio.sleep(random.uniform(0.1, 0.4))

    if batch_indices is not None:
        print(f"[MOCK] Processing rows with indices: {batch_indices}")

    results = []
    for _summary in batch_summaries:
        lang = "en"
        result = {
            "Actor of repression": pick_random_themes(actor_themes, lang, n=2),
            "Subject of repression": pick_random_themes(subject_themes, lang, n=2),
            "Mechanism of repression": pick_random_themes(mechanism_themes, lang, n=2),
            "Type of event": pick_random_themes(type_themes, lang, n=2),
        }
        for key in FIELDS:
            result[key] = to_comma_separated(result[key])
        results.append(result)

    return results, None


# ==========================================================
# OPENAI EXTRACTOR
# ==========================================================
async def extract_batch(
    batch_summaries: list[str],
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    mock_mode: bool = False,
    batch_indices: list[int] | None = None,
):
    if mock_mode:
        return await mock_extract_batch(
            batch_summaries=batch_summaries,
            actor_themes=actor_themes,
            subject_themes=subject_themes,
            mechanism_themes=mechanism_themes,
            type_themes=type_themes,
            batch_indices=batch_indices,
        )

    if batch_indices is not None:
        print(f"[OPENAI] Processing rows with indices: {batch_indices}")

    prompt = build_prompt(
        batch_summaries=batch_summaries,
        actor_themes=actor_themes,
        subject_themes=subject_themes,
        mechanism_themes=mechanism_themes,
        type_themes=type_themes,
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content.strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start == -1 or end <= 0:
                    raise ValueError("Model response did not contain a valid JSON array.")
                data = json.loads(content[start:end])

            if not isinstance(data, list):
                raise ValueError("Model response is not a JSON array.")

            cleaned = []
            for res in data:
                row = {}
                for key in FIELDS:
                    row[key] = to_comma_separated(res.get(key, ""))
                cleaned.append(row)

            # Keep response length aligned to batch length
            while len(cleaned) < len(batch_summaries):
                cleaned.append({})
            cleaned = cleaned[:len(batch_summaries)]

            return cleaned, None

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

    # Critical safety fix:
    # return blank dictionaries, not "Error" strings that pollute/crash the dataframe.
    return [{} for _ in batch_summaries], "OpenAI extraction failed after retries"


# ==========================================================
# MAIN PROCESS
# ==========================================================
async def process_all(
    df_source: pd.DataFrame,
    actor_themes: dict,
    subject_themes: dict,
    mechanism_themes: dict,
    type_themes: dict,
    mock_mode: bool = False,
) -> pd.DataFrame:
    df_out = normalize_output_dataframe(df_source)

    # Critical dtype fix: make entire dataframe assignment-safe.
    df_out = df_out.astype(object)

    print("Starting processing from latest output_final.csv")

    for col in FIELDS:
        if col not in df_out.columns:
            df_out[col] = ""
        df_out[col] = df_out[col].astype(object).map(safe_str)

    permanently_failed = []
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)

    rows_to_process = df_out[df_out.apply(is_row_fully_blank, axis=1)]

    total_rows = len(df_out)
    blank_rows_count = len(rows_to_process)
    skipped_rows_count = total_rows - blank_rows_count

    print(f"Total rows in dataset: {total_rows}")
    print(f"Fully blank rows to process: {blank_rows_count}")
    print(f"Rows skipped (already filled): {skipped_rows_count}")

    if rows_to_process.empty:
        print("No fully blank rows to process. Nothing to do.")
        write_outputs(df_out)
        return df_out

    batches = build_batches(rows_to_process, max_rows=MAX_BATCH_SIZE)
    print(f"Total batches to process (fully blank rows): {len(batches)}")

    async def process_batch(batch_indices: list[int], batch_summaries: list[str]):
        async with semaphore:
            results, error = await extract_batch(
                batch_summaries=batch_summaries,
                actor_themes=actor_themes,
                subject_themes=subject_themes,
                mechanism_themes=mechanism_themes,
                type_themes=type_themes,
                mock_mode=mock_mode,
                batch_indices=batch_indices,
            )

            if error:
                permanently_failed.append(
                    {"start_idx": batch_indices[0], "rows": batch_indices, "error": error}
                )

            for j, res in enumerate(results):
                idx = batch_indices[j]
                filled = 0

                for key in FIELDS:
                    value = safe_str(res.get(key, ""))

                    # Only write meaningful valid labels.
                    # Do not write "Error" or blanks into dataframe.
                    if value:
                        df_out.loc[idx, key] = value
                        filled += 1

                if filled == 0:
                    print(f"[ROW {idx}] No valid labels returned; row left blank for next run.")
                else:
                    print(f"[ROW {idx}] Filled {filled}/{len(FIELDS)} fields.")

    for i in range(0, len(batches), CONCURRENT_BATCHES):
        chunk = batches[i:i + CONCURRENT_BATCHES]
        tasks = [process_batch(*b) for b in chunk]
        await tqdm_asyncio.gather(*tasks, desc="Processing batches", total=len(chunk))

    # Only write once at the end to avoid partial corrupted CSVs.
    write_outputs(df_out)

    if permanently_failed:
        with open(PERMANENTLY_FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(permanently_failed, f, indent=2)
    elif PERMANENTLY_FAILED_FILE.exists():
        PERMANENTLY_FAILED_FILE.unlink()

    print(f"Processing complete! Fully blank rows processed: {blank_rows_count}")
    return df_out


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    mock_mode = os.getenv("MOCK_MODE", "false").strip().lower() in {"1", "true", "yes"}

    fetch_required_input_files()

    themes = load_themes(THEMES_FILE)
    ACTOR_THEMES = themes["ACTOR_THEMES"]
    SUBJECT_THEMES = themes["SUBJECT_THEMES"]
    MECHANISM_THEMES = themes["MECHANISM_THEMES"]
    TYPE_THEMES = themes["TYPE_THEMES"]

    df_source = load_input_dataframe(INPUT_CSV, test_rows=TEST_ROWS)

    df_prev = load_input_dataframe(INPUT_CSV)

    blank_rows = df_prev[df_prev.apply(is_row_fully_blank, axis=1)]
    total_rows = len(df_prev)
    skipped_rows = total_rows - len(blank_rows)

    print(
        f"Total rows: {total_rows} | "
        f"Fully blank rows to process: {len(blank_rows)} | "
        f"Skipped rows: {skipped_rows}"
    )

    df_out = asyncio.run(
        process_all(
            df_source=df_source,
            actor_themes=ACTOR_THEMES,
            subject_themes=SUBJECT_THEMES,
            mechanism_themes=MECHANISM_THEMES,
            type_themes=TYPE_THEMES,
            mock_mode=mock_mode,
        )
    )

    summary_message = (
        f"Processing complete! Total rows: {len(df_out)} | "
        f"Fully blank rows processed: {len(blank_rows)} | "
        f"Skipped rows: {skipped_rows} | "
        f"Mock mode: {mock_mode}"
    )
    print(summary_message)

    upload_output_files(verify=True)

    if NOTIFY_EMAIL:
        permanently_failed_count = 0
        if PERMANENTLY_FAILED_FILE.exists():
            with open(PERMANENTLY_FAILED_FILE, "r", encoding="utf-8") as f:
                permanently_failed = json.load(f)
            permanently_failed_count = len(permanently_failed)

        send_summary_update_email(
            to_email=NOTIFY_EMAIL,
            total_rows=len(df_out),
            processed_rows=len(blank_rows),
            skipped_rows=skipped_rows,
            output_csv=OUTPUT_CSV,
            output_parquet=OUTPUT_PARQUET,
            permanently_failed_count=permanently_failed_count,
            mock_mode=mock_mode,
        )
