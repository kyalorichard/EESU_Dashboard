#!/usr/bin/env python3
import os
import re
import paramiko
import pandas as pd
import logging
import hashlib
import smtplib
from email.message import EmailMessage
from datetime import datetime

# ==========================================================
# CONFIG (ENVIRONMENT VARIABLES)
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR") or "exports"
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")

# ---------------- EMAIL CONFIG ----------------
SMTP_SERVER = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
EMAIL_TO = os.getenv("NOTIFY_EMAIL")
EMAIL_SUBJECT = "Data Download Update Notification"

RAW_FILENAME = "raw_data.csv"
FINAL_FILENAME = "output_final.csv"
CHANGELOG_FILENAME = "change_log.csv"

os.makedirs(LOCAL_DIR, exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==========================================================
# GENERAL HELPERS
# ==========================================================
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def set_github_env(key: str, value: str) -> None:
    """
    Expose variables to subsequent GitHub Actions steps.
    """
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def set_github_output(key: str, value: str) -> None:
    """
    Expose step outputs to GitHub Actions.
    Useful when workflow checks: steps.sync.outputs.new_files
    """
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def ensure_remote_dir(sftp, path: str):
    """
    Create remote directories if missing.
    Supports nested folders.
    """
    parts = [p for p in path.strip("/").split("/") if p]
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}" if cur else p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def extract_date_dt(filename: str):
    """
    Extracts date from filenames such as:
    EventsExports_2026_05_06_1.csv
    """
    m = re.search(r"(\d{4}_\d{2}_\d{2})", filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y_%m_%d")
    except ValueError:
        return None


# ==========================================================
# NORMALIZATION + DEDUPLICATION HELPERS
# ==========================================================
def normalize_text(series: pd.Series) -> pd.Series:
    """
    Normalize text values so minor spacing, case, newline, and punctuation
    inconsistencies do not create duplicate records.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\n\r\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u00a0", " ", regex=False)
    )


def normalize_date(series: pd.Series) -> pd.Series:
    """
    Normalize date formats before building the UID.

    This prevents duplicates caused by the same date appearing as:
    - 8/21/2025
    - 21/08/2025
    - 2025-08-21
    - 2025/08/21

    The function first tries normal parsing, then day-first parsing.
    """
    s = series.fillna("").astype(str).str.strip()

    parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
    parsed_dayfirst = pd.to_datetime(s, errors="coerce", dayfirst=True)

    parsed = parsed.fillna(parsed_dayfirst)

    normalized = parsed.dt.strftime("%Y-%m-%d")
    fallback = normalize_text(s)

    return normalized.fillna(fallback)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize incoming/exported dataframe columns and values.
    """
    df = df.copy().fillna("")

    rename_map = {
        "Title": "post_title",
        "Content": "summary",
        "Date": "creation_date",
        "Countries": "alert-country",
        "Impact": "alert-impact",
        "Alert types": "alert-type",
        "Enabling principles": "enabling-principle",
    }
    df.rename(columns=rename_map, inplace=True)

    # Remove unnamed CSV index columns if they exist
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True, errors="ignore")

    # Normalize enabling-principle separators
    if "enabling-principle" in df.columns:
        df["enabling-principle"] = (
            df["enabling-principle"]
            .astype(str)
            .str.replace(r"\|", ",", regex=True)
            .str.replace(r"\s*,\s*", ",", regex=True)
            .str.strip(", ")
        )

    # Normalize core text fields lightly for storage consistency
    for col in ["post_title", "summary", "alert-country", "alert-impact", "alert-type", "enabling-principle"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\n\r\t]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # Store dates consistently where possible
    if "creation_date" in df.columns:
        parsed = pd.to_datetime(df["creation_date"].astype(str).str.strip(), errors="coerce", dayfirst=False)
        parsed_dayfirst = pd.to_datetime(df["creation_date"].astype(str).str.strip(), errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed_dayfirst)
        df["creation_date"] = parsed.dt.strftime("%Y-%m-%d").fillna(df["creation_date"].astype(str).str.strip())

    return df


def make_uid(df: pd.DataFrame) -> pd.Series:
    """
    Create stable row IDs for incremental deduplication.

    Important:
    Do not rely on raw date text. Normalize dates first to avoid duplicate rows
    caused by formats such as 8/21/2025 vs 2025-08-21.
    """
    df = df.copy()

    key_cols = ["post_title", "creation_date", "alert-country", "alert-type"]
    for c in key_cols:
        if c not in df.columns:
            df[c] = ""

    key = (
        normalize_text(df["post_title"]) + "||" +
        normalize_date(df["creation_date"]) + "||" +
        normalize_text(df["alert-country"]) + "||" +
        normalize_text(df["alert-type"])
    )

    return key.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())


def dedupe_by_uid(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    """
    Deduplicate a dataframe using the stable UID.
    """
    if df is None or df.empty:
        return df

    out = df.copy().fillna("")
    out["_uid"] = make_uid(out)

    before = len(out)
    out = out.drop_duplicates(subset=["_uid"], keep=keep)
    after = len(out)

    removed = before - after
    if removed > 0:
        logging.info(f"Removed {removed} duplicate rows using stable UID.")

    return out.drop(columns=["_uid"], errors="ignore")


def align_columns(df_final: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Align columns before concatenation.
    Keeps existing final columns and appends any new raw columns.
    """
    final_cols = list(df_final.columns)
    raw_cols = list(df_raw.columns)

    all_cols = final_cols + [c for c in raw_cols if c not in final_cols]

    return (
        df_final.reindex(columns=all_cols, fill_value=""),
        df_raw.reindex(columns=all_cols, fill_value="")
    )


def is_blank_value(value) -> bool:
    """
    Returns True when a value should be treated as missing/blank.
    """
    if pd.isna(value):
        return True

    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na", "n/a"}


def upsert_existing_records(df_final: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Update existing rows in output_final.csv using matching raw_data.csv rows.

    Append-only logic is not enough because a raw export may contain the same
    event UID but with newly filled columns such as actor, subject, mechanism,
    type of event, enabling principles, or other classification fields.

    Policy:
    - New UID in raw_data.csv: append as a new row.
    - Existing UID: fill blank values in output_final.csv from raw_data.csv.
    - Existing UID: do not overwrite non-blank values already in output_final.csv.

    Returns:
    - updated_final_df
    - new_rows_df
    - updated_cell_count
    - updated_row_count
    """
    if df_final is None or df_final.empty:
        new_rows = df_raw.copy()
        return df_raw.drop(columns=["_uid"], errors="ignore"), new_rows, 0, 0

    final_work = df_final.copy().reset_index(drop=True)
    raw_work = df_raw.copy().reset_index(drop=True)

    if "_uid" not in final_work.columns:
        final_work["_uid"] = make_uid(final_work)
    if "_uid" not in raw_work.columns:
        raw_work["_uid"] = make_uid(raw_work)

    # Keep the last raw version per UID because it is usually the most recent export state
    raw_unique = raw_work.drop_duplicates(subset=["_uid"], keep="last").copy()

    final_uid_to_index = {
        uid: idx for idx, uid in final_work["_uid"].astype(str).items()
    }

    new_rows = []
    updated_cell_count = 0
    updated_row_uids = set()

    for _, raw_row in raw_unique.iterrows():
        uid = str(raw_row["_uid"])

        if uid not in final_uid_to_index:
            new_rows.append(raw_row)
            continue

        final_idx = final_uid_to_index[uid]

        for col in raw_unique.columns:
            if col == "_uid":
                continue

            raw_value = raw_row.get(col, "")
            final_value = final_work.at[final_idx, col] if col in final_work.columns else ""

            if is_blank_value(final_value) and not is_blank_value(raw_value):
                final_work.at[final_idx, col] = raw_value
                updated_cell_count += 1
                updated_row_uids.add(uid)

    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        updated_final = pd.concat(
            [
                final_work.drop(columns=["_uid"], errors="ignore"),
                new_rows_df.drop(columns=["_uid"], errors="ignore"),
            ],
            ignore_index=True,
        )
    else:
        new_rows_df = pd.DataFrame(columns=raw_work.columns)
        updated_final = final_work.drop(columns=["_uid"], errors="ignore")

    updated_final = dedupe_by_uid(updated_final, keep="last")

    return updated_final, new_rows_df, updated_cell_count, len(updated_row_uids)


# ==========================================================
# EMAIL FUNCTION
# ==========================================================
def send_update_email(new_rows_count, latest_file, local_path):
    """
    Sends a professional HTML email summarizing the SFTP CSV update.
    Only sends if new_rows_count > 0.
    """
    if not new_rows_count or new_rows_count <= 0:
        logging.info("No new rows; email not sent.")
        return

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if not EMAIL_FROM:
        logging.error("ALERT_EMAIL_FROM is not set; cannot send email.")
        return
    if not SMTP_SERVER or not SMTP_PORT:
        logging.error("SMTP_HOST/SMTP_PORT not set; cannot send email.")
        return
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logging.error("SMTP_USER/SMTP_PASS not set; cannot send email.")
        return

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height:1.5; color:#333;">
        <h2 style="color:green;">✅ SFTP Data Sync Successful</h2>
        <p>The scheduled SFTP data synchronization completed successfully.</p>
        <table style="border-collapse: collapse; width: 600px;">
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Status</td>
                <td style="padding: 8px; border: 1px solid #ccc;">SUCCESS</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Downloaded file</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{latest_file}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Saved as</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{local_path}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">New updates added</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{new_rows_count}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Date</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{now}</td></tr>
        </table>
        <p style="margin-top: 20px;">This is an automated notification from the SFTP Data Sync system.</p>
    </body>
    </html>
    """

    text_body = (
        "SFTP Data Sync Successful\n"
        f"Status: SUCCESS\n"
        f"Downloaded file: {latest_file}\n"
        f"Saved as: {local_path}\n"
        f"New updates added: {new_rows_count}\n"
        f"Date: {now}\n"
        "This is an automated notification from the SFTP Data Sync system.\n"
    )

    def parse_recipients(value):
        if not value:
            return []
        raw = (
            ",".join([str(v).strip() for v in value])
            if isinstance(value, (list, tuple, set))
            else str(value).strip()
        )
        recips = [r.strip() for r in re.split(r"[;,]\s*", raw) if r and r.strip()]
        return [r for r in recips if r.lower() != "none"]

    recipients = parse_recipients(EMAIL_TO)
    if not recipients:
        logging.warning("NOTIFY_EMAIL is empty/invalid; skipping email notification.")
        return

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = EMAIL_SUBJECT.strip() if EMAIL_SUBJECT else "Data Download Update Notification"
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"Update email sent successfully to {recipients}: {new_rows_count} new rows.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


# ==========================================================
# MAIN
# ==========================================================
transport = None
sftp = None

try:
    require_env("SFTP_HOST")
    require_env("SFTP_USERNAME")
    require_env("SFTP_PASSWORD")

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    logging.info("Connected to SFTP.")

    ensure_remote_dir(sftp, REMOTE_DIR)

    raw_changed = False
    final_changed = False

    # ---------------- LIST FILES ----------------
    remote_files = sftp.listdir(REMOTE_DIR)
    csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

    # Ignore generated outputs to avoid selecting output_final.csv as source input
    generated_files = {RAW_FILENAME.lower(), FINAL_FILENAME.lower(), CHANGELOG_FILENAME.lower()}
    csv_files = [f for f in csv_files if f.lower() not in generated_files]

    csv_files_with_dates = [(f, extract_date_dt(f)) for f in csv_files]
    csv_files_with_dates = [(f, dt) for (f, dt) in csv_files_with_dates if dt is not None]

    if not csv_files_with_dates:
        raise RuntimeError(f"No dated source CSV files found in remote folder: {REMOTE_DIR}")

    latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
    remote_path = f"{REMOTE_DIR}/{latest_file}"

    local_raw_path = os.path.join(LOCAL_DIR, RAW_FILENAME)
    final_path = os.path.join(LOCAL_DIR, FINAL_FILENAME)
    change_log_path = os.path.join(LOCAL_DIR, CHANGELOG_FILENAME)

    remote_raw_path = f"{REMOTE_DIR}/{RAW_FILENAME}"
    remote_final_path = f"{REMOTE_DIR}/{FINAL_FILENAME}"
    remote_changelog_path = f"{REMOTE_DIR}/{CHANGELOG_FILENAME}"

    # ---------------- DOWNLOAD LATEST EXPORT ----------------
    remote_attr = sftp.stat(remote_path)
    remote_size = remote_attr.st_size
    remote_mtime = remote_attr.st_mtime

    download_needed = True
    if os.path.exists(local_raw_path):
        local_size = os.path.getsize(local_raw_path)
        local_mtime = int(os.path.getmtime(local_raw_path))
        if local_size == remote_size and local_mtime >= remote_mtime:
            download_needed = False

    if download_needed:
        sftp.get(remote_path, local_raw_path)
        os.utime(local_raw_path, (remote_mtime, remote_mtime))
        logging.info(f"Downloaded latest source file: {latest_file} -> {local_raw_path}")
        raw_changed = True
    else:
        logging.info(f"{RAW_FILENAME} is already up to date locally. Skipping download.")

    # ---------------- SYNC REMOTE FINAL -> LOCAL ----------------
    try:
        sftp.stat(remote_final_path)
        sftp.get(remote_final_path, final_path)
        logging.info(f"Downloaded existing remote FINAL for dedupe -> {final_path}")
    except FileNotFoundError:
        logging.info("No remote FINAL found yet; starting fresh.")

    # ---------------- LOAD + STANDARDIZE RAW ----------------
    df_raw = pd.read_csv(local_raw_path).fillna("")
    df_raw = standardize_dataframe(df_raw)

    if "post_title" not in df_raw.columns:
        raise RuntimeError(f"Missing required column post_title after rename. Found: {list(df_raw.columns)}")

    # Remove duplicates inside the new raw export itself
    before_raw = len(df_raw)
    df_raw = dedupe_by_uid(df_raw, keep="last")
    logging.info(f"Raw rows loaded: {before_raw}; after raw dedupe: {len(df_raw)}")

    # ---------------- LOAD + STANDARDIZE EXISTING FINAL ----------------
    if os.path.exists(final_path):
        df_final = pd.read_csv(final_path).fillna("")
        df_final = standardize_dataframe(df_final)
    else:
        df_final = pd.DataFrame(columns=df_raw.columns)

    before_final = len(df_final)
    df_final = dedupe_by_uid(df_final, keep="last")
    logging.info(f"Final rows loaded: {before_final}; after final dedupe: {len(df_final)}")

    # If the final file had duplicates, rewrite/upload it even if no new rows
    if before_final != len(df_final):
        final_changed = True
        logging.info("Existing FINAL contained duplicates and will be cleaned.")

    # Align columns before UID/filter/concat
    df_final, df_raw = align_columns(df_final, df_raw)

    # ---------------- UPSERT: APPEND NEW + FILL MISSING EXISTING VALUES ----------------
    df_raw["_uid"] = make_uid(df_raw)
    df_final["_uid"] = make_uid(df_final) if len(df_final) > 0 else pd.Series(dtype=str)

    logging.info(f"df_final rows available for upsert check: {len(df_final)}")

    updated_final_df, new_rows, updated_cell_count, updated_row_count = upsert_existing_records(
        df_final,
        df_raw
    )

    has_new_rows = not new_rows.empty
    has_updated_existing_rows = updated_cell_count > 0

    if has_new_rows or has_updated_existing_rows or final_changed:
        updated_final_df.to_csv(final_path, index=False)
        final_changed = True

        logging.info(
            f"FINAL updated -> {final_path}; "
            f"new rows appended: {len(new_rows)}; "
            f"existing rows updated: {updated_row_count}; "
            f"blank cells filled: {updated_cell_count}"
        )

        # Write changelog for genuinely new rows only
        if has_new_rows:
            new_rows_out = new_rows.copy()
            cols = [c for c in df_raw.columns if c != "_uid"] + ["_uid"]
            new_rows_out = new_rows_out.reindex(columns=cols, fill_value="")
            new_rows_out.to_csv(
                change_log_path,
                mode="a",
                header=not os.path.exists(change_log_path),
                index=False
            )
            logging.info(f"Change log updated with {len(new_rows)} new rows -> {change_log_path}")

        # Send email when new rows are appended or existing rows were enriched
        email_count = len(new_rows) if has_new_rows else updated_row_count
        send_update_email(email_count, latest_file, local_raw_path)

    else:
        logging.info("No new rows and no missing values to update. Email not sent.")

        # Ensure final exists on first run
        if not os.path.exists(final_path):
            df_final.drop(columns=["_uid"], errors="ignore").to_csv(final_path, index=False)

    # ---------------- EXPORT FLAGS TO GITHUB ACTIONS ----------------
    flag_value = "1" if final_changed else "0"
    set_github_env("NEW_FILES_DOWNLOADED", flag_value)
    set_github_output("new_files", flag_value)
    logging.info(f"NEW_FILES_DOWNLOADED={flag_value}")
    logging.info(f"new_files={flag_value}")

    # ---------------- UPLOAD BACK TO REMOTE ONLY IF CHANGED ----------------
    try:
        if raw_changed:
            sftp.put(local_raw_path, remote_raw_path)
            logging.info(f"Uploaded RAW -> {remote_raw_path}")
        else:
            logging.info("RAW unchanged; skipping RAW upload.")

        if final_changed:
            sftp.put(final_path, remote_final_path)
            logging.info(f"Uploaded FINAL -> {remote_final_path}")

            if os.path.exists(change_log_path):
                sftp.put(change_log_path, remote_changelog_path)
                logging.info(f"Uploaded CHANGELOG -> {remote_changelog_path}")
            else:
                logging.info("No changelog to upload.")
        else:
            logging.info("FINAL unchanged; skipping FINAL/CHANGELOG upload.")

    except Exception as e:
        logging.exception(f"Failed to upload outputs to remote exports folder: {e}")

    logging.info("Incremental update completed successfully.")

finally:
    try:
        if sftp:
            sftp.close()
    finally:
        if transport:
            transport.close()
    logging.info("SFTP session closed.")
