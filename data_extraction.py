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
# CONFIG
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR") or "exports"
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==========================================================
# GENERAL HELPERS
# ==========================================================
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def set_github_env(key: str, value: str) -> None:
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def set_github_output(key: str, value: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def ensure_remote_dir(sftp, path: str):
    parts = [p for p in path.strip("/").split("/") if p]
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}" if cur else p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def extract_date_dt(filename: str):
    m = re.search(r"(\d{4}_\d{2}_\d{2})", filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y_%m_%d")
    except ValueError:
        return None


# ==========================================================
# DATA CLEANING + UID HELPERS
# ==========================================================
def normalize_text(series: pd.Series) -> pd.Series:
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
    s = series.fillna("").astype(str).str.strip()

    parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
    parsed_dayfirst = pd.to_datetime(s, errors="coerce", dayfirst=True)
    parsed = parsed.fillna(parsed_dayfirst)

    normalized = parsed.dt.strftime("%Y-%m-%d")
    return normalized.fillna(normalize_text(s))


def drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Critical fix:
    Remove rows where every cell is blank before dedupe/upsert.
    This prevents thousands of blank rows from being preserved in output_final.csv.
    """
    if df is None or df.empty:
        return df

    mask = df.apply(lambda row: any(str(v).strip() for v in row), axis=1)
    removed = int((~mask).sum())

    if removed > 0:
        logging.info(f"Removed {removed} fully blank rows.")

    return df.loc[mask].copy().reset_index(drop=True)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True, errors="ignore")

    if "enabling-principle" in df.columns:
        df["enabling-principle"] = (
            df["enabling-principle"]
            .astype(str)
            .str.replace(r"\|", ",", regex=True)
            .str.replace(r"\s*,\s*", ",", regex=True)
            .str.strip(", ")
        )

    for col in [
        "post_title",
        "summary",
        "alert-country",
        "alert-impact",
        "alert-type",
        "enabling-principle",
        "Actor of repression",
        "Subject of repression",
        "Mechanism of repression",
        "Type of event",
    ]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\n\r\t]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    if "creation_date" in df.columns:
        parsed = pd.to_datetime(df["creation_date"].astype(str).str.strip(), errors="coerce", dayfirst=False)
        parsed_dayfirst = pd.to_datetime(df["creation_date"].astype(str).str.strip(), errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed_dayfirst)
        df["creation_date"] = parsed.dt.strftime("%Y-%m-%d").fillna(
            df["creation_date"].astype(str).str.strip()
        )

    df = drop_fully_blank_rows(df)

    return df


def make_uid(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    key_cols = ["post_title", "creation_date", "alert-country", "alert-type"]
    for col in key_cols:
        if col not in df.columns:
            df[col] = ""

    key = (
        normalize_text(df["post_title"]) + "||" +
        normalize_date(df["creation_date"]) + "||" +
        normalize_text(df["alert-country"]) + "||" +
        normalize_text(df["alert-type"])
    )

    return key.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())


def dedupe_by_uid(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = drop_fully_blank_rows(df.copy().fillna(""))
    if out.empty:
        return out

    out["_uid"] = make_uid(out)

    before = len(out)
    out = out.drop_duplicates(subset=["_uid"], keep=keep)
    after = len(out)

    removed = before - after
    if removed > 0:
        logging.info(f"Removed {removed} duplicate non-blank rows using stable UID.")

    return out.drop(columns=["_uid"], errors="ignore").reset_index(drop=True)


def align_columns(df_final: pd.DataFrame, df_raw: pd.DataFrame):
    final_cols = list(df_final.columns)
    raw_cols = list(df_raw.columns)
    all_cols = final_cols + [c for c in raw_cols if c not in final_cols]

    return (
        df_final.reindex(columns=all_cols, fill_value=""),
        df_raw.reindex(columns=all_cols, fill_value=""),
    )


def is_blank_value(value) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na", "n/a"}


def upsert_existing_records(df_final: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Append new UIDs and fill blank fields in existing UIDs.
    Does not overwrite existing non-blank values.
    """
    df_final = drop_fully_blank_rows(df_final.copy().fillna(""))
    df_raw = drop_fully_blank_rows(df_raw.copy().fillna(""))

    if df_final.empty:
        new_rows = df_raw.copy()
        return dedupe_by_uid(df_raw, keep="last"), new_rows, 0, 0

    final_work = df_final.copy().reset_index(drop=True)
    raw_work = df_raw.copy().reset_index(drop=True)

    final_work["_uid"] = make_uid(final_work)
    raw_work["_uid"] = make_uid(raw_work)

    # Keep the last version inside raw export
    raw_unique = raw_work.drop_duplicates(subset=["_uid"], keep="last").copy()

    # Keep the last existing final row if final already contains duplicates
    final_work = final_work.drop_duplicates(subset=["_uid"], keep="last").reset_index(drop=True)

    final_uid_to_index = {
        str(uid): idx for idx, uid in final_work["_uid"].astype(str).items()
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
# EMAIL
# ==========================================================
def send_update_email(new_rows_count, latest_file, local_path):
    if not new_rows_count or new_rows_count <= 0:
        logging.info("No new rows/updates; email not sent.")
        return

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if not EMAIL_FROM or not EMAIL_TO:
        logging.warning("Email sender/recipient not configured; skipping notification.")
        return
    if not SMTP_SERVER or not SMTP_USERNAME or not SMTP_PASSWORD:
        logging.warning("SMTP config incomplete; skipping notification.")
        return

    def parse_recipients(value):
        raw = str(value).strip()
        recips = [r.strip() for r in re.split(r"[;,]\s*", raw) if r and r.strip()]
        return [r for r in recips if r.lower() != "none"]

    recipients = parse_recipients(EMAIL_TO)
    if not recipients:
        logging.warning("NOTIFY_EMAIL is empty/invalid; skipping email notification.")
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
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Rows added/updated</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{new_rows_count}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Date</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{now}</td></tr>
        </table>
    </body>
    </html>
    """

    text_body = (
        "SFTP Data Sync Successful\n"
        f"Downloaded file: {latest_file}\n"
        f"Saved as: {local_path}\n"
        f"Rows added/updated: {new_rows_count}\n"
        f"Date: {now}\n"
    )

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = EMAIL_SUBJECT
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"Update email sent successfully to {recipients}.")
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

    remote_files = sftp.listdir(REMOTE_DIR)
    csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

    # Do not ever use generated files as source input
    generated_files = {RAW_FILENAME.lower(), FINAL_FILENAME.lower(), CHANGELOG_FILENAME.lower()}
    csv_files = [f for f in csv_files if f.lower() not in generated_files]

    csv_files_with_dates = [(f, extract_date_dt(f)) for f in csv_files]
    csv_files_with_dates = [(f, dt) for f, dt in csv_files_with_dates if dt is not None]

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

    try:
        sftp.stat(remote_final_path)
        sftp.get(remote_final_path, final_path)
        logging.info(f"Downloaded existing remote FINAL for dedupe -> {final_path}")
    except FileNotFoundError:
        logging.info("No remote FINAL found yet; starting fresh.")

    df_raw = pd.read_csv(local_raw_path, dtype=str, keep_default_na=False)
    df_raw = standardize_dataframe(df_raw)

    if "post_title" not in df_raw.columns:
        raise RuntimeError(f"Missing required column post_title after rename. Found: {list(df_raw.columns)}")

    before_raw = len(df_raw)
    df_raw = dedupe_by_uid(df_raw, keep="last")
    logging.info(f"Raw rows loaded: {before_raw}; after blank-row removal and dedupe: {len(df_raw)}")

    if os.path.exists(final_path):
        df_final = pd.read_csv(final_path, dtype=str, keep_default_na=False)
        df_final = standardize_dataframe(df_final)
    else:
        df_final = pd.DataFrame(columns=df_raw.columns)

    before_final = len(df_final)
    df_final = dedupe_by_uid(df_final, keep="last")
    logging.info(f"Final rows loaded: {before_final}; after blank-row removal and dedupe: {len(df_final)}")

    if before_final != len(df_final):
        final_changed = True
        logging.info("Existing FINAL was polluted with blanks/duplicates and will be cleaned.")

    df_final, df_raw = align_columns(df_final, df_raw)

    updated_final_df, new_rows, updated_cell_count, updated_row_count = upsert_existing_records(
        df_final,
        df_raw,
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
            f"blank cells filled: {updated_cell_count}; "
            f"final rows: {len(updated_final_df)}"
        )

        if has_new_rows:
            new_rows_out = new_rows.copy()
            if "_uid" not in new_rows_out.columns:
                new_rows_out["_uid"] = make_uid(new_rows_out)

            new_rows_out.to_csv(
                change_log_path,
                mode="a",
                header=not os.path.exists(change_log_path),
                index=False,
            )
            logging.info(f"Change log updated with {len(new_rows)} new rows -> {change_log_path}")

        email_count = len(new_rows) + updated_row_count
        send_update_email(email_count, latest_file, local_raw_path)

    else:
        logging.info("No new rows and no missing values to update. Email not sent.")

        if not os.path.exists(final_path):
            df_final.to_csv(final_path, index=False)

    flag_value = "1" if final_changed else "0"
    set_github_env("NEW_FILES_DOWNLOADED", flag_value)
    set_github_output("new_files", flag_value)

    logging.info(f"NEW_FILES_DOWNLOADED={flag_value}")
    logging.info(f"new_files={flag_value}")

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
