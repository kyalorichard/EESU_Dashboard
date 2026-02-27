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
SFTP_PORT = 22
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR") or "exports"
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")

# ---------------- EMAIL CONFIG ----------------
SMTP_SERVER = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("NOTIFY_EMAIL")
EMAIL_TO = os.getenv("NOTIFY_EMAIL")
EMAIL_SUBJECT = "Data Download Update Notification"

RAW_FILENAME = "raw_data.csv"
FINAL_FILENAME = "output_final.csv"
CHANGELOG_FILENAME = "change_log.csv"

os.makedirs(LOCAL_DIR, exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

# ---------------- EMAIL FUNCTION ----------------
def send_update_email(new_rows_count, latest_file, local_path):
    """
    Sends a professional HTML email summarizing the SFTP CSV update.
    Only sends if new_rows_count > 0.
    """
    if not new_rows_count or new_rows_count <= 0:
        logging.info("No new rows; email not sent.")
        return

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Validate email/SMTP config (fail softly)
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
        raw = ",".join([str(v).strip() for v in value]) if isinstance(value, (list, tuple, set)) else str(value).strip()
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

# ---------------- HELPERS ----------------
def ensure_remote_dir(sftp, path: str):
    """Create remote directories if missing (supports nested)."""
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}" if cur else p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)

def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.replace(r"[\n\r]+", " ", regex=True)

def extract_date_dt(filename: str):
    m = re.search(r"(\d{4}_\d{2}_\d{2})", filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y_%m_%d")
    except ValueError:
        return None

def make_uid(df: pd.DataFrame) -> pd.Series:
    # stronger composite key than title alone
    key_cols = ["post_title", "creation_date", "alert-country", "alert-type"]
    for c in key_cols:
        if c not in df.columns:
            df[c] = ""
    key = (
        normalize_text(df[key_cols[0]]) + "||" +
        normalize_text(df[key_cols[1]]) + "||" +
        normalize_text(df[key_cols[2]]) + "||" +
        normalize_text(df[key_cols[3]])
    )
    return key.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())

# ==========================================================
# MAIN
# ==========================================================
transport = None
sftp = None

try:
    # fail-fast for required SFTP envs
    require_env("SFTP_HOST")
    require_env("SFTP_USERNAME")
    require_env("SFTP_PASSWORD")

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    logging.info("Connected to SFTP.")

    ensure_remote_dir(sftp, REMOTE_DIR)

    # ---------------- LIST FILES ----------------
    remote_files = sftp.listdir(REMOTE_DIR)
    csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

    csv_files_with_dates = [(f, extract_date_dt(f)) for f in csv_files]
    csv_files_with_dates = [(f, dt) for (f, dt) in csv_files_with_dates if dt is not None]

    if not csv_files_with_dates:
        raise RuntimeError(f"No dated CSV files found in remote folder: {REMOTE_DIR}")

    latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
    remote_path = f"{REMOTE_DIR}/{latest_file}"

    # Local staging
    local_raw_path = os.path.join(LOCAL_DIR, RAW_FILENAME)

    # Remote “same folder” outputs
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
        logging.info(f"Downloaded latest file: {latest_file} -> {local_raw_path}")
    else:
        logging.info(f"{RAW_FILENAME} is already up to date locally. Skipping download.")

    # ---------------- LOAD RAW CSV ----------------
    df_raw = pd.read_csv(local_raw_path).fillna("")

    rename_map = {
        "Title": "post_title",
        "Content": "summary",
        "Date": "creation_date",
        "Countries": "alert-country",
        "Impact": "alert-impact",
        "Alert types": "alert-type",
        "Enabling principles": "enabling-principle",
    }
    df_raw.rename(columns=rename_map, inplace=True)

    if "enabling-principle" in df_raw.columns:
        df_raw["enabling-principle"] = (
            df_raw["enabling-principle"]
            .astype(str)
            .str.replace(r"\|", ",", regex=True)
            .str.replace(r"\s*,\s*", ",", regex=True)
        )

    if "post_title" not in df_raw.columns:
        raise RuntimeError(f"Missing required column post_title after rename. Found: {list(df_raw.columns)}")

    # ---------------- LOAD EXISTING FINAL (LOCAL STAGING) ----------------
    final_path = os.path.join(LOCAL_DIR, FINAL_FILENAME)
    if os.path.exists(final_path):
        df_final = pd.read_csv(final_path).fillna("")
    else:
        df_final = pd.DataFrame(columns=df_raw.columns)

    # ---------------- UID + FILTER NEW ----------------
    df_raw["_uid"] = make_uid(df_raw)
    if len(df_final) > 0:
        df_final["_uid"] = make_uid(df_final)
    else:
        df_final["_uid"] = pd.Series(dtype=str)

    new_rows = df_raw[~df_raw["_uid"].isin(df_final["_uid"])].copy()

    # ---------------- UPDATE FINAL + CHANGELOG (LOCAL) ----------------
    change_log_path = os.path.join(LOCAL_DIR, CHANGELOG_FILENAME)

    if not new_rows.empty:
        combined_df = pd.concat([df_final.drop(columns=["_uid"], errors="ignore"), new_rows.drop(columns=["_uid"], errors="ignore")], ignore_index=True)
        combined_df.to_csv(final_path, index=False)
        logging.info(f"New rows appended: {len(new_rows)} -> {final_path}")

        # changelog append with uid for traceability
        new_rows_out = new_rows.copy()
        cols = [c for c in df_raw.columns if c != "_uid"] + ["_uid"]
        new_rows_out = new_rows_out[cols]
        new_rows_out.to_csv(change_log_path, mode="a", header=not os.path.exists(change_log_path), index=False)
        logging.info(f"Change log updated with {len(new_rows)} rows -> {change_log_path}")

        # Send email
        send_update_email(len(new_rows), latest_file, local_raw_path)
    else:
        logging.info("No new rows to append. Email not sent.")

        # Ensure final exists even on first run (optional)
        if not os.path.exists(final_path):
            df_final.drop(columns=["_uid"], errors="ignore").to_csv(final_path, index=False)

    # ---------------- UPLOAD BACK TO SAME REMOTE FOLDER ----------------
    # Always keep a stable raw snapshot on remote as raw_data.csv
    try:
        sftp.put(local_raw_path, remote_raw_path)
        logging.info(f"Uploaded RAW -> {remote_raw_path}")

        sftp.put(final_path, remote_final_path)
        logging.info(f"Uploaded FINAL -> {remote_final_path}")

        if os.path.exists(change_log_path):
            sftp.put(change_log_path, remote_changelog_path)
            logging.info(f"Uploaded CHANGELOG -> {remote_changelog_path}")
        else:
            logging.info("No changelog to upload (no changes yet).")

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
