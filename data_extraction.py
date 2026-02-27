#!/usr/bin/env python3
import os
import re
import paramiko
import pandas as pd
import logging
import hashlib
import smtplib
from email.message import EmailMessage
from datetime import datetime   # <-- make sure this is here

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
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
EMAIL_TO = os.getenv("NOTIFY_EMAIL")
EMAIL_SUBJECT = " Data Download Update Notification"

RAW_FILENAME = "raw_data.csv"
FINAL_FILENAME = "output_final.csv"
CHANGELOG_FILENAME = "change_log.csv"

os.makedirs(LOCAL_DIR, exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- EMAIL FUNCTION ----------------
def send_update_email(new_rows_count, latest_file, local_path):
    """
    Sends a professional HTML email summarizing the SFTP CSV update.
    Only sends if new_rows_count > 0.
    """
    # Optional guard (keeps behavior explicit)
    if not new_rows_count or new_rows_count <= 0:
        logging.info("No new rows; email not sent.")
        return

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # --- Validate minimum email/SMTP config (fail softly) ---
    if not EMAIL_FROM:
        logging.error("ALERT_EMAIL_FROM is not set; cannot send email.")
        return
    if not SMTP_SERVER or not SMTP_PORT:
        logging.error("SMTP_HOST/SMTP_PORT not set; cannot send email.")
        return
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logging.error("SMTP_USER/SMTP_PASS not set; cannot send email.")
        return

    # --- Build HTML body ---
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height:1.5; color:#333;">
        <h2 style="color:green;">✅ SFTP Data Sync Successful</h2>
        <p>The scheduled SFTP data synchronization completed successfully.</p>
        <table style="border-collapse: collapse; width: 600px;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Status</td>
                <td style="padding: 8px; border: 1px solid #ccc;">SUCCESS</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Downloaded file</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{latest_file}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Saved as</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{local_path}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">New updates added</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{new_rows_count}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Date</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{now}</td>
            </tr>
        </table>
        <p style="margin-top: 20px;">This is an automated notification from the SFTP Data Sync system.</p>
    </body>
    </html>
    """

    # --- Plain-text fallback (some clients block HTML-only emails) ---
    text_body = (
        "SFTP Data Sync Successful\n"
        f"Status: SUCCESS\n"
        f"Downloaded file: {latest_file}\n"
        f"Saved as: {local_path}\n"
        f"New updates added: {new_rows_count}\n"
        f"Date: {now}\n"
        "This is an automated notification from the SFTP Data Sync system.\n"
    )

    # --- Robust recipient parsing (never returns None entries) ---
    def parse_recipients(value):
        if not value:
            return []
        if isinstance(value, (list, tuple, set)):
            raw = ",".join([str(v).strip() for v in value if v])
        else:
            raw = str(value).strip()

        # split on comma/semicolon, trim, drop empties/"none"
        recips = [r.strip() for r in re.split(r"[;,]\s*", raw) if r and r.strip()]
        recips = [r for r in recips if r.lower() != "none"]
        return recips

    recipients = parse_recipients(EMAIL_TO)
    if not recipients:
        logging.warning("NOTIFY_EMAIL is empty/invalid; skipping email notification.")
        return

    # --- Construct message ---
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = EMAIL_SUBJECT.strip() if EMAIL_SUBJECT else "Data Download Update Notification"
    msg["Date"] = now  # optional; EmailMessage will add one if omitted

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    # --- Send ---
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
        
# ---------------- SFTP CONNECTION ----------------
transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
sftp = paramiko.SFTPClient.from_transport(transport)
logging.info("Connected to SFTP.")

# ---------------- LIST FILES ----------------
remote_files = sftp.listdir(REMOTE_DIR)
csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

def extract_date(filename):
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    return match.group(1) if match else None

csv_files_with_dates = [(f, extract_date(f)) for f in csv_files if extract_date(f)]
if not csv_files_with_dates:
    logging.error("No CSV files with dates found.")
    sftp.close()
    transport.close()
    exit()

latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
remote_path = f"{REMOTE_DIR}/{latest_file}"
local_path = os.path.join(LOCAL_DIR, RAW_FILENAME)

# ---------------- DOWNLOAD ----------------
remote_size = sftp.stat(remote_path).st_size
if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
    logging.info(f"{RAW_FILENAME} is already up to date. Skipping download.")
else:
    sftp.get(remote_path, local_path)
    logging.info(f"Downloaded latest file: {latest_file}")

# ---------------- LOAD RAW CSV ----------------
df_raw = pd.read_csv(local_path).fillna("")

# Rename columns to match final CSV
rename_map = {
    "Title": "post_title",
    "Content": "summary",
    "Date": "creation_date",
    "Countries": "alert-country",
    "Impact": "alert-impact",
    "Alert types": "alert-type",
    "Enabling principles": "enabling-principle"
}
df_raw.rename(columns=rename_map, inplace=True)

# Clean enabling-principle column
if "enabling-principle" in df_raw.columns:
    df_raw["enabling-principle"] = (
        df_raw["enabling-principle"]
        .astype(str)
        .str.replace(r"\|", ",", regex=True)
        .str.replace(r"\s*,\s*", ",", regex=True)
    )

logging.info(f"Columns renamed and cleaned in {RAW_FILENAME}")

# ---------------- LOAD EXISTING FINAL CSV ----------------
final_path = os.path.join(LOCAL_DIR, FINAL_FILENAME)
if os.path.exists(final_path):
    df_final = pd.read_csv(final_path).fillna("")
else:
    df_final = pd.DataFrame(columns=df_raw.columns)

# ---------------- GENERATE UNIQUE ID ----------------
def normalize_text(series):
    return series.astype(str).str.strip().str.lower().str.replace(r"[\n\r]+", " ", regex=True)

df_raw["_uid"] = normalize_text(df_raw["post_title"]).apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
df_final["_uid"] = normalize_text(df_final["post_title"]).apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())

# ---------------- FILTER NEW ROWS ----------------
new_rows = df_raw[~df_raw["_uid"].isin(df_final["_uid"])].copy()

if not new_rows.empty:
    # Append new rows
    combined_df = pd.concat([df_final, new_rows], ignore_index=True)
    combined_df.to_csv(final_path, index=False)
    logging.info(f"New rows appended: {len(new_rows)}")

    # Update change log
    change_log_path = os.path.join(LOCAL_DIR, CHANGELOG_FILENAME)
    new_rows.to_csv(change_log_path, mode='a', header=not os.path.exists(change_log_path), index=False)
    logging.info(f"Change log updated with {len(new_rows)} rows.")

    # Send professional email only if new rows exist
    send_update_email(len(new_rows), latest_file, local_path)
else:
    logging.info("No new rows to append. Email not sent.")

# ---------------- CLOSE SFTP ----------------
sftp.close()
transport.close()
logging.info("SFTP session closed. Incremental update completed successfully.")

