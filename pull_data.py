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

# ---------------- CONFIG ----------------
SFTP_HOST = "83.149.119.154"
SFTP_PORT = 22
SFTP_USERNAME = "events-eusee.hivos.o_iwfvvmfr82h"
SFTP_PASSWORD = "~Po7Rpdi9&oY3wkr"

REMOTE_DIR = "exports"
LOCAL_DIR = "data"

RAW_FILENAME = "raw_data.csv"
FINAL_FILENAME = "output_final.csv"
CHANGELOG_FILENAME = "change_log.csv"

# ---------------- EMAIL CONFIG ----------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "kyalorichard11@gmail.com"
SMTP_PASSWORD = "nwkq vyly slsi bexj"
EMAIL_FROM = "kyalorichard11@gmail.com"
EMAIL_TO = ["kyalorichard11@gmail.com"]
EMAIL_SUBJECT = "Data Download Update Notification"

os.makedirs(LOCAL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- EMAIL FUNCTION ----------------
def send_update_email(new_rows_count, latest_file):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html_body = f"""
    <html>
    <body style="font-family: Arial;">
        <h2 style="color:green;">SFTP Data Sync Successful</h2>
        <p><b>Latest file:</b> {latest_file}</p>
        <p><b>New rows added:</b> {new_rows_count}</p>
        <p><b>Date:</b> {now}</p>
        <p>This is an automated notification.</p>
    </body>
    </html>
    """

    msg = EmailMessage()
    msg['From'] = EMAIL_FROM
    msg['To'] = ", ".join(EMAIL_TO)
    msg['Subject'] = EMAIL_SUBJECT
    msg.add_alternative(html_body, subtype='html')

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Email failed: {e}")

# ---------------- CONNECT SFTP ----------------
try:
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    logging.info("Connected to SFTP.")
except Exception as e:
    logging.error(f"SFTP connection failed: {e}")
    exit()

local_file = "data/countries_metadata.json"
remote_file = f"{REMOTE_DIR}/countries_metadata.json"

sftp.put(local_file, remote_file)
logging.info(f"Uploaded {local_file} to {remote_file}")

# ---------------- LOG REMOTE EXPORT DIRECTORY CONTENTS ----------------
try:
    export_files = sftp.listdir(REMOTE_DIR)
    logging.info(f"Files available in remote '{REMOTE_DIR}' directory:")

    for file in export_files:
        try:
            file_path = f"{REMOTE_DIR}/{file}"
            attrs = sftp.stat(file_path)
            size = attrs.st_size
            modified_time = datetime.utcfromtimestamp(attrs.st_mtime).strftime("%Y-%m-%d %H:%M:%S UTC")

            logging.info(f" - {file} | Size: {size} bytes | Modified: {modified_time}")

        except Exception as e:
            logging.warning(f"Could not stat file {file}: {e}")

except Exception as e:
    logging.error(f"Failed to list directory '{REMOTE_DIR}': {e}")

try:
    # ---------------- GET LATEST RAW CSV ----------------
    remote_files = sftp.listdir(REMOTE_DIR)
    csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

    def extract_date(filename):
        match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
        return match.group(1) if match else None

    csv_files_with_dates = [(f, extract_date(f)) for f in csv_files if extract_date(f)]

    if not csv_files_with_dates:
        logging.error("No dated CSV files found.")
        exit()

    latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
    remote_raw_path = f"{REMOTE_DIR}/{latest_file}"
    local_raw_path = os.path.join(LOCAL_DIR, RAW_FILENAME)

    sftp.get(remote_raw_path, local_raw_path)
    logging.info(f"Downloaded latest raw file: {latest_file}")

    # ---------------- LOAD RAW CSV ----------------
    df_raw = pd.read_csv(local_raw_path).fillna("")

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

    if "enabling-principle" in df_raw.columns:
        df_raw["enabling-principle"] = (
            df_raw["enabling-principle"]
            .astype(str)
            .str.replace(r"\|", ",", regex=True)
            .str.replace(r"\s*,\s*", ",", regex=True)
        )

    # ---------------- DOWNLOAD REMOTE FINAL CSV (SOURCE OF TRUTH) ----------------
    local_final_path = os.path.join(LOCAL_DIR, FINAL_FILENAME)
    remote_final_path = f"{REMOTE_DIR}/{FINAL_FILENAME}"

    try:
        sftp.stat(remote_final_path)
        sftp.get(remote_final_path, local_final_path)
        logging.info("Downloaded existing remote output_final.csv")
    except FileNotFoundError:
        logging.info("Remote output_final.csv not found. Creating new.")
        pd.DataFrame(columns=df_raw.columns).to_csv(local_final_path, index=False)

    df_final = pd.read_csv(local_final_path).fillna("")

    # ---------------- GENERATE UID ----------------
    def normalize_text(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\n\r]+", " ", regex=True)
        )

    df_raw["_uid"] = normalize_text(df_raw["post_title"]).apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )

    df_final["_uid"] = normalize_text(df_final["post_title"]).apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )

    # ---------------- FILTER NEW ROWS ----------------
    new_rows = df_raw[~df_raw["_uid"].isin(df_final["_uid"])].copy()

    if not new_rows.empty:
        combined_df = pd.concat([df_final, new_rows], ignore_index=True)
        combined_df.to_csv(local_final_path, index=False)
        logging.info(f"Appended {len(new_rows)} new rows.")

        # ---------------- BACKUP REMOTE BEFORE OVERWRITE ----------------
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{REMOTE_DIR}/backup_output_final_{timestamp}.csv"

        try:
            sftp.rename(remote_final_path, backup_path)
            logging.info(f"Remote backup created: {backup_path}")
        except FileNotFoundError:
            logging.info("No previous remote file to backup.")

        # ---------------- UPLOAD UPDATED FILE ----------------
        sftp.put(local_final_path, remote_final_path)
        logging.info("Uploaded updated output_final.csv")

        # ---------------- UPDATE CHANGE LOG ----------------
        local_changelog_path = os.path.join(LOCAL_DIR, CHANGELOG_FILENAME)
        remote_changelog_path = f"{REMOTE_DIR}/{CHANGELOG_FILENAME}"

        new_rows.to_csv(
            local_changelog_path,
            mode="a",
            header=not os.path.exists(local_changelog_path),
            index=False
        )

        sftp.put(local_changelog_path, remote_changelog_path)
        logging.info("Uploaded updated change_log.csv")

        send_update_email(len(new_rows), latest_file)

    else:
        logging.info("No new rows found. Nothing uploaded.")

finally:
    sftp.close()
    transport.close()
    logging.info("SFTP session closed.")