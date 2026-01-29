#!/usr/bin/env python3

import os
import paramiko
import smtplib
import fcntl
import stat
from email.message import EmailMessage
from datetime import date
import sys

# ==========================================================
# CONFIG (ENVIRONMENT VARIABLES)
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR", "exports")
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("NOTIFY_EMAIL")

LOCK_FILE = "/tmp/sftp_csv_download.lock"
SUCCESS_MARKER = f"/tmp/sftp_success_email_{date.today().isoformat()}"

# ---------------- LOCK CLASS ----------------
class FileLock:
    def __init__(self, path):
        self.path = path
        self.fp = None

    def acquire(self):
        self.fp = open(self.path, "w")
        try:
            fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another sync is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            self.fp.close()

# ---------------- EMAIL ----------------
def send_email(file_count, success=True, error_msg=None):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    if success:
        msg["Subject"] = "SFTP CSV Sync – Daily Success Summary"
        msg.set_content(
            f"SFTP CSV incremental sync completed successfully.\n"
            f"New/updated CSV files downloaded: {file_count}\n"
            f"Date: {date.today().isoformat()}"
        )
    else:
        msg["Subject"] = "SFTP CSV Sync – FAILURE Alert"
        msg.set_content(
            f"SFTP CSV incremental sync FAILED.\n"
            f"Error: {error_msg}\n"
            f"Date: {date.today().isoformat()}"
        )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print("Failed to send email:", e)

# ---------------- DOWNLOAD ----------------
def download_latest_csv():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    latest_file = None
    latest_date = None
    downloaded = 0
    local_path = os.path.join(LOCAL_DIR, "raw_data.csv")

    transport = paramiko.Transport((SFTP_HOST, 22))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        sftp.chdir(REMOTE_DIR)
        files = [f for f in sftp.listdir() if f.lower().endswith(".csv")]

        # Identify latest file based on date in filename: EventsExports_YYYY_MM_DD_*.csv
        for f in files:
            parts = f.rstrip(".csv").split("_")
            try:
                file_date = date(int(parts[1]), int(parts[2]), int(parts[3]))
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = f
            except Exception:
                continue

        if latest_file:
            remote_path = os.path.join(sftp.getcwd(), latest_file)
            # Skip download if local exists and size matches
            attr = sftp.stat(remote_path)
            if os.path.exists(local_path) and os.path.getsize(local_path) == attr.st_size:
                print(f"Latest file {latest_file} already up-to-date.")
            else:
                temp_path = local_path + ".tmp"
                sftp.get(remote_path, temp_path)
                os.replace(temp_path, local_path)
                downloaded = 1
                print(f"Downloaded latest file: {latest_file}")
        else:
            print("No dated CSV files found.")

    finally:
        sftp.close()
        transport.close()

    return downloaded, local_path

# ---------------- MAIN ----------------
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()
    try:
        downloaded, local_path = download_latest_csv()
        if downloaded:
            send_email(downloaded, success=True)
        print(f"Local file path: {local_path}")

        # ✅ Output for GitHub Actions
        print(f"::set-output name=NEW_FILES_DOWNLOADED::{downloaded}")

    except Exception as e:
        send_email(0, success=False, error_msg=str(e))
        print("Error during sync:", e)
        print(f"::set-output name=NEW_FILES_DOWNLOADED::0")
        raise
    finally:
        lock.release()

if __name__ == "__main__":
    main()
