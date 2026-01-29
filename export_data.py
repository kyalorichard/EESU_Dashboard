#!/usr/bin/env python3

import os
import sys
import stat
import fcntl
import paramiko
import smtplib
import traceback
from email.message import EmailMessage
from datetime import date, datetime
import re

# ==========================================================
# CONFIG (ENV VARS)
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR", "exports")

LOCAL_DIR = os.getenv("LOCAL_DIR", "data")
LOCAL_FILE = os.path.join(LOCAL_DIR, "raw_data.csv")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("NOTIFY_EMAIL")

LOCK_FILE = "/tmp/sftp_csv_download.lock"

# ==========================================================
# FILE LOCK
# ==========================================================
class FileLock:
    def __init__(self, path):
        self.path = path
        self.fp = None

    def acquire(self):
        self.fp = open(self.path, "w")
        try:
            fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another sync already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            self.fp.close()

# ==========================================================
# EMAIL
# ==========================================================
def send_email(subject, body):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

def send_success_email(file_updated):
    if not file_updated:
        return

    send_email(
        subject="SFTP CSV Sync – Success",
        body=(
            "SFTP incremental sync completed successfully.\n\n"
            "raw_data.csv has been created or updated.\n"
            f"Date: {date.today().isoformat()}"
        )
    )

def send_failure_email(error):
    send_email(
        subject="❌ SFTP CSV Sync – FAILED",
        body=(
            "The SFTP CSV incremental sync has FAILED.\n\n"
            f"Timestamp (UTC): {datetime.utcnow().isoformat()}\n\n"
            "Error:\n"
            f"{error}\n\n"
            "Stack trace:\n"
            f"{traceback.format_exc()}"
        )
    )

# ==========================================================
# DATE EXTRACTION (robust for your filenames)
# ==========================================================
def extract_date(filename):
    """
    Extracts YYYY_MM_DD from filenames like:
    EventsExports_2026_01_26_1.csv
    """
    match = re.search(r"(\d{4}_\d{2}_\d{2})", filename)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, "%Y_%m_%d")
        except ValueError:
            return None
    return None

# ==========================================================
# DOWNLOAD LATEST CSV ONLY (with debug)
# ==========================================================
def download_latest_csv():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    updated = False

    transport = paramiko.Transport((SFTP_HOST, 22))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        try:
            sftp.chdir(REMOTE_DIR)
        except IOError:
            sftp.chdir(".")

        # ================= DEBUG: list all remote files =================
        remote_files = sftp.listdir()
        print("=== DEBUG: Remote directory listing ===")
        for f in remote_files:
            print(f)
        print("=== END DEBUG ===")

        # pick latest dated CSV
        latest_file = None
        latest_date = None
        for filename in remote_files:
            if not filename.lower().endswith(".csv"):
                continue
            file_date = extract_date(filename)
            if not file_date:
                continue
            if not latest_date or file_date > latest_date:
                latest_date = file_date
                latest_file = filename

        if not latest_file:
            print("No dated CSV files found.")
            return False

        print(f"Latest remote file: {latest_file}")

        remote_path = f"{sftp.getcwd()}/{latest_file}"
        temp_path = LOCAL_FILE + ".tmp"

        attr = sftp.stat(remote_path)
        if not stat.S_ISREG(attr.st_mode):
            raise RuntimeError("Latest remote file is not a regular file")

        if os.path.exists(LOCAL_FILE):
            if os.path.getsize(LOCAL_FILE) == attr.st_size:
                print("raw_data.csv already up to date.")
                return False

        sftp.get(remote_path, temp_path)
        os.replace(temp_path, LOCAL_FILE)
        updated = True
        print("raw_data.csv updated")

    finally:
        sftp.close()
        transport.close()

    return updated

# ==========================================================
# MAIN
# ==========================================================
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()

    try:
        updated = download_latest_csv()
        send_success_email(updated)

        # Required for GitHub Actions
        print(f"NEW_FILES_DOWNLOADED={1 if updated else 0}")
        print("Local file path:", os.path.abspath(LOCAL_FILE))

    except Exception as e:
        print("ERROR:", e)
        send_failure_email(str(e))
        raise

    finally:
        lock.release()

if __name__ == "__main__":
    main()
