#!/usr/bin/env python3

import os
import sys
import paramiko
import fcntl
import smtplib
from email.message import EmailMessage
from datetime import date

# ==========================================================
# CONFIG (ENVIRONMENT VARIABLES)
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR", "/exports")
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("NOTIFY_EMAIL")

LOCK_FILE = "/tmp/sftp_csv_download.lock"
SUCCESS_MARKER = f"/tmp/sftp_success_email_{date.today().isoformat()}"

# ==========================================================
# VALIDATION
# ==========================================================
if not all([SFTP_HOST, SFTP_USERNAME, SFTP_PASSWORD]):
    print("Missing required SFTP environment variables")
    sys.exit(1)

# ==========================================================
# LOCKING (PREVENT OVERLAPS)
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
            print("Another sync is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            self.fp.close()

# ==========================================================
# EMAIL FUNCTIONS
# ==========================================================
def notify_failure(error_msg):
    if not all([
        SMTP_HOST,
        SMTP_USER,
        SMTP_PASSWORD,
        ALERT_EMAIL_FROM,
        ALERT_EMAIL_TO
    ]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = "SFTP CSV Incremental Sync FAILED"
    msg.set_content(error_msg)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def notify_success_summary(file_count):
    # Suppress success email if no files changed
    if file_count == 0:
        return

    # Only once per day
    if os.path.exists(SUCCESS_MARKER):
        return

    if not all([
        SMTP_HOST,
        SMTP_USER,
        SMTP_PASSWORD,
        ALERT_EMAIL_FROM,
        ALERT_EMAIL_TO
    ]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = "SFTP CSV Sync – Daily Success Summary"
    msg.set_content(
        f"SFTP CSV incremental sync completed successfully.\n\n"
        f"New or updated CSV files downloaded: {file_count}\n"
        f"Date: {date.today().isoformat()}"
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

    open(SUCCESS_MARKER, "w").close()

# ==========================================================
# CORE SYNC LOGIC
# ==========================================================
def download_csv_files():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    downloaded = 0

    transport = paramiko.Transport((SFTP_HOST, 22))
    transport.connect(
        username=SFTP_USERNAME,
        password=SFTP_PASSWORD
    )
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        sftp.chdir(SFTP_REMOTE_DIR)

        for filename in sftp.listdir():
            if not filename.lower().endswith(".csv"):
                continue

            remote_path = f"{SFTP_REMOTE_DIR}/{filename}"
            local_path = os.path.join(LOCAL_DIR, filename)
            temp_path = local_path + ".tmp"

            remote_stat = sftp.stat(remote_path)

            # Incremental check
            if os.path.exists(local_path):
                if os.path.getsize(local_path) == remote_stat.st_size:
                    continue

            # Atomic download
            sftp.get(remote_path, temp_path)
            os.replace(temp_path, local_path)
            downloaded += 1

    finally:
        sftp.close()
        transport.close()

    return downloaded

# ==========================================================
# MAIN
# ==========================================================
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()

    try:
        file_count = download_csv_files()
        notify_success_summary(file_count)

    except Exception as e:
        notify_failure(str(e))
        raise

    finally:
        lock.release()

if __name__ == "__main__":
    main()
