#!/usr/bin/env python3

import os
import paramiko
import smtplib
import fcntl
import stat
from email.message import EmailMessage
from datetime import date, datetime
import sys
import re

# ==========================================================
# CONFIG (ENVIRONMENT VARIABLES)
# ==========================================================
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR") or "exports"
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("NOTIFY_EMAIL")
GITHUB_ENV = os.getenv("GITHUB_ENV")

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
def send_success_email(file_name):
    if not file_name or os.path.exists(SUCCESS_MARKER):
        return
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = "SFTP CSV Sync – Daily Success Summary"
    msg.set_content(
        f"SFTP CSV incremental sync completed successfully.\n"
        f"Downloaded CSV: {file_name}\n"
        f"Date: {date.today().isoformat()}"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        open(SUCCESS_MARKER, "w").close()
    except Exception as e:
        print("Failed to send success email:", e)

def send_failure_email(error_msg):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = "SFTP CSV Sync – Failure Alert"
    msg.set_content(
        f"SFTP CSV incremental sync failed.\n"
        f"Error: {error_msg}\n"
        f"Date: {date.today().isoformat()}"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print("Failed to send failure email:", e)

# ---------------- DOWNLOAD ----------------
def download_latest_csv():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    latest_file = None
    downloaded = 0

    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.chdir(REMOTE_DIR)
        except IOError:
            print(f"Remote folder '{REMOTE_DIR}' not found. Using home directory.")
            sftp.chdir(".")
        print("Current remote folder:", sftp.getcwd())

        # Identify latest CSV by date in filename
        csv_files = [f for f in sftp.listdir() if f.lower().endswith(".csv")]
        date_pattern = re.compile(r".*?(\d{4}_\d{2}_\d{2}).*\.csv$")
        latest_date = None
        for f in csv_files:
            match = date_pattern.match(f)
            if match:
                file_date = datetime.strptime(match.group(1), "%Y_%m_%d").date()
                if not latest_date or file_date > latest_date:
                    latest_date = file_date
                    latest_file = f

        if not latest_file:
            print("No dated CSV files found.")
            return 0, None

        remote_path = os.path.join(sftp.getcwd(), latest_file)
        local_path = os.path.join(LOCAL_DIR, "raw_data.csv")
        temp_path = local_path + ".tmp"

        # Download only if file is new or different size
        attr = sftp.stat(remote_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) != attr.st_size:
            sftp.get(remote_path, temp_path)
            os.replace(temp_path, local_path)
            downloaded = 1
            print(f"Downloaded latest file: {latest_file}")
        else:
            print(f"Latest file already exists locally: {latest_file}")

    except Exception as e:
        raise RuntimeError(f"Failed to download CSV: {e}")
    finally:
        try:
            sftp.close()
            transport.close()
        except:
            pass

    return downloaded, latest_file

# ---------------- MAIN ----------------
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()
    try:
        downloaded, latest_file = download_latest_csv()
        if downloaded:
            send_success_email(latest_file)
        else:
            print("No new CSV downloaded.")
    except Exception as e:
        print("Error during sync:", e)
        send_failure_email(str(e))
        downloaded = 0
    finally:
        lock.release()

    # ✅ Write GitHub Actions environment variable
    if GITHUB_ENV:
        with open(GITHUB_ENV, "a") as f:
            f.write(f"NEW_FILES_DOWNLOADED={downloaded}\n")

if __name__ == "__main__":
    main()
