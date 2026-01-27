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
def send_success_email(file_count):
    if file_count == 0 or os.path.exists(SUCCESS_MARKER):
        return
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
        return

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = "SFTP CSV Sync – Daily Success Summary"
    msg.set_content(
        f"SFTP CSV incremental sync completed successfully.\n"
        f"New/updated CSV files downloaded: {file_count}\n"
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

# ---------------- DOWNLOAD ----------------
def download_csv_files():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    downloaded = 0

    transport = paramiko.Transport((SFTP_HOST, 22))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        # Access remote folder
        try:
            print(f"Trying to access remote folder: '{REMOTE_DIR}'")
            sftp.chdir(REMOTE_DIR)
        except IOError:
            print(f"Folder '{REMOTE_DIR}' not found. Using home directory instead.")
            sftp.chdir(".")
        print("Current remote folder:", sftp.getcwd())
        print("Remote files/folders:", sftp.listdir())

        # Download CSVs
        for filename in sftp.listdir():
            if not filename.lower().endswith(".csv"):
                continue

            remote_path = os.path.join(sftp.getcwd(), filename)
            local_path = os.path.join(LOCAL_DIR, filename)
            temp_path = local_path + ".tmp"

            try:
                attr = sftp.stat(remote_path)

                # Check if it's a regular file
                if not stat.S_ISREG(attr.st_mode):
                    print(f"Skipping {filename} (not a regular file)")
                    continue

                # Skip if unchanged
                if os.path.exists(local_path) and os.path.getsize(local_path) == attr.st_size:
                    continue

                # Download to temp file first
                sftp.get(remote_path, temp_path)
                os.replace(temp_path, local_path)
                downloaded += 1
                print(f"Downloaded/Updated: {filename}")
            except Exception as e:
                print(f"Failed {filename}: {e}")

    finally:
        sftp.close()
        transport.close()

    return downloaded

# ---------------- MAIN ----------------
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()
    try:
        file_count = download_csv_files()
        send_success_email(file_count)
    except Exception as e:
        print("Error during sync:", e)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
