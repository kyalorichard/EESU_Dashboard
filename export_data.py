#!/usr/bin/env python3

import os
import paramiko
import smtplib
import fcntl
import ssl
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
            print("Another sync is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            self.fp.close()

# ==========================================================
# EMAIL CORE
# ==========================================================
def _send_email(subject, body):
    missing = [
        k for k, v in {
            "SMTP_HOST": SMTP_HOST,
            "SMTP_USER": SMTP_USER,
            "SMTP_PASS": SMTP_PASSWORD,
            "ALERT_EMAIL_FROM": ALERT_EMAIL_FROM,
            "NOTIFY_EMAIL": ALERT_EMAIL_TO,
        }.items() if not v
    ]

    if missing:
        print("Email skipped. Missing env vars:", ", ".join(missing))
        return False

    msg = EmailMessage()
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    # ---- STARTTLS ----
    try:
        print(f"Attempting SMTP STARTTLS on {SMTP_HOST}:{SMTP_PORT}")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.set_debuglevel(1)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("Email sent via STARTTLS")
        return True
    except Exception as e:
        print("STARTTLS failed:", e)

    # ---- SSL fallback ----
    try:
        print("Attempting SMTP SSL on port 465")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, 465, context=context, timeout=20) as server:
            server.set_debuglevel(1)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("Email sent via SSL")
        return True
    except Exception as e:
        print("SSL failed:", e)

    return False


def send_success_email(file_name):
    if not file_name:
        print("Success email skipped: no file.")
        return
    if os.path.exists(SUCCESS_MARKER):
        print("Success email already sent today.")
        return

    sent = _send_email(
        subject="SFTP CSV Sync – Success",
        body=(
            "SFTP CSV incremental sync completed successfully.\n\n"
            f"Downloaded file: {file_name}\n"
            f"Date: {date.today().isoformat()}"
        )
    )

    if sent:
        open(SUCCESS_MARKER, "w").close()


def send_failure_email(error_msg):
    _send_email(
        subject="SFTP CSV Sync – FAILURE",
        body=(
            "SFTP CSV incremental sync FAILED.\n\n"
            f"Error:\n{error_msg}\n\n"
            f"Date: {date.today().isoformat()}"
        )
    )

# ==========================================================
# DOWNLOAD LOGIC
# ==========================================================
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
            print(f"Remote folder '{REMOTE_DIR}' not found. Using home.")
            sftp.chdir(".")

        print("Current remote folder:", sftp.getcwd())

        csv_files = [f for f in sftp.listdir() if f.lower().endswith(".csv")]
        date_pattern = re.compile(r".*?(\d{4}_\d{2}_\d{2}).*\.csv$")

        latest_date = None
        for f in csv_files:
            match = date_pattern.match(f)
            if match:
                d = datetime.strptime(match.group(1), "%Y_%m_%d").date()
                if not latest_date or d > latest_date:
                    latest_date = d
                    latest_file = f

        if not latest_file:
            print("No dated CSV files found.")
            return 0, None

        remote_path = os.path.join(sftp.getcwd(), latest_file)
        local_path = os.path.join(LOCAL_DIR, "raw_data.csv")
        temp_path = local_path + ".tmp"

        attr = sftp.stat(remote_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) != attr.st_size:
            sftp.get(remote_path, temp_path)
            os.replace(temp_path, local_path)
            downloaded = 1
            print(f"Downloaded latest file: {latest_file}")
        else:
            print(f"Latest file already exists locally: {latest_file}")

    except Exception as e:
        raise RuntimeError(f"SFTP download failed: {e}")
    finally:
        try:
            sftp.close()
            transport.close()
        except Exception:
            pass

    return downloaded, latest_file

# ==========================================================
# MAIN
# ==========================================================
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()

    downloaded = 0
    latest_file = None

    try:
        downloaded, latest_file = download_latest_csv()
        if downloaded:
            send_success_email(latest_file)
        else:
            print("No new CSV downloaded.")
    except Exception as e:
        print("Error during sync:", e)
        send_failure_email(str(e))
    finally:
        lock.release()

    # GitHub Actions output
    if GITHUB_ENV:
        with open(GITHUB_ENV, "a") as f:
            f.write(f"NEW_FILES_DOWNLOADED={downloaded}\n")

if __name__ == "__main__":
    main()
