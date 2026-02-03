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
LAST_FILE_RECORD = os.path.join(LOCAL_DIR, "last_downloaded.txt")  # tracks last file

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")

ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
ALERT_EMAIL_TO = os.getenv("NOTIFY_EMAIL")

GITHUB_ENV = os.getenv("GITHUB_ENV")

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
            print("Another sync is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            self.fp.close()

# ==========================================================
# EMAIL CORE
# ==========================================================
def _send_email(subject, text_body, html_body):
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
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print("STARTTLS failed:", e)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, 465, context=context, timeout=20) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print("SSL failed:", e)

    return False

# ==========================================================
# SUCCESS / FAILURE EMAIL
# ==========================================================
def send_success_email(file_name):
    text_body = f"SFTP Data Sync – SUCCESS\n\nFile downloaded: {file_name}\nSaved as: data/raw_data.csv\nDate: {date.today().isoformat()}"
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
<h2 style="color:#2e7d32;">✅ SFTP Data Sync Successful</h2>
<p>File downloaded: {file_name}</p>
<p>Saved as: <code>data/raw_data.csv</code></p>
<p>Date: {date.today().isoformat()}</p>
</body>
</html>
"""
    _send_email("SFTP Data Sync – Completed Successfully", text_body, html_body)

def send_failure_email(error_msg):
    text_body = f"SFTP Data Sync – FAILURE\n\nError:\n{error_msg}\nDate: {date.today().isoformat()}"
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
<h2 style="color:#c62828;">❌ SFTP Data Sync Failed</h2>
<pre>{error_msg}</pre>
</body>
</html>
"""
    _send_email("ALERT: SFTP Data Sync Failed", text_body, html_body)

# ==========================================================
# DOWNLOAD LOGIC (only if new)
# ==========================================================
def download_latest_csv():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    latest_file = None

    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            sftp.chdir(REMOTE_DIR)
        except IOError:
            sftp.chdir(".")

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
            return 0, None

        # Check if latest_file is same as last downloaded
        last_file = None
        if os.path.exists(LAST_FILE_RECORD):
            with open(LAST_FILE_RECORD) as f:
                last_file = f.read().strip()

        if latest_file == last_file:
            print(f"No new file. Latest file '{latest_file}' already downloaded.")
            return 0, latest_file

        # Download file
        remote_path = os.path.join(sftp.getcwd(), latest_file)
        local_path = os.path.join(LOCAL_DIR, "raw_data.csv")
        temp_path = local_path + ".tmp"
        sftp.get(remote_path, temp_path)
        os.replace(temp_path, local_path)

        # Save latest filename
        with open(LAST_FILE_RECORD, "w") as f:
            f.write(latest_file)

        return 1, latest_file

    except Exception as e:
        raise RuntimeError(f"SFTP download failed: {e}")
    finally:
        try:
            sftp.close()
            transport.close()
        except Exception:
            pass

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
    except Exception as e:
        send_failure_email(str(e))
    finally:
        lock.release()

    if GITHUB_ENV:
        with open(GITHUB_ENV, "a") as f:
            f.write(f"NEW_FILES_DOWNLOADED={downloaded}\n")

if __name__ == "__main__":
    main()
