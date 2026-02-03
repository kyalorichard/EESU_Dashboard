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
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")

GITHUB_OUTPUT = os.getenv("GITHUB_OUTPUT")  # GitHub Actions output

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
# EMAIL FUNCTIONS
# ==========================================================
def _send_email(subject, text_body, html_body):
    missing = [
        k for k, v in {
            "SMTP_HOST": SMTP_HOST,
            "SMTP_USER": SMTP_USER,
            "SMTP_PASS": SMTP_PASSWORD,
            "ALERT_EMAIL_FROM": ALERT_EMAIL_FROM,
            "ALERT_EMAIL_TO": ALERT_EMAIL_TO,
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
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print("SMTP STARTTLS failed:", e)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, 465, context=context, timeout=20) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print("SMTP SSL failed:", e)

    return False


def send_success_email(file_name):
    if not file_name or os.path.exists(SUCCESS_MARKER):
        return

    text_body = f"""
SFTP Data Sync – SUCCESS

File downloaded: {file_name}
Stored as: {os.path.join(LOCAL_DIR, "raw_data.csv")}
Date: {date.today().isoformat()}
"""

    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
  <h2 style="color:#2e7d32;">✅ SFTP Data Sync Successful</h2>
  <p>The scheduled SFTP data synchronization completed successfully.</p>
  <table cellpadding="6" cellspacing="0">
    <tr><td><b>Status</b></td><td style="color:#2e7d32;">SUCCESS</td></tr>
    <tr><td><b>Downloaded file</b></td><td>{file_name}</td></tr>
    <tr><td><b>Saved as</b></td><td><code>{os.path.join(LOCAL_DIR, "raw_data.csv")}</code></td></tr>
    <tr><td><b>Date</b></td><td>{date.today().isoformat()}</td></tr>
  </table>
</body>
</html>
"""
    if _send_email("SFTP Data Sync – Completed Successfully", text_body.strip(), html_body):
        open(SUCCESS_MARKER, "w").close()


def send_failure_email(error_msg):
    text_body = f"SFTP Data Sync – FAILURE\n\nError:\n{error_msg}\nDate: {date.today().isoformat()}"
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
  <h2 style="color:#c62828;">❌ SFTP Data Sync Failed</h2>
  <pre style="background:#f8f8f8;padding:10px;border:1px solid #ddd;">{error_msg}</pre>
</body>
</html>
"""
    _send_email("ALERT: SFTP Data Sync Failed", text_body, html_body)


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

        remote_path = os.path.join(sftp.getcwd(), latest_file)
        local_path = os.path.join(LOCAL_DIR, "raw_data.csv")
        temp_path = local_path + ".tmp"

        attr = sftp.stat(remote_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) != attr.st_size:
            sftp.get(remote_path, temp_path)
            os.replace(temp_path, local_path)
            downloaded = 1

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
    except Exception as e:
        send_failure_email(str(e))
    finally:
        lock.release()

    # Export to GitHub Actions output
    if GITHUB_OUTPUT:
        with open(GITHUB_OUTPUT, "a") as f:
            f.write(f"NEW_FILES_DOWNLOADED={downloaded}\n")


if __name__ == "__main__":
    main()
