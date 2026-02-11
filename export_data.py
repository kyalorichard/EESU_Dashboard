#!/usr/bin/env python3

import os
import paramiko
import smtplib
import ssl
import portalocker
from email.message import EmailMessage
from datetime import date, datetime
import sys
import re
import pandas as pd
import traceback

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

LOCK_FILE = os.path.join(os.path.expanduser("~"), "sftp_csv_download.lock")
SUCCESS_MARKER = f"{os.path.expanduser('~')}/sftp_success_email_{date.today().isoformat()}"

CHUNK_SIZE = 100000  # rows per chunk for large CSVs

MASTER_CSV = os.path.join(LOCAL_DIR, "output_final.csv")

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
            portalocker.lock(self.fp, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            print("Another sync is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            portalocker.unlock(self.fp)
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


def send_success_email(file_path, new_rows=0):
    if not file_path or os.path.exists(SUCCESS_MARKER):
        return

    text_body = f"""
SFTP Data Sync – SUCCESS

File updated: {os.path.basename(file_path)}
New rows appended: {new_rows}
Stored as: {file_path}
Date: {date.today().isoformat()}
"""
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
  <h2 style="color:#2e7d32;">✅ SFTP Data Sync Successful</h2>
  <p>The scheduled SFTP data synchronization completed successfully.</p>

  <table cellpadding="6" cellspacing="0">
    <tr><td><b>Status</b></td><td style="color:#2e7d32;">SUCCESS</td></tr>
    <tr><td><b>Downloaded file</b></td><td>{os.path.basename(file_path)}</td></tr>
    <tr><td><b>New rows appended</b></td><td>{new_rows}</td></tr>
    <tr><td><b>Saved as</b></td><td><code>{file_path}</code></td></tr>
    <tr><td><b>Date</b></td><td>{date.today().isoformat()}</td></tr>
  </table>

  <p>No action is required.</p>
  <hr>
  <p style="font-size:12px;color:#777;">Automated notification – Data Pipeline</p>
</body>
</html>
"""
    if _send_email(
        "SFTP Data Sync – Completed Successfully",
        text_body.strip(),
        html_body,
    ):
        open(SUCCESS_MARKER, "w").close()


def send_failure_email(error_msg):
    text_body = f"""
SFTP Data Sync – FAILURE

Error:
{error_msg}

Date: {date.today().isoformat()}
"""
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
  <h2 style="color:#c62828;">❌ SFTP Data Sync Failed</h2>
  <p>The scheduled SFTP data synchronization did not complete successfully.</p>

  <pre style="background:#f8f8f8;padding:10px;border:1px solid #ddd;">
{error_msg}
  </pre>

  <ul>
    <li>Check SFTP credentials</li>
    <li>Verify SMTP configuration</li>
    <li>Review logs</li>
  </ul>

  <hr>
  <p style="font-size:12px;color:#777;">Automated alert – Data Pipeline</p>
</body>
</html>
"""
    _send_email("ALERT: SFTP Data Sync Failed", text_body.strip(), html_body)

# ==========================================================
# INCREMENTAL CSV UPDATE TO MASTER FILE
# ==========================================================
def download_latest_csv_to_master(chunk_size=CHUNK_SIZE):
    """
    Append only new rows from latest SFTP CSV to output_final.csv.
    Uses row hashes to avoid duplicates and preserve headers.
    """
    os.makedirs(LOCAL_DIR, exist_ok=True)
    new_rows_count = 0
    latest_file = None

    try:
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.chdir(REMOTE_DIR)
        except IOError:
            sftp.chdir(".")

        # Find latest CSV
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
            print("No matching CSV found. Available files:", csv_files)
            return 0, None, MASTER_CSV

        remote_path = os.path.join(sftp.getcwd(), latest_file)
        temp_path = MASTER_CSV + ".tmp"
        sftp.get(remote_path, temp_path)

        # Load existing hashes
        existing_hashes = set()
        file_exists = os.path.exists(MASTER_CSV)
        if file_exists:
            for chunk in pd.read_csv(MASTER_CSV, chunksize=chunk_size):
                for row in chunk.itertuples(index=False, name=None):
                    existing_hashes.add(hash(row))

        # Append new rows from remote CSV
        for chunk in pd.read_csv(temp_path, chunksize=chunk_size):
            new_rows = []
            for row in chunk.itertuples(index=False, name=None):
                h = hash(row)
                if h not in existing_hashes:
                    existing_hashes.add(h)
                    new_rows.append(row)

            if new_rows:
                df_new = pd.DataFrame(new_rows, columns=chunk.columns)
                if not file_exists:
                    df_new.to_csv(MASTER_CSV, mode='w', index=False)
                    file_exists = True
                else:
                    df_new.to_csv(MASTER_CSV, mode='a', index=False, header=False)
                new_rows_count += len(new_rows)

    except Exception as e:
        raise RuntimeError(f"SFTP download failed: {traceback.format_exc()}")
    finally:
        try:
            sftp.close()
            transport.close()
        except Exception:
            pass

    return new_rows_count, latest_file, MASTER_CSV

# ==========================================================
# MAIN
# ==========================================================
def main():
    lock = FileLock(LOCK_FILE)
    lock.acquire()

    downloaded = 0
    latest_file = None
    local_path = MASTER_CSV

    try:
        downloaded, latest_file, local_path = download_latest_csv_to_master()

        if downloaded:
            send_success_email(local_path, new_rows=downloaded)
        else:
            print("No new CSV entries detected. Exiting workflow.")
            if GITHUB_ENV:
                with open(GITHUB_ENV, "a") as f:
                    f.write("NEW_ROWS_DOWNLOADED=0\n")
            sys.exit(0)

    except Exception as e:
        send_failure_email(str(e))
        sys.exit(1)
    finally:
        lock.release()

    if GITHUB_ENV:
        with open(GITHUB_ENV, "a") as f:
            f.write(f"NEW_ROWS_DOWNLOADED={downloaded}\n")

if __name__ == "__main__":
    main()
