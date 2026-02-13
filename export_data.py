#!/usr/bin/env python3
import os
<<<<<<< Updated upstream
import re
import paramiko
import pandas as pd
import logging
import hashlib
import smtplib
from email.message import EmailMessage
from datetime import datetime   # <-- make sure this is here
=======
import paramiko
import smtplib
import portalocker
import ssl
from email.message import EmailMessage
from datetime import date, datetime
import sys
import re
import tempfile
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

# ==========================================================
# CONFIG (ENVIRONMENT VARIABLES)
# ==========================================================
<<<<<<< Updated upstream
<<<<<<< Updated upstream
SFTP_HOST = os.getenv("SFTP_HOST")
SFTP_PORT = 22
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
=======
SFTP_HOST ='83.149.119.154' #os.getenv("SFTP_HOST")
SFTP_USERNAME = 'events-eusee.hivos.o_iwfvvmfr82h'  #os.getenv("SFTP_USERNAME") 
SFTP_PASSWORD = '~Po7Rpdi9&oY3wkr' #os.getenv("SFTP_PASSWORD")
>>>>>>> Stashed changes
=======
SFTP_HOST ='83.149.119.154' #os.getenv("SFTP_HOST")
SFTP_USERNAME = 'events-eusee.hivos.o_iwfvvmfr82h'  #os.getenv("SFTP_USERNAME") 
SFTP_PASSWORD = '~Po7Rpdi9&oY3wkr' #os.getenv("SFTP_PASSWORD")
>>>>>>> Stashed changes
REMOTE_DIR = os.getenv("SFTP_REMOTE_DIR") or "exports"
LOCAL_DIR = os.getenv("LOCAL_DIR", "data")


# ---------------- EMAIL CONFIG ----------------
SMTP_SERVER = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM")
EMAIL_TO = os.getenv("NOTIFY_EMAIL")
EMAIL_SUBJECT = " Data Download Update Notification"

RAW_FILENAME = "raw_data.csv"
FINAL_FILENAME = "output_final.csv"
CHANGELOG_FILENAME = "change_log.csv"

os.makedirs(LOCAL_DIR, exist_ok=True)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- EMAIL FUNCTION ----------------
def send_update_email(new_rows_count, latest_file, local_path):
    """
    Sends a professional HTML email summarizing the SFTP CSV update.
    Only sends if new_rows_count > 0.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # HTML email body
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height:1.5; color:#333;">
        <h2 style="color:green;">✅ SFTP Data Sync Successful</h2>
        <p>The scheduled SFTP data synchronization completed successfully.</p>
        <table style="border-collapse: collapse; width: 600px;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Status</td>
                <td style="padding: 8px; border: 1px solid #ccc;">SUCCESS</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Downloaded file</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{latest_file}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Saved as</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{local_path}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">New updates added</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{new_rows_count}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc; font-weight:bold;">Date</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{now}</td>
            </tr>
        </table>
        <p style="margin-top: 20px;">This is an automated notification from the SFTP Data Sync system.</p>
    </body>
    </html>
    """
=======
=======
>>>>>>> Stashed changes
LOCK_FILE = "/tmp/sftp_csv_download.lock"
SUCCESS_MARKER = f"/tmp/sftp_success_email_{date.today().isoformat()}"

# ==========================================================
# FILE LOCK
# ==========================================================
class FileLock:
    def __init__(self, filename="sftp_csv_download.lock"):
        # Use a safe temp folder
        temp_dir = tempfile.gettempdir()
        # Make sure the folder exists
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.path = os.path.join(temp_dir, filename)
        self.fp = None

    def acquire(self):
        # Ensure parent folder exists just in case
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fp = open(self.path, "w")
        try:
            portalocker.lock(self.fp, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            print("Another instance is already running. Exiting.")
            sys.exit(0)

    def release(self):
        if self.fp:
            try:
                portalocker.unlock(self.fp)
                self.fp.close()
            except Exception:
                pass
# ==========================================================
# EMAIL CORE (HTML + TEXT)
# ==========================================================
def _send_email(subject, text_body, html_body):
    required_vars = {
        "SMTP_HOST": SMTP_HOST,
        "SMTP_USER": SMTP_USER,
        "SMTP_PASS": SMTP_PASSWORD,
        "ALERT_EMAIL_FROM": ALERT_EMAIL_FROM,
        "NOTIFY_EMAIL": ALERT_EMAIL_TO,
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        # Gracefully skip without spamming, just log
        print(f"Email skipped: missing {len(missing)} SMTP/alert config(s).")
        return False
>>>>>>> Stashed changes

    msg = EmailMessage()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO 
    msg['Subject'] = EMAIL_SUBJECT
    msg.add_alternative(html_body, subtype='html')  # use HTML email

    # Try STARTTLS first
    try:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"Update email sent successfully: {new_rows_count} new rows.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# ---------------- SFTP CONNECTION ----------------
transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
sftp = paramiko.SFTPClient.from_transport(transport)
logging.info("Connected to SFTP.")

# ---------------- LIST FILES ----------------
remote_files = sftp.listdir(REMOTE_DIR)
csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

def extract_date(filename):
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    return match.group(1) if match else None
=======
=======
>>>>>>> Stashed changes
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        print("STARTTLS email failed:", e)

    # Fallback to SSL
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, 465, context=context, timeout=20) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully via SSL to {ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        print("SSL email failed:", e)

    print("Email not sent due to configuration or connection issues.")
    return False


# ==========================================================
# SUCCESS EMAIL
# ==========================================================
def send_success_email(file_path):
    if not file_path or os.path.exists(SUCCESS_MARKER):
        return
>>>>>>> Stashed changes

csv_files_with_dates = [(f, extract_date(f)) for f in csv_files if extract_date(f)]
if not csv_files_with_dates:
    logging.error("No CSV files with dates found.")
    sftp.close()
    transport.close()
    exit()

<<<<<<< Updated upstream
<<<<<<< Updated upstream
latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
remote_path = f"{REMOTE_DIR}/{latest_file}"
local_path = os.path.join(LOCAL_DIR, RAW_FILENAME)

# ---------------- DOWNLOAD ----------------
remote_size = sftp.stat(remote_path).st_size
if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
    logging.info(f"{RAW_FILENAME} is already up to date. Skipping download.")
else:
    sftp.get(remote_path, local_path)
    logging.info(f"Downloaded latest file: {latest_file}")

# ---------------- LOAD RAW CSV ----------------
df_raw = pd.read_csv(local_path).fillna("")

# Rename columns to match final CSV
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
=======
=======
>>>>>>> Stashed changes
File downloaded: {os.path.basename(file_path)}
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

# ==========================================================
# FAILURE EMAIL
# ==========================================================
def send_failure_email(error_msg):
    text_body = f"""
SFTP Data Sync – FAILURE
>>>>>>> Stashed changes

# Clean enabling-principle column
if "enabling-principle" in df_raw.columns:
    df_raw["enabling-principle"] = (
        df_raw["enabling-principle"]
        .astype(str)
        .str.replace(r"\|", ",", regex=True)
        .str.replace(r"\s*,\s*", ",", regex=True)
    )

<<<<<<< Updated upstream
logging.info(f"Columns renamed and cleaned in {RAW_FILENAME}")
=======
Date: {date.today().isoformat()}
"""

    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;">
  <h2 style="color:#c62828;">❌ SFTP Data Sync Failed</h2>
  <p>The scheduled SFTP data synchronization did not complete successfully.</p>
>>>>>>> Stashed changes

# ---------------- LOAD EXISTING FINAL CSV ----------------
final_path = os.path.join(LOCAL_DIR, FINAL_FILENAME)
if os.path.exists(final_path):
    df_final = pd.read_csv(final_path).fillna("")
else:
    df_final = pd.DataFrame(columns=df_raw.columns)

<<<<<<< Updated upstream
# ---------------- GENERATE UNIQUE ID ----------------
def normalize_text(series):
    return series.astype(str).str.strip().str.lower().str.replace(r"[\n\r]+", " ", regex=True)

df_raw["_uid"] = normalize_text(df_raw["post_title"]).apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
df_final["_uid"] = normalize_text(df_final["post_title"]).apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())

# ---------------- FILTER NEW ROWS ----------------
new_rows = df_raw[~df_raw["_uid"].isin(df_final["_uid"])].copy()

if not new_rows.empty:
    # Append new rows
    combined_df = pd.concat([df_final, new_rows], ignore_index=True)
    combined_df.to_csv(final_path, index=False)
    logging.info(f"New rows appended: {len(new_rows)}")

    # Update change log
    change_log_path = os.path.join(LOCAL_DIR, CHANGELOG_FILENAME)
    new_rows.to_csv(change_log_path, mode='a', header=not os.path.exists(change_log_path), index=False)
    logging.info(f"Change log updated with {len(new_rows)} rows.")

    # Send professional email only if new rows exist
    send_update_email(len(new_rows), latest_file, local_path)
else:
    logging.info("No new rows to append. Email not sent.")

# ---------------- CLOSE SFTP ----------------
sftp.close()
transport.close()
logging.info("SFTP session closed. Incremental update completed successfully.")

=======
  <ul>
    <li>Check SFTP credentials</li>
    <li>Verify SMTP configuration</li>
    <li>Review GitHub Actions logs</li>
  </ul>

  <hr>
  <p style="font-size:12px;color:#777;">Automated alert – Data Pipeline</p>
</body>
</html>
"""

    _send_email(
        "ALERT: SFTP Data Sync Failed",
        text_body.strip(),
        html_body,
    )

# ==========================================================
# ==========================================================
# DOWNLOAD LATEST CSV
# ==========================================================
def download_latest_csv(retain_last=4, force_download=False):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    latest_file = None
    local_path = None
    downloaded = 0

    try:
        print(f"Connecting to SFTP {SFTP_HOST} as {SFTP_USERNAME}...")
        transport = paramiko.Transport((SFTP_HOST, 22))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.chdir(REMOTE_DIR)
        except IOError:
            print(f"Remote directory '{REMOTE_DIR}' not found, using current dir.")
            sftp.chdir(".")

        csv_files = [f for f in sftp.listdir() if f.lower().endswith(".csv")]
        print("Remote CSV files found:", csv_files)

        date_pattern = re.compile(r".*?(\d{4}_\d{2}_\d{2}|\d{8}).*\.csv$")
        latest_date = None
        for f in csv_files:
            match = date_pattern.match(f)
            if match:
                date_str = match.group(1)
                if "_" in date_str:
                    d = datetime.strptime(date_str, "%Y_%m_%d").date()
                else:
                    d = datetime.strptime(date_str, "%Y%m%d").date()
                if not latest_date or d > latest_date:
                    latest_date = d
                    latest_file = f

        if not latest_file:
            print("No matching CSV files found on SFTP.")
            return 0, None, None

        print("Latest CSV detected:", latest_file, "with date", latest_date)

        local_path = os.path.join(LOCAL_DIR, f"raw_data_{latest_date}.csv")
        remote_path = os.path.join(sftp.getcwd(), latest_file)

        if not os.path.exists(local_path) or force_download:
            print(f"Downloading '{latest_file}' to '{local_path}'...")
            temp_path = local_path + ".tmp"
            sftp.get(remote_path, temp_path)
            os.replace(temp_path, local_path)
            remote_mtime = sftp.stat(remote_path).st_mtime
            os.utime(local_path, (remote_mtime, remote_mtime))
            downloaded = 1
            print("Download complete.")
        else:
            print(f"Local file '{local_path}' already exists. Skipping download.")

        # Cleanup old CSVs
        all_local_csvs = sorted(
            [f for f in os.listdir(LOCAL_DIR) if f.lower().endswith(".csv")],
            key=lambda x: re.search(r"(\d{4}_\d{2}_\d{2}|\d{8})", x).group(1)
            if re.search(r"(\d{4}_\d{2}_\d{2}|\d{8})", x) else "",
            reverse=True
        )
        for old_file in all_local_csvs[retain_last:]:
            try:
                os.remove(os.path.join(LOCAL_DIR, old_file))
                print(f"Removed old CSV: {old_file}")
            except Exception as e:
                print(f"Failed to remove {old_file}: {e}")

    except Exception as e:
        raise RuntimeError(f"SFTP download failed: {e}")
    finally:
        try:
            sftp.close()
            transport.close()
        except Exception:
            pass

    return downloaded, latest_file, local_path

# ==========================================================
# MAIN WORKFLOW
# ==========================================================
def main():
    lock = FileLock()
    lock.acquire()

    downloaded = 0
    latest_file = None
    local_path = None

    try:
        downloaded, latest_file, local_path = download_latest_csv(retain_last=2, force_download=True)

        if downloaded:
            send_success_email(local_path)
        else:
            print("No new CSV file detected. Exiting workflow.")
            if GITHUB_ENV:
                with open(GITHUB_ENV, "a") as f:
                    f.write("NEW_FILES_DOWNLOADED=0\n")
            sys.exit(0)

    except Exception as e:
        send_failure_email(str(e))
        sys.exit(1)
    finally:
        lock.release()

    if GITHUB_ENV:
        with open(GITHUB_ENV, "a") as f:
            f.write(f"NEW_FILES_DOWNLOADED={downloaded}\n")

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
<<<<<<< Updated upstream
    main()
>>>>>>> Stashed changes
=======
    main()
>>>>>>> Stashed changes
