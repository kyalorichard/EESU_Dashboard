import os
import paramiko
import re
import pandas as pd

SFTP_HOST = os.environ["SFTP_HOST"]
SFTP_PORT = int(os.environ.get("SFTP_PORT", 22))
SFTP_USERNAME = os.environ["SFTP_USERNAME"]
SFTP_PASSWORD = os.environ["SFTP_PASSWORD"]

REMOTE_DIR = "exports"
LOCAL_DIR = "data"
RAW_FILENAME = "raw_data.csv"

os.makedirs(LOCAL_DIR, exist_ok=True)

transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
sftp = paramiko.SFTPClient.from_transport(transport)

remote_files = sftp.listdir(REMOTE_DIR)
csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

def extract_date(filename):
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    return match.group(1) if match else None

csv_files_with_dates = [(f, extract_date(f)) for f in csv_files]
csv_files_with_dates = [t for t in csv_files_with_dates if t[1] is not None]

if csv_files_with_dates:
    latest_file = sorted(csv_files_with_dates, key=lambda x: x[1], reverse=True)[0][0]
    remote_path = f"{REMOTE_DIR}/{latest_file}"
    local_path = os.path.join(LOCAL_DIR, RAW_FILENAME)
    sftp.get(remote_path, local_path)

    df = pd.read_csv(local_path)
    rename_map = {
        "Title": "post_title",
        "Content": "summary",
        "Date": "creation_date",
        "Countries": "alert-country",
        "Impact": "alert-impact",
        "Alert types": "alert-type",
        "Enabling principles": "enabling-principle"
    }
    df.rename(columns=rename_map, inplace=True)
    if "enabling-principle" in df.columns:
        df["enabling-principle"] = (
            df["enabling-principle"]
            .astype(str)
            .str.replace("|", ",")
            .str.replace(r"\s*,\s*", ",", regex=True)
        )
    df.to_csv(local_path, index=False)

sftp.close()
transport.close()
print("SFTP download and processing completed.")
