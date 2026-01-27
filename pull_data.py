import os
import paramiko
import re
import pandas as pd
import hashlib
import sys

# ---------------- CONFIG ----------------
REMOTE_DIR = "exports"
LOCAL_DIR = "data"
RAW_FILENAME = "raw_data.csv"

# ---------------- ENV VALIDATION ----------------
def require_env(name, default=None, cast=str):
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        return cast(value)
    except Exception:
        raise RuntimeError(f"Invalid value for {name}")

SFTP_HOST = require_env("SFTP_HOST")
SFTP_PORT = require_env("SFTP_PORT", 22, int)
SFTP_USERNAME = require_env("SFTP_USERNAME")
SFTP_PASSWORD = require_env("SFTP_PASSWORD")

# ---------------- HELPERS ----------------
def file_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

os.makedirs(LOCAL_DIR, exist_ok=True)
local_path = os.path.join(LOCAL_DIR, RAW_FILENAME)
old_hash = file_hash(local_path)

# ---------------- SFTP CONNECT ----------------
try:
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
except Exception as e:
    sys.exit(f"SFTP connection failed: {e}")

# ---------------- FIND LATEST CSV ----------------
try:
    remote_files = sftp.listdir(REMOTE_DIR)
except Exception as e:
    sys.exit(f"Failed to list remote directory: {e}")

csv_files = [f for f in remote_files if f.lower().endswith(".csv")]

def extract_date(filename):
    m = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    return m.group(1) if m else None

dated_files = [(f, extract_date(f)) for f in csv_files]
dated_files = [f for f in dated_files if f[1]]

if not dated_files:
    sys.exit("No dated CSV files found on SFTP")

latest_file = sorted(dated_files, key=lambda x: x[1], reverse=True)[0][0]
remote_path = f"{REMOTE_DIR}/{latest_file}"

# ---------------- DOWNLOAD (OVERWRITE) ----------------
try:
    sftp.get(remote_path, local_path)
except Exception as e:
    sys.exit(f"Download failed: {e}")

# ---------------- PROCESS DATA ----------------
try:
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
            .str.replace("|", ",", regex=False)
            .str.replace(r"\s*,\s*", ",", regex=True)
        )

    df.to_csv(local_path, index=False)

except Exception as e:
    sys.exit(f"Data processing failed: {e}")

# ---------------- CHANGE DETECTION ----------------
new_hash = file_hash(local_path)
data_changed = old_hash != new_hash

# ---------------- CLEANUP ----------------
sftp.close()
transport.close()

print("SFTP download and processing completed.")
print(f"Latest source file: {latest_file}")
print(f"Data changed: {data_changed}")

# Optional: expose for GitHub Actions
with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
    f.write(f"data_changed={str(data_changed).lower()}\n")
