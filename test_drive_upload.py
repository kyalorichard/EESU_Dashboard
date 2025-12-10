#!/usr/bin/env python3
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GDRIVE_JSON = os.getenv("GDRIVE_JSON")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")  # optional

def ensure_service_account_file():
    if not GDRIVE_JSON:
        print("[ensure_service_account_file] No credentials provided")
        return None
    # If it's a file path
    if os.path.exists(GDRIVE_JSON):
        return GDRIVE_JSON
    # If it's raw JSON
    try:
        parsed = json.loads(GDRIVE_JSON)
        path = os.path.join(OUTPUT_FOLDER, "service-account.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        return path
    except Exception as e:
        print(f"[ensure_service_account_file] Failed: {e}")
        return None

def get_drive_service():
    cred_path = ensure_service_account_file()
    if not cred_path or not os.path.exists(cred_path):
        raise FileNotFoundError(f"Service account file missing: {cred_path}")
    creds = service_account.Credentials.from_service_account_file(
        cred_path,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return build("drive", "v3", credentials=creds)

def upload_test_file():
    service = get_drive_service()
    test_file = os.path.join(OUTPUT_FOLDER, "drive_test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a Drive upload test.")
    media = MediaFileUpload(test_file, resumable=True)
    meta = {"name": "drive_test.txt"}
    if GDRIVE_FOLDER_ID:
        meta["parents"] = [GDRIVE_FOLDER_ID]
    file_id = service.files().create(body=meta, media_body=media, fields="id").execute().get("id")
    print(f"Upload successful! File ID: {file_id}")

if __name__ == "__main__":
    try:
        upload_test_file()
    except Exception as e:
        print(f"Drive upload failed: {e}")
