import os
import paramiko
import logging
from datetime import datetime

# ---------- CONFIG ----------
HOST = "83.149.119.154"
USERNAME = "events-eusee.hivos.o_iwfvvmfr82h@83.149.119.154"
PASSWORD = "~Po7Rpdi9&oY3wkr"
REMOTE_DIR = "/exports"
LOCAL_DIR = "data"
LOG_DIR = "logs"

# ---------- SETUP ----------
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"sftp_download_{datetime.now().date()}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("Starting scheduled SFTP download")

    transport = None
    sftp = None

    try:
        transport = paramiko.Transport((HOST, 22))
        transport.connect(username=USERNAME, password=PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        sftp.chdir(REMOTE_DIR)

        for file in sftp.listdir():
            if file.lower().endswith(".csv"):
                remote_path = f"{REMOTE_DIR}/{file}"
                local_path = os.path.join(LOCAL_DIR, file)

                logging.info(f"Downloading {file}")
                sftp.get(remote_path, local_path)

        logging.info("Download completed successfully")

    except Exception as e:
        logging.error(f"Download failed: {e}")
        raise

    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

if __name__ == "__main__":
    main()
