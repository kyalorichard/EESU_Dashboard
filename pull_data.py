"""
WordPress Incremental Sync Script (JSON output)
- Uses .env for configuration
- Fetches new/updated posts from MySQL
- Merges with cumulative CSV in S3
- Updates checkpoint
- Sends SES notifications
- Pushes CloudWatch metrics
- Outputs JSON for GitHub Actions
"""

import os
import pymysql
import boto3
import json
import logging
import time
from io import StringIO
import pandas as pd
from dotenv import load_dotenv

# ------------------- Load environment -------------------
load_dotenv()  # Load from .env file

S3_BUCKET = os.environ['S3_BUCKET_NAME']
S3_PREFIX = os.environ.get('S3_PREFIX', 'wp_incremental_csv')
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']
SES_EMAIL = os.environ.get('SES_EMAIL')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 500))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
RETRY_DELAY = float(os.environ.get('RETRY_DELAY', 2))
METRIC_NAMESPACE = os.environ.get('METRIC_NAMESPACE', 'WordPressSyncMetrics')

# ------------------- Setup clients and logging -------------------
s3 = boto3.client('s3')
ses = boto3.client('ses')
cloudwatch = boto3.client('cloudwatch')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CHECKPOINT_KEY = f"{S3_PREFIX}/checkpoint.json"
CUMULATIVE_CSV_KEY = f"{S3_PREFIX}/wordpress_cumulative.csv"

# ------------------- Retry decorator -------------------
def retry(func):
    def wrapper(*args, **kwargs):
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapper

# ------------------- AWS & DB wrappers -------------------
@retry
def s3_get_object(**kwargs):
    return s3.get_object(**kwargs)

@retry
def s3_put_object(**kwargs):
    return s3.put_object(**kwargs)

@retry
def ses_send_email(**kwargs):
    return ses.send_email(**kwargs)

@retry
def db_connect():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME)

@retry
def put_metric(name, value):
    cloudwatch.put_metric_data(
        Namespace=METRIC_NAMESPACE,
        MetricData=[{
            'MetricName': name,
            'Value': value,
            'Unit': 'Count' if name != "ExecutionTime" else 'Seconds'
        }]
    )

# ------------------- Utility functions -------------------
def get_last_sync_time():
    try:
        resp = s3_get_object(Bucket=S3_BUCKET, Key=CHECKPOINT_KEY)
        checkpoint = json.loads(resp['Body'].read())
        last_sync = checkpoint.get('last_sync')
        if last_sync:
            logger.info(f"Last sync time from checkpoint: {last_sync}")
        return last_sync
    except s3.exceptions.NoSuchKey:
        logger.info("No checkpoint found. Fetching all posts.")
        return None
    except Exception as e:
        logger.error(f"Error reading checkpoint: {e}")
        return None

def update_checkpoint(timestamp):
    try:
        s3_put_object(
            Bucket=S3_BUCKET,
            Key=CHECKPOINT_KEY,
            Body=json.dumps({"last_sync": timestamp})
        )
        logger.info(f"Updated checkpoint to {timestamp}")
    except Exception as e:
        logger.error(f"Failed to update checkpoint: {e}")

def send_email(subject, body):
    if not SES_EMAIL:
        logger.warning("SES_EMAIL not set. Skipping email notification.")
        return
    try:
        ses_send_email(
            Source=SES_EMAIL,
            Destination={'ToAddresses': [SES_EMAIL]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        )
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send SES email: {e}")

def fetch_posts_batch(cursor, last_modified, batch_size):
    if last_modified:
        cursor.execute(
            "SELECT ID, post_title, post_date, post_modified FROM wp_posts "
            "WHERE post_status='publish' AND post_type='post' AND post_modified > %s "
            "ORDER BY post_modified ASC LIMIT %s",
            (last_modified, batch_size)
        )
    else:
        cursor.execute(
            "SELECT ID, post_title, post_date, post_modified FROM wp_posts "
            "WHERE post_status='publish' AND post_type='post' "
            "ORDER BY post_modified ASC LIMIT %s",
            (batch_size,)
        )
    return cursor.fetchall()

# ------------------- Main Lambda handler -------------------
def lambda_handler(event, context):
    start_time = time.time()
    batch_count = 0
    try:
        last_sync = get_last_sync_time()

        # Load existing cumulative CSV
        df_existing = pd.DataFrame(columns=['ID', 'Title', 'Date', 'Modified'])
        try:
            resp = s3_get_object(Bucket=S3_BUCKET, Key=CUMULATIVE_CSV_KEY)
            df_existing = pd.read_csv(resp['Body'])
            logger.info("Loaded existing cumulative CSV from S3.")
        except s3.exceptions.NoSuchKey:
            logger.info("No cumulative CSV found. Creating new.")
        except Exception as e:
            logger.error(f"Error reading cumulative CSV: {e}")

        # Connect to DB
        conn = db_connect()
        cursor = conn.cursor()

        all_new_rows = []
        last_modified = last_sync

        # Batch fetch loop
        while True:
            rows = fetch_posts_batch(cursor, last_modified, BATCH_SIZE)
            if not rows:
                break

            df_batch = pd.DataFrame(rows, columns=['ID', 'Title', 'Date', 'Modified'])
            all_new_rows.append(df_batch)
            last_modified = df_batch['Modified'].max()
            batch_count += 1
            logger.info(f"Fetched batch {batch_count} with {len(df_batch)} posts; next batch starts after {last_modified}")

        cursor.close()
        conn.close()

        if not all_new_rows:
            logger.info("No new or updated posts to sync.")
            send_email("WordPress Sync Completed", "No new or updated posts since last sync.")
            put_metric("PostsSynced", 0)
            put_metric("BatchCount", 0)
            execution_time = time.time() - start_time
            put_metric("ExecutionTime", execution_time)
            return {
                "status": "no_new_posts"
            }

        # Merge all batches
        df_new = pd.concat(all_new_rows, ignore_index=True)
        df_cumulative = pd.concat([df_existing, df_new], ignore_index=True)
        df_cumulative.drop_duplicates(subset=['ID'], keep='last', inplace=True)
        df_cumulative.sort_values('Modified', inplace=True)

        # Upload cumulative CSV
        csv_buffer = StringIO()
        df_cumulative.to_csv(csv_buffer, index=False)
        s3_put_object(Bucket=S3_BUCKET, Key=CUMULATIVE_CSV_KEY, Body=csv_buffer.getvalue())
        logger.info(f"Cumulative CSV updated in S3: {CUMULATIVE_CSV_KEY}")

        # Update checkpoint
        checkpoint_time = df_new['Modified'].max()
        update_checkpoint(checkpoint_time)

        # Send success email
        send_email(
            "WordPress Sync Successful",
            f"Successfully synced {len(df_new)} new or updated posts.\nTotal posts: {len(df_cumulative)}"
        )

        # Push metrics
        put_metric("PostsSynced", len(df_new))
        put_metric("TotalPosts", len(df_cumulative))
        put_metric("BatchCount", batch_count)
        execution_time = time.time() - start_time
        put_metric("ExecutionTime", execution_time)

        # Return JSON output
        return {
            "status": "success",
            "new_rows": len(df_new),
            "total_rows": len(df_cumulative),
            "batches": batch_count,
            "execution_time": execution_time
        }

    except Exception as e:
        logger.error(f"Error in WordPress sync: {e}")
        send_email("WordPress Sync Failed", f"Error: {e}")
        put_metric("SyncErrors", 1)
        execution_time = time.time() - start_time
        put_metric("ExecutionTime", execution_time)
        return {
            "status": "failure",
            "error": str(e),
            "execution_time": execution_time
        }

# ------------------- Run locally -------------------
if __name__ == "__main__":
    output = lambda_handler({}, {})
    print(json.dumps(output))  # Ensure workflow can parse JSON...
