"""
Lambda function to:
- Connect to WordPress MySQL database
- Fetch new/updated posts since last sync
- Merge with cumulative CSV in S3
- Update checkpoint in S3
- Send SES email notifications on success/failure
- Logs to CloudWatch
"""

import os
import pymysql
import boto3
import csv
import json
from datetime import datetime
import logging
from io import StringIO
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables for DB, S3, and SES
S3_BUCKET = os.environ['S3_BUCKET_NAME']
S3_PREFIX = os.environ.get('S3_PREFIX', 'wp_incremental_csv')
DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_NAME = os.environ['DB_NAME']
SES_EMAIL = os.environ.get('SES_EMAIL')

s3 = boto3.client('s3')
ses = boto3.client('ses')

CHECKPOINT_KEY = f"{S3_PREFIX}/checkpoint.json"
CUMULATIVE_CSV_KEY = f"{S3_PREFIX}/wordpress_cumulative.csv"

def get_last_sync_time():
    """Fetch last sync timestamp from S3 checkpoint"""
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=CHECKPOINT_KEY)
        checkpoint = json.loads(resp['Body'].read())
        return checkpoint.get('last_sync')
    except s3.exceptions.NoSuchKey:
        logger.info("No checkpoint found. Fetching all posts.")
        return None

def update_checkpoint(timestamp):
    """Update checkpoint in S3 with latest sync time"""
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=json.dumps({"last_sync": timestamp})
    )
    logger.info(f"Updated checkpoint to {timestamp}")

def send_email(subject, body):
    """Send SES email notification"""
    if not SES_EMAIL:
        logger.warning("SES_EMAIL not set. Skipping email notification.")
        return
    try:
        ses.send_email(
            Source=SES_EMAIL,
            Destination={'ToAddresses': [SES_EMAIL]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        )
    except Exception as e:
        logger.error(f"Failed to send SES email: {e}")

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        last_sync = get_last_sync_time()
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME)
        cursor = conn.cursor()

        # Fetch only new posts since last sync
        if last_sync:
            logger.info(f"Fetching posts updated since {last_sync}")
            cursor.execute(
                "SELECT ID, post_title, post_date FROM wp_posts "
                "WHERE post_status='publish' AND post_type='post' AND post_date > %s",
                (last_sync,)
            )
        else:
            logger.info("Fetching all posts")
            cursor.execute(
                "SELECT ID, post_title, post_date FROM wp_posts "
                "WHERE post_status='publish' AND post_type='post'"
            )

        rows = cursor.fetchall()
        if not rows:
            logger.info("No new posts to sync.")
            send_email("WordPress Sync Completed", "No new posts since last sync.")
            return {"status": "no_new_posts"}

        # Read existing cumulative CSV from S3 if exists
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=CUMULATIVE_CSV_KEY)
            df_existing = pd.read_csv(resp['Body'])
            logger.info("Loaded existing cumulative CSV from S3.")
        except s3.exceptions.NoSuchKey:
            df_existing = pd.DataFrame(columns=['ID', 'Title', 'Date'])
            logger.info("No cumulative CSV found. Creating new.")

        # Merge new rows with existing CSV
        df_new = pd.DataFrame(rows, columns=['ID', 'Title', 'Date'])
        df_cumulative = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['ID'])

        # Save cumulative CSV in memory and upload to S3
        csv_buffer = StringIO()
        df_cumulative.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=CUMULATIVE_CSV_KEY, Body=csv_buffer.getvalue())
        logger.info(f"Cumulative CSV updated in S3: {CUMULATIVE_CSV_KEY}")

        # Update checkpoint
        now_iso = datetime.utcnow().isoformat()
        update_checkpoint(now_iso)

        send_email(
            "WordPress Sync Successful",
            f"Successfully synced {len(df_new)} new posts.\nTotal posts: {len(df_cumulative)}"
        )

        return {"status": "success", "new_rows": len(df_new), "total_rows": len(df_cumulative)}

    except Exception as e:
        logger.error(f"Error in WordPress sync: {e}")
        send_email("WordPress Sync Failed", f"Error: {e}")
        raise
