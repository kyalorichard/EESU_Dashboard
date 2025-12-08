#!/bin/bash
# ==========================================================
# Single deploy script for full WordPress -> S3 -> Lambda -> Streamlit stack
# Includes CloudFormation deploy, Lambda upload, and outputs Streamlit URL
# ==========================================================
set -e

STACK_NAME="wordpress-dashboard-stack"
TEMPLATE_FILE="cloudformation/full-stack-handoff-monitored.yaml"
AWS_REGION="us-east-1"

# --------------- Deployment Parameters -------------------
KEY_NAME="your-keypair"                               # EC2 keypair for SSH
EC2_INSTANCE_TYPE="t3.micro"
UBUNTU_AMI_ID="ami-0abcdef1234567890"               # Ubuntu 22.04 AMI
VPC_ID="vpc-0123456789abcdef0"
SUBNET_IDS="subnet-1234abcd,subnet-5678efgh"
ACM_CERT_ARN="arn:aws:acm:region:account:certificate/xxxxxxxx"
GITHUB_REPO="https://github.com/yourusername/wordpress-dashboard-full.git"
GITHUB_BRANCH="main"
NOTIFICATION_EMAIL="your-email@example.com"
SES_EMAIL="$NOTIFICATION_EMAIL"
LAMBDA_CRON="cron(0 0 * * ? *)"                     # Daily at midnight UTC

# ---------------- Upload Lambda --------------------------
echo "Uploading Lambda code to S3..."
aws s3 cp lambda/lambda.zip s3://$STACK_NAME-data/lambda/lambda.zip

# ---------------- Deploy CloudFormation ------------------
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
    --stack-name $STACK_NAME \
    --template-file $TEMPLATE_FILE \
    --parameter-overrides \
        KeyName=$KEY_NAME \
        EC2InstanceType=$EC2_INSTANCE_TYPE \
        UbuntuAmiId=$UBUNTU_AMI_ID \
        VPCId=$VPC_ID \
        SubnetIds=$SUBNET_IDS \
        ACMCertificateArn=$ACM_CERT_ARN \
        GitHubRepo=$GITHUB_REPO \
        GitHubBranch=$GITHUB_BRANCH \
        NotificationEmail=$NOTIFICATION_EMAIL \
        SESNotificationEmail=$SES_EMAIL \
        LambdaScheduleExpression="$LAMBDA_CRON" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

echo "Waiting for stack to complete..."
aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region $AWS_REGION

# ---------------- Output Streamlit URL -------------------
ALB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='StreamlitURL'].OutputValue" \
    --output text --region $AWS_REGION)

echo "✅ Streamlit dashboard URL: http://$ALB_ENDPOINT:8501"
echo "✅ Lambda SES notifications sent to: $SES_EMAIL"
