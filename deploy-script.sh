#!/bin/bash
# ==========================================================
# Multi-environment deploy script
# Usage: bash deploy.sh [staging|production]
# ==========================================================
set -e

ENV=$1
if [[ -z "$ENV" ]]; then
  echo "Usage: bash deploy.sh [staging|production]"
  exit 1
fi

STACK_NAME="wordpress-dashboard-$ENV"
TEMPLATE_FILE="cloudformation/full-stack-multi-env.yaml"
AWS_REGION="us-east-1"

# Environment-specific parameters
if [ "$ENV" == "staging" ]; then
  KEY_NAME="staging-key"
  DB_HOST="staging-db-host"
  DB_USER="staging-user"
  DB_PASSWORD="staging-pass"
  DB_NAME="staging-db"
  GITHUB_BRANCH="staging"
else
  KEY_NAME="prod-key"
  DB_HOST="prod-db-host"
  DB_USER="prod-user"
  DB_PASSWORD="prod-pass"
  DB_NAME="prod-db"
  GITHUB_BRANCH="main"
fi

# Upload Lambda
echo "Uploading Lambda code..."
aws s3 cp lambda/lambda.zip s3://$STACK_NAME-data/lambda/lambda.zip

# Deploy CloudFormation
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
  --stack-name $STACK_NAME \
  --template-file $TEMPLATE_FILE \
  --parameter-overrides \
      Environment=$ENV \
      KeyName=$KEY_NAME \
      UbuntuAmiId=ami-0abcdef1234567890 \
      VPCId=vpc-0123456789abcdef0 \
      SubnetIds=subnet-1234abcd,subnet-5678efgh \
      GitHubRepo=https://github.com/yourusername/wordpress-dashboard-full.git \
      GitHubBranch=$GITHUB_BRANCH \
      NotificationEmail=your-email@example.com \
      SESNotificationEmail=your-email@example.com \
      LambdaScheduleExpression="cron(0 0 * * ? *)" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region $AWS_REGION

# Output Streamlit URL
ALB_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query "Stacks[0].Outputs[?OutputKey=='StreamlitURL'].OutputValue" \
  --output text --region $AWS_REGION)

echo "✅ $ENV Streamlit dashboard: http://$ALB_ENDPOINT:8501"
