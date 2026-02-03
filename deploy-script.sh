#!/bin/bash
# ==========================================================
# Automated deployment with rollback for EESU_Dashboard
# Supports staging and production with HTTPS auto SSL
# Usage: bash deploy.sh [staging|production]
# ==========================================================
set -e

ENV=$1
if [[ -z "$ENV" ]]; then
  echo "Usage: bash deploy.sh [staging|production]"
  exit 1
fi

# ---------------- PARAMETERS ----------------
APP_DIR="/home/youruser/EESU_Dashboard"
BACKUP_DIR="/home/youruser/EESU_Dashboard_backup"
VENV_DIR="$APP_DIR/venv"
REPO_URL="https://github.com/kyalorichard/EESU_Dashboard.git"
SYSTEMD_SERVICE="eesu_dashboard"
DOMAIN_STAGING="staging.yourdomain.com"
DOMAIN_PROD="yourdomain.com"
PYTHON_BIN="python3"
DEPLOY_TIMEOUT=15   # seconds to check Streamlit startup
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "[$TIMESTAMP] Deploying $ENV environment"

# ---------------- SELECT ENV ----------------
if [ "$ENV" == "staging" ]; then
  GIT_BRANCH="staging"
  DOMAIN=$DOMAIN_STAGING
else
  GIT_BRANCH="main"
  DOMAIN=$DOMAIN_PROD
fi

# ---------------- BACKUP CURRENT VERSION ----------------
if [ -d "$APP_DIR" ]; then
  echo "[$TIMESTAMP] Backing up current version..."
  rm -rf "$BACKUP_DIR"
  cp -r "$APP_DIR" "$BACKUP_DIR"
fi

# ---------------- CLONE OR PULL REPO ----------------
if [ ! -d "$APP_DIR" ]; then
  echo "[$TIMESTAMP] Cloning repository..."
  git clone -b $GIT_BRANCH $REPO_URL $APP_DIR
else
  echo "[$TIMESTAMP] Pulling latest code..."
  cd $APP_DIR
  git fetch origin
  git reset --hard origin/$GIT_BRANCH
fi

# ---------------- SETUP VIRTUAL ENV ----------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[$TIMESTAMP] Creating virtual environment..."
  $PYTHON_BIN -m venv $VENV_DIR
fi

echo "[$TIMESTAMP] Installing dependencies..."
$VENV_DIR/bin/pip install --upgrade pip
$VENV_DIR/bin/pip install -r $APP_DIR/requirements.txt

# ---------------- SYSTEMD SERVICE ----------------
SERVICE_FILE="/etc/systemd/system/$SYSTEMD_SERVICE.service"
if [ ! -f "$SERVICE_FILE" ]; then
  echo "[$TIMESTAMP] Creating systemd service..."
  sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Streamlit service for EESU_Dashboard
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/streamlit run $APP_DIR/app.py --server.port 8501 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable $SYSTEMD_SERVICE
fi

# ---------------- DEPLOY WITH ROLLBACK CHECK ----------------
echo "[$TIMESTAMP] Restarting Streamlit service..."
sudo systemctl restart $SYSTEMD_SERVICE
sleep $DEPLOY_TIMEOUT

# Test if Streamlit is running
if curl -s "http://127.0.0.1:8501" | grep -q "Streamlit"; then
  echo "[$TIMESTAMP] ✅ Deployment successful."
else
  echo "[$TIMESTAMP] ❌ Deployment failed! Rolling back..."
  # Restore backup
  rm -rf "$APP_DIR"
  mv "$BACKUP_DIR" "$APP_DIR"
  sudo systemctl restart $SYSTEMD_SERVICE
  echo "[$TIMESTAMP] ⚠️ Rolled back to previous version."
  exit 1
fi

# ---------------- NGINX CONFIG ----------------
NGINX_CONF="/etc/nginx/sites-available/$ENV.eesu_dashboard"
if [ ! -f "$NGINX_CONF" ]; then
  echo "[$TIMESTAMP] Creating Nginx config..."
  sudo tee $NGINX_CONF > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
  sudo ln -sf $NGINX_CONF /etc/nginx/sites-enabled/
  sudo nginx -t
  sudo systemctl restart nginx
fi

# ---------------- CERTBOT SSL ----------------
if ! command -v certbot &> /dev/null; then
  echo "[$TIMESTAMP] Installing Certbot..."
  sudo apt update
  sudo apt install -y certbot python3-certbot-nginx
fi

echo "[$TIMESTAMP] Obtaining/renewing SSL certificate for $DOMAIN..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m your-email@domain.com --redirect || true

# ---------------- FINAL OUTPUT ----------------
echo "[$TIMESTAMP] ✅ Deployment complete!"
echo "Dashboard is available at: https://$DOMAIN"
