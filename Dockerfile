# ====================================================
# Streamlit Dashboard Container (Production Ready)
# ====================================================

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file first (optimizes Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional but recommended) Install AWS CLI - enables S3 access
RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# Copy application source code
COPY . .

# Make Streamlit visible outside container
EXPOSE 8501

# Health check for container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false
