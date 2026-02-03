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

# Copy application source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check for container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false
