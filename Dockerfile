# ====================================================
# Streamlit Dashboard Container (Production Ready)
# ====================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for geospatial packages and curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for Docker caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit using JSON-array CMD (fixes JSONArgsRecommended warning)
CMD ["streamlit", "run", "app.py", 
     "--server.port=8501", 
     "--server.address=0.0.0.0", 
     "--server.enableCORS=false", 
     "--server.enableXsrfProtection=false", 
     "--browser.gatherUsageStats=false"]
