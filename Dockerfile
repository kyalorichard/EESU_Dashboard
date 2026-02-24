# ====================================================
# EESU Streamlit Dashboard - Production Ready Dockerfile
# ====================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# -----------------------------
# Install system dependencies
# Needed for geospatial libraries (geopandas, shapely, GDAL) and curl
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy dependencies and install Python packages
# -----------------------------
COPY requirements.txt .

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY . .

# -----------------------------.
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Healthcheck for container
# -----------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -----------------------------
# Run Streamlit app (JSON-array CMD, single-line)
# -----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
