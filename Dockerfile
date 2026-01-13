# Multi-stage build for ModelLab
# This builds the backend with FastAPI

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for R and build tools
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install R packages
RUN Rscript -e "install.packages(c('readr', 'jsonlite', 'mgcv'), repos='http://cran.rstudio.com/')"

# Copy backend application code
COPY backend/ .

# Create directories for data and runs
RUN mkdir -p /app/data /app/runs /app/tmp_r

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
