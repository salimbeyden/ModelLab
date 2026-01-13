FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for R and build tools
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install R packages (mgcv is included in r-base, add readr for CSV handling)
RUN Rscript -e "install.packages(c('readr', 'jsonlite'), repos='http://cran.rstudio.com/')"

# Copy application code
COPY . .

# Create directories for data and runs
RUN mkdir -p /app/data /app/runs /app/tmp_r

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
