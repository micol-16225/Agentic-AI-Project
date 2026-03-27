# Use 3.11-slim as requested, but added build-essentials for scikit-learn
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system dependencies needed for compiling some C-based math libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
# Ensure we use the CPU version of torch we discussed for the 1GB RAM limit
RUN pip install --no-cache-dir -r requirements.txt

# 3. --- CRITICAL: PRE-DOWNLOAD MODEL ---
# This ensures the 80MB model is inside the image before it hits AWS
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 4. Copy application code
COPY . .

# 5. Streamlit Configuration
EXPOSE 8501

# Healthcheck to let AWS Target Groups know the app is alive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "streamlit_app4.py", "--server.port=8501", "--server.address=0.0.0.0"]