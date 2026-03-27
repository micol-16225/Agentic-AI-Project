FROM python:3.11-slim

WORKDIR /app

# 1. Install system tools and clean up immediately
RUN apt-get update && apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 2. FORCE the 150MB Torch CPU version first
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install sentence-transformers separately (to ensure it links to the CPU torch above)
RUN pip install --no-cache-dir sentence-transformers==3.0.1

# 4. Install the rest of your UI/API tools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY statutory_truth_with_ids.csv .
COPY optimizer_intelligence_with_ids.csv .
COPY generate_embeddings.py .

RUN python generate_embeddings.py

# 5. Copy your application code
COPY . .

EXPOSE 8501

# The model will download on the first run to save build-time disk space
CMD ["streamlit", "run", "streamlit_app4.py", "--server.port=8501", "--server.address=0.0.0.0"]