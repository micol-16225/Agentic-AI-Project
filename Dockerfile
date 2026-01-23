FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (Agents, Data Scripts, Apps)
COPY . .

# Default to Streamlit (since that's your deployment target)
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app4.py", "--server.port=8501", "--server.address=0.0.0.0"]