# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install native compilation tooling dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository structure blocks to container workspace
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

EXPOSE 8000

# Fire up uvicorn production instance workers to host API endpoints
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
