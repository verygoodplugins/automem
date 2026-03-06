# Dockerfile - Flask API runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full application source
COPY . .

# Qdrant connection — set either QDRANT_URL (full URL, takes precedence) or
# QDRANT_HOST (internal hostname, e.g. qdrant.railway.internal) + QDRANT_PORT.
# Using QDRANT_HOST avoids the need to know the port in Railway/internal-network setups.
ENV QDRANT_URL="" \
    QDRANT_API_KEY="" \
    QDRANT_HOST="" \
    QDRANT_PORT="6333"

EXPOSE 8001

CMD ["python", "app.py"]
