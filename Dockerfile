# Dockerfile - Complete setup
FROM falkordb/falkordb:latest

# Install Python and dependencies for API
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
WORKDIR /app
COPY requirements.txt .

# Create virtual environment and install dependencies (Python 3.11+ best practice)
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose ports
EXPOSE 6379 3000 8001

# Start both Redis/FalkorDB and Flask API
CMD redis-server --loadmodule /FalkorDB/bin/linux-x64-release/src/falkordb.so --daemonize yes && \
    sleep 2 && \
    /app/venv/bin/python app.py