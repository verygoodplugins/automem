# Dockerfile - Flask API runtime image with optional Graph Viewer
# Multi-stage build: Node.js for frontend, Python for backend

# Stage 1: Build the Graph Viewer frontend
FROM node:20-slim AS frontend-builder

WORKDIR /build

# Copy package files and install dependencies
COPY packages/graph-viewer/package*.json ./
RUN npm ci --silent

# Copy source and build
COPY packages/graph-viewer/ ./
RUN npm run build

# Stage 2: Python runtime
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

# Copy the built frontend from stage 1
COPY --from=frontend-builder /build/dist/ ./automem/static/viewer/

EXPOSE 8001

CMD ["python", "app.py"]
