# Dockerfile for Graph Viewer (Node.js static frontend)
FROM node:20-slim AS builder

WORKDIR /build

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --silent

# Copy source and build
COPY . .
RUN npm run build

# Production stage - lightweight static server
FROM node:20-slim

WORKDIR /app

# Copy only what's needed for production
COPY package*.json ./
COPY server.mjs ./
COPY --from=builder /build/dist ./dist

# Railway uses PORT env var
ENV HOST=0.0.0.0
EXPOSE 3000

CMD ["node", "server.mjs"]
