# Dockerfile for Graph Viewer - paths relative to packages/graph-viewer
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files (these are in the current directory, not packages/graph-viewer/)
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Production - serve static files
FROM node:20-slim

WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/server.mjs ./
COPY --from=builder /app/package*.json ./

ENV HOST=0.0.0.0
EXPOSE 3000

CMD ["node", "server.mjs"]
