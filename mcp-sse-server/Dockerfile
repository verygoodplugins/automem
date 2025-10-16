FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY server.js ./

# Railway injects PORT automatically
EXPOSE 8080

CMD ["node", "server.js"]

