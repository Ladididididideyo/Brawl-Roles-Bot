FROM node:18-slim

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node dependencies
RUN npm ci --omit=dev

# Copy application code
COPY . .

# Start the bot
CMD ["node", "index.js"]