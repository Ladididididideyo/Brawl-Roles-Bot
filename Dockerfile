FROM node:18-slim

# Install Python, OpenCV dependencies, and required system libraries
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  libgl1 \
  libglib2.0-0 \
  tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY requirements.txt ./

# Install Node dependencies
RUN npm install --omit=dev

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Start the bot
CMD ["node", "index.js"]