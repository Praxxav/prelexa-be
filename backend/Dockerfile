FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Prisma CLI
RUN pip install prisma

# Copy source code
COPY . .

# Generate Prisma client
RUN prisma generate

# Railway provides PORT automatically
EXPOSE 8000

# Start FastAPI (NO reload)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
