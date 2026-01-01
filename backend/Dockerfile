FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install prisma

# Copy backend as a package
COPY . /app/backend

# Make Python see backend
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "python -m prisma generate --schema backend/prisma/schema.prisma && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
