FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000

# Fewer workers = less memory; bump timeout to avoid slow cold start issues
CMD gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:${PORT} app:app
