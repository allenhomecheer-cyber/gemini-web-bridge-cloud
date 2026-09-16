FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EIV DATA_DIR=/app/data

RUNiapt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUNipip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CME['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8080']
