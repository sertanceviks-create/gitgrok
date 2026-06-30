# GitGrok — production image
FROM python:3.11-slim

# git: repoları klonlamak için gerekli
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# bağımlılıklar (önce sadece requirements → katman cache'i)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# uygulama dosyaları
COPY server.py repomind.py verify.py gitgrok.html ./

# Render/Fly PORT ortam değişkenini verir; yoksa 8000
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
