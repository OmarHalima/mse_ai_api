FROM python:3.10-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y \
    curl ca-certificates \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 7860
ENV PORT=7860

CMD ["python", "main.py"]
