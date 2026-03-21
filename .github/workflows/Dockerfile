FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# PORT env var → main.py reads it automatically
ENV PORT=7860

CMD ["python", "main.py"]
