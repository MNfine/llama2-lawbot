FROM python:3.10-slim

WORKDIR /app

# Cài poppler + tesseract (OCR)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       poppler-utils \
       tesseract-ocr \
       tesseract-ocr-vie \
       build-essential \
       git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements và cài Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .


EXPOSE 5002

CMD ["python", "-u", "app.py"]
