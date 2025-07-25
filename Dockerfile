FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY deploy.prototxt .
COPY res10_300x300_ssd_iter_140000_fp16.caffemodel .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]