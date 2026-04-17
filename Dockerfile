FROM python:3.12-slim

WORKDIR /app

# Install system deps (optional but useful for PIL / torch)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY api ./api
COPY models ./models

EXPOSE 4523

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "4523"]