FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# 1️⃣ Install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Make python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3️⃣ Upgrade pip
RUN pip3 install --upgrade pip

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