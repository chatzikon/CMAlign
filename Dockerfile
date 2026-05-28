FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

WORKDIR /app

# 1️⃣ Install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Make python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

## 3️⃣ Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install system deps (optional but useful for PIL / torch)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy project code
COPY api ./api
COPY models ./models

COPY models/Showo /app/Show-o
COPY models/Showo/configs /app/configs

COPY api/prompt.txt /app/api/prompt.txt

EXPOSE 4523

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "4523"]