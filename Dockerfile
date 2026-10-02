FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

WORKDIR /app


# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# Make `python` point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python


# ------------------------------------------------------------
# Python virtual environment
# ------------------------------------------------------------

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install -r requirements.txt


# ------------------------------------------------------------
# Application code
# ------------------------------------------------------------

COPY api ./api
COPY models ./models


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8001"]