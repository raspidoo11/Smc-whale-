# SMC Whale — container image for any always-on host (AWS Fargate/Lightsail/
# EC2, Railway, Fly.io, a VPS). This bot is a persistent worker (20s-cadence
# scheduler, SQLite state, in-memory caches, model files on disk) — it is NOT
# a fit for AWS Lambda (15-min execution cap, ephemeral disk, cold starts on
# a ~300MB ML stack, no sub-minute scheduling). Use this image on a container
# service instead.
#
#   docker build -t smc-whale .
#   docker run --env-file .env -v smc_data:/app/data smc-whale
#
# Mount a persistent volume at /app/data (state + models live there).

FROM python:3.13-slim

# libgomp1: OpenMP runtime required by LightGBM/XGBoost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data

CMD ["python", "main.py"]
