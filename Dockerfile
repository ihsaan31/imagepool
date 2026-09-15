FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IMAGEPOOL_DATA_DIR=/app/data \
    IMAGEPOOL_MODEL_CACHE=/opt/imagepool-models \
    TORCH_HOME=/opt/imagepool-models/torch \
    HF_HOME=/opt/imagepool-models/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gosu libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY scripts/cache_model.py /tmp/cache_model.py
ARG IMAGEPOOL_MODEL_NAME=ViT-B-32
ARG IMAGEPOOL_MODEL_PRETRAINED=laion2b_s34b_b79k
RUN mkdir -p /opt/imagepool-models \
    && IMAGEPOOL_MODEL_NAME=${IMAGEPOOL_MODEL_NAME} \
       IMAGEPOOL_MODEL_PRETRAINED=${IMAGEPOOL_MODEL_PRETRAINED} \
       python /tmp/cache_model.py \
    && rm /tmp/cache_model.py

RUN groupadd --system imagepool \
    && useradd --system --gid imagepool --create-home imagepool \
    && mkdir -p /app/data \
    && chown -R imagepool:imagepool /app /opt/imagepool-models

COPY --chown=imagepool:imagepool . .
RUN chmod +x /app/scripts/docker-entrypoint.sh

EXPOSE 8501
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
