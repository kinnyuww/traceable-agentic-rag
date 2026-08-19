# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DATA_DIR=/data \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8080

WORKDIR /app

RUN addgroup --system --gid 10001 ragagent \
    && adduser --system --uid 10001 --ingroup ragagent --home /app ragagent

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

RUN mkdir -p /data && chown -R ragagent:ragagent /app /data
USER ragagent

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=2)"]

CMD ["uvicorn", "ragagent.main:app", "--host", "0.0.0.0", "--port", "8080"]

