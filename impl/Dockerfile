# Build from this directory (repo root once split out):
#   docker build -t startup-funding-rag .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Optional extra CA certs for building behind a TLS-intercepting corporate
# proxy (e.g. Zscaler). certs/ is empty by default (see certs/.gitkeep) so
# this is a no-op for normal builds (CI/GHCR/Cloud Build); local devs on a
# proxied network can drop their corporate root CA .crt file(s) into certs/
# (gitignored) before building. Not a secret — CA certs have no private key.
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Python's ssl module, the `requests`/`certifi` stack (used to call the HF
# Inference API for embeddings), and grpc's own bundled BoringSSL roots
# (pymilvus, for Zilliz) each maintain independent trust stores that ignore
# the OS one by default — point all three at it explicitly so any extra
# certs/ CAs above are actually honored at runtime, not just by apt/pip.
# Harmless default value in unproxied environments (points at the same
# standard CA bundle that would otherwise be used).
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rag_api/ /app/rag_api/
COPY rag_core/ /app/rag_core/
COPY server.py /app/server.py
COPY mcp_server.py /app/mcp_server.py
# Default answer-generation tuning config (see generation_config.json +
# AGENTS.md). Bind-mount your own file over this exact path at `docker run`
# time to tune it live, without rebuilding this image.
COPY generation_config.json /app/generation_config.json

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PORT=8080 \
    API_HOST=0.0.0.0

CMD ["python", "server.py"]
