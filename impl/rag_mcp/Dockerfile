# Build from this directory (repo root once split out):
#   docker build -t rag-mcp .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rag_mcp/ /app/rag_mcp/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PORT=8080 \
    MCP_HOST=0.0.0.0

CMD ["python", "-m", "rag_mcp.server"]
