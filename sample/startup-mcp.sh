#!/bin/bash
set -e

echo "=========================================================="
echo " Launching Multi-Service RAG Stack (REST API, MCP, UI)"
echo "=========================================================="

# 1. Start FastAPI REST Backend in background (Port 8000)
echo "--> Starting FastAPI REST API Service on port 8000..."
uvicorn startup_funding_retrieval_api:app --host 0.0.0.0 --port 8000 &

# 2. Start MCP SSE Server in background (Port 8001)
echo "--> Starting Model Context Protocol (MCP) Server on port 8001..."
if [ -f mcp_server.py ]; then
    python3 mcp_server.py &
else
    echo "mcp_server.py not found. Skipping MCP service startup."
fi

# 3. Brief pause to ensure backend endpoints bind cleanly
sleep 3

# 4. Start Frontend UI (Streamlit on Port 7860 or Static HTML fallback)
if [ -f streamlit_frontend.py ]; then
    echo "--> Starting Streamlit Frontend UI on port 7860..."
    exec streamlit run streamlit_frontend.py \
        --server.port=7860 \
        --server.address=0.0.0.0 \
        --server.headless=true
else
    echo "--> Streamlit frontend file not detected. Holding process for background services..."
    wait -n
fi
