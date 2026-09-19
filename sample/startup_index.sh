#!/bin/bash
set -e

# 1. Start FastAPI Orchestrator Backend in background on port 8000
echo "Starting FastAPI Backend Service on port 8000..."
uvicorn startup_funding_retrieval_api:app --host 0.0.0.0 --port 8000 &

# 2. Brief pause to ensure FastAPI server boots up
sleep 3

