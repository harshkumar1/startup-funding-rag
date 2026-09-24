import time

from concurrent.futures import TimeoutError as FuturesTimeoutError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.retrieve.grok_retrieval_service import GroqRetrievalService
from src.user_management.session_manager import SessionManager
from src.retrieve.config.retrieval_config import RetrievalConfig
from src.retrieve.config.datastore_config import DataStoreConfig
from src.retrieve.index_loader import build_index

"""
command to run service:
uvicorn services.startup_funding_retrieval_api:app --host 0.0.0.0 --port 8000 --reload
"""

app = FastAPI(
    title="RAG Retrieval Service",
    description="API for querying documents with user/session context",
    version="1.0.0",
)

cfg = RetrievalConfig()
dscfg = DataStoreConfig()
index = build_index(
    collection_name=dscfg.collection,
    dim=dscfg.dim,
    host=dscfg.host,
    port=dscfg.port,
    embed_model_name=cfg.embed_model,
)
retriever = index.as_retriever(similarity_top_k=cfg.top_k)

# Initialize service and session manager
svc = GroqRetrievalService(retriever, cfg)
sm = SessionManager(cfg=cfg)


# Request schema
class QueryRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query: str
    new_chat: bool = False
    timeout_seconds: int = 3600  # allow client to request a larger/smaller guard


# Response schema
class QueryResponse(BaseModel):
    answer: str
    session_id: str
    user_id: str
    history: Optional[list]
    latency: int


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    try:
        if not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be blank")
        t0 = time.time()
        # Handle user
        user_id = request.user_id  # or sm.create_user()

        # Handle session
        session_id = request.session_id
        if request.new_chat or not session_id:
            session_id = sm.create_session(user_id)

        # Get answer from retrieval service
        result = svc.answer_query(
            query=request.query,
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=request.timeout_seconds,
        )
        result["latency_ms"] = int((time.time() - t0) * 1000)
        # Save turn
        sm.append_turn(session_id, "user", request.query)
        sm.append_turn(session_id, "assistant", result["answer"])

        return QueryResponse(
            answer=result["answer"],
            session_id=session_id,
            user_id=user_id,
            history=sm.get_history(session_id),
            latency=result["latency_ms"],
        )
    except FuturesTimeoutError:
        # Timeout while waiting on LLM slot or guarded generation
        raise HTTPException(status_code=504, detail="Upstream model timed out")
    except Exception as e:
        # generic failure
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str) -> Dict[str, Any]:
    try:
        sessions = sm.get_user_sessions(user_id)
        return {"user_id": user_id, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str) -> Dict[str, Any]:
    try:
        history = sm.get_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
