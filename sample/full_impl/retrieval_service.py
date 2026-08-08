import logging
import threading 

from typing import List, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from prompt.system_prompt import SYSTEM_PROMPT
from prompt.user_prompt import render_user_prompt
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger("retrieval_service")


class RetrievalService:
    def __init__(self, retriever, cfg):
        self.retriever = retriever
        max_conc = getattr(cfg, "max_concurrent_llm", 2)
        # prevents more than max_conc overlapping generations
        self._llm_semaphore = threading.BoundedSemaphore(max_conc)
        # Ollama LLM via LlamaIndex wrapper
        try:
            self.llm = Ollama(
                                model=cfg.ollama_model,
                                request_timeout=cfg.ollama_timeout,
                                context_window=cfg.ollama_context_window,
                                keep_alive=cfg.keep_alive,
                                temperature=cfg.temperature,
                                stream=cfg.stream
                            )
        except Exception as e:
            self.logger.error("Failed to init Ollama client: %s", e)
            self.llm = None

    def answer_query(
                        self,
                        query: str,
                        user_id: str,
                        session_id: str,
                        timeout_seconds: int,
                        history: List = None
                    ) -> Dict[str, Any]:
        try:
            # Step 1: Retrieve most relevant nodes
            nodes: List[NodeWithScore] = self.retriever.retrieve(query)
        except Exception as e:
            logger.error("LLM call failed: %s", e, exc_info=True)
            return {
                        "answer": "Error occurred while getting relevant chunks.",
                        "sources": [],
                        "user_id": user_id,
                        "session_id": session_id
                    }

        if not nodes:
            logger.warning("No results returned from retriever.")
            return {"answer": "No relevant context found.", "sources": [], "user_id": user_id, "session_id": session_id}

        sources = [n.node.get_content()[:200] for n in nodes]
        # Build context from top nodes
        context_block = "\n\n".join([n.node.get_content() for n in nodes if n.node])

        # Step 2: Render final prompt
        user_prompt = render_user_prompt(context_block, query, history)
        full_prompt = SYSTEM_PROMPT.strip() + "\n\n" + user_prompt.strip()

        # Step 3: Get answer from LLM
        answer_text = ""
        try:
            acquired = self._llm_semaphore.acquire(timeout=timeout_seconds)
            if not acquired:
                raise TimeoutError("Timeout waiting for LLM concurrency slot")

            def _do_llm_call(prompt: str) -> str:
                if self.llm is None:
                    raise RuntimeError("LLM is not initialized")
                response = self.llm.complete(prompt)
                return getattr(response, "text", None) or str(response)
            try:
                # Use a short-lived thread to enforce wall clock timeout on generation
                with ThreadPoolExecutor(max_workers=1) as _exec:
                    fut = _exec.submit(_do_llm_call, full_prompt)
                    answer_text = fut.result(timeout=timeout_seconds)
            finally:
                self._llm_semaphore.release()
        except TimeoutError as e:
            logger.error("LLM timeout: %s", e, exc_info=True)
            raise  # let FastAPI translate to 504
        except Exception as e:
            logger.error("LLM call failed: %s", e, exc_info=True)
            if sources:
                fallback = "I couldn't generate a reliable answer, here are the most relevant chunks:\n\n"
                answer_text = fallback
            else:
                answer_text = "Sorry, I couldn't find relevant information."
            raise
        print(f"Answer_text: {answer_text}")
        return {
            "answer": answer_text,
            "sources": sources,
            "user_id": user_id,
            "session_id": session_id,
        }

# To test without API and streamlit
if __name__ == "__main__":
    import uuid
    from retrieve.config.retrieval_config import RetrievalConfig
    from retrieve.config.datastore_config import DataStoreConfig
    from retrieve.index_loader import build_index

    session_id = str(uuid.uuid4())
    dscfg = DataStoreConfig()
    cfg = RetrievalConfig()
    index = build_index(
                        collection_name=dscfg.collection,
                        dim=dscfg.dim,
                        host=dscfg.host,
                        port=dscfg.port,
                        embed_model_name=cfg.embed_model
                    )
    retriever = index.as_retriever(similarity_top_k=cfg.top_k)
    svc = RetrievalService(retriever, cfg)
    resp = svc.answer_query(
            query="What is seed fund scheme?",
            user_id='neeraj',
            session_id=session_id,
            timeout_seconds=10000,
        )
    print(f"Answer: {resp['answer']}")
