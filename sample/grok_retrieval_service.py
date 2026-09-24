import logging
import re
import threading
import os

from typing import List, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.llms.groq import Groq
from src.prompt.system_prompt import SYSTEM_PROMPT
from src.prompt.user_prompt import render_user_prompt
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger("retrieval_service")

# Fallback for when a reasoning model's <think>...</think> block leaks through
# despite reasoning_format="hidden" (e.g. unsupported model/older API version).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_block(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    # If stripping leaves nothing (i.e. generation was truncated mid-<think>
    # and never reached the answer), surface that clearly instead of silently
    # returning an empty string.
    if not cleaned and "<think>" in text:
        logger.warning(
            "Response was truncated inside a <think> block with no answer "
            "text after it — likely max_tokens is too low for this model."
        )
        return ""
    return cleaned


class GroqRetrievalService:
    def __init__(self, retriever, cfg, reranker=None):
        self.retriever = retriever
        # Optional NodeReranker (see src/retrieve/reranking.py). When provided,
        # answer_query() reranks retrieved nodes before building the LLM prompt.
        self.reranker = reranker
        max_conc = getattr(cfg, "max_concurrent_llm", 2)
        self._llm_semaphore = threading.BoundedSemaphore(max_conc)

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.error("GROQ_API_KEY not found in .env file!")
            self.llm = None
            return

        try:
            self.llm = Groq(
                model=cfg.groq_model,
                api_key=groq_key,
                temperature=getattr(cfg, "temperature", 0.7),
                # Reasoning models (e.g. deepseek-r1-distill, qwen3) spend a large
                # chunk of the token budget on the <think>...</think> block before
                # emitting the final answer. Too low a limit truncates generation
                # mid-thought, leaving an unclosed <think> tag and NO answer text.
                max_tokens=getattr(cfg, "max_tokens", 4096),
                additional_kwargs={
                    # Groq-specific: ask reasoning-capable models to keep the
                    # <think> block out of the returned content entirely, so we
                    # don't burn tokens on it or have to strip it client-side.
                    # Safely ignored by non-reasoning models.
                    "reasoning_format": getattr(cfg, "reasoning_format", "hidden"),
                },
            )
            logger.info(f"Groq LLM initialized with model: {cfg.groq_model}")
        except Exception as e:
            logger.error("Failed to init Groq LLM: %s", e)
            self.llm = None

    def answer_query(
        self,
        query: str,
        user_id: str,
        session_id: str,
        timeout_seconds: int,
        history: List = None,
    ) -> Dict[str, Any]:
        # Same logic as your original Ollama service
        try:
            nodes: List[NodeWithScore] = self.retriever.retrieve(query)
        except Exception as e:
            logger.error("Retrieval failed: %s", e, exc_info=True)
            return {
                "answer": "Error occurred while getting relevant chunks.",
                "sources": [],
                "user_id": user_id,
                "session_id": session_id,
            }

        if not nodes:
            logger.warning("No results returned from retriever.")
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "user_id": user_id,
                "session_id": session_id,
            }

        if self.reranker is not None:
            try:
                nodes = self.reranker.rerank(nodes, query)
            except Exception as e:
                # Belt-and-suspenders: NodeReranker.rerank() already falls back
                # internally, but never let a reranker bug break the answer path.
                logger.error(
                    "Reranking step failed, using original nodes: %s", e, exc_info=True
                )

        if not nodes:
            logger.warning("No results left after reranking.")
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "user_id": user_id,
                "session_id": session_id,
            }

        sources = [n.node.get_content()[:200] for n in nodes]
        context_block = "\n\n".join([n.node.get_content() for n in nodes if n.node])
        for src in sources:
            print("=========")
            print(src)

        user_prompt = render_user_prompt(context_block, query, history)
        print(f"USER PROMPT: {user_prompt}")
        full_prompt = SYSTEM_PROMPT.strip() + "\n\n" + user_prompt.strip()

        answer_text = ""
        try:
            acquired = self._llm_semaphore.acquire(timeout=timeout_seconds)
            if not acquired:
                raise TimeoutError("Timeout waiting for LLM concurrency slot")

            def _do_llm_call(prompt: str) -> str:
                if self.llm is None:
                    raise RuntimeError("LLM is not initialized")
                response = self.llm.complete(prompt)
                raw_text = getattr(response, "text", None) or str(response)
                return _strip_think_block(raw_text)

            try:
                with ThreadPoolExecutor(max_workers=1) as _exec:
                    fut = _exec.submit(_do_llm_call, full_prompt)
                    answer_text = fut.result(timeout=timeout_seconds)
            finally:
                self._llm_semaphore.release()

            if not answer_text:
                # Truncated mid-<think> with nothing after it, or a genuinely
                # empty completion. Fall back to raw chunks rather than
                # returning nothing.
                logger.warning("Empty answer after LLM call; falling back to raw chunks.")
                answer_text = (
                    "I couldn't generate a complete answer (response was cut off). "
                    "Here are the most relevant chunks:\n\n" + "\n".join(sources)
                    if sources
                    else "Sorry, I couldn't find relevant information."
                )
        except TimeoutError as e:
            logger.error("LLM timeout: %s", e, exc_info=True)
            raise
        except Exception as e:
            logger.error("LLM call failed: %s", e, exc_info=True)
            if sources:
                answer_text = (
                    "I couldn't generate a reliable answer, here are the most relevant chunks:\n\n"
                    + "\n".join(sources)
                )
            else:
                answer_text = "Sorry, I couldn't find relevant information."
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
    from src.retrieve.config.retrieval_config import RetrievalConfig
    from src.retrieve.config.datastore_config import DataStoreConfig
    from src.retrieve.index_loader import build_index
    from src.retrieve.reranking import build_reranker_from_config

    session_id = str(uuid.uuid4())
    dscfg = DataStoreConfig()
    cfg = RetrievalConfig()
    index = build_index(
        collection_name=dscfg.collection,
        dim=dscfg.dim,
        host=dscfg.host,
        port=dscfg.port,
        embed_model_name=cfg.embed_model,
    )
    # Fetch more candidates than we need so the reranker has something to
    # narrow down (e.g. top_k=20 vector hits -> rerank down to top_n=5).
    retriever = index.as_retriever(similarity_top_k=cfg.top_k)
    reranker = build_reranker_from_config(cfg)
    svc = GroqRetrievalService(retriever, cfg, reranker=reranker)
    resp = svc.answer_query(
        query="What is seed fund scheme?",
        user_id="neeraj",
        session_id=session_id,
        timeout_seconds=10000,
    )
    print(f"Answer: {resp['answer']}")
