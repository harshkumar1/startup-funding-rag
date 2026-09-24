"""
app.py

Streamlit front end for the RAG API (GroqRetrievalService).

Flow:
  1. User registers with just their name (used as user_id).
  2. Every new chat gets a fresh session_id (uuid4).
  3. Each session's turn history is cached in Redis as JSON.
  4. On every new query, the cached history for that session_id is loaded
     from Redis and sent alongside the query so the API can use it as
     conversation context; the new turn is then appended back to Redis.

Run:
    pip install streamlit redis requests
    streamlit run app.py
"""

import json
import uuid
from datetime import datetime, timezone

import requests
import redis
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these to match your actual deployment
# ─────────────────────────────────────────────────────────────────────────
RAG_API_BASE_URL = "http://localhost:8000"
RAG_API_QUERY_PATH = "/query"  # <-- change if your real route differs
RAG_API_TIMEOUT_SECONDS = 30  # HTTP client-side timeout (requests)
RAG_ANSWER_TIMEOUT_SECONDS = 25  # sent as `timeout_seconds` to the API/LLM

REDIS_HOST = "localhost"
REDIS_PORT = 6380  # matches the host port mapped in docker-compose.yml for
                    # the dedicated `rag-redis` container (kept off the
                    # default 6379 to avoid colliding with other apps' Redis)
REDIS_DB = 0  # this instance is dedicated to this app, so DB 0 is fine here
REDIS_KEY_PREFIX = "ragchat:"  # still namespaced as extra insurance

CHAT_HISTORY_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
MAX_HISTORY_TURNS_SENT = 10  # cap how many past turns get sent as context

# ─────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_redis_client() -> redis.Redis:
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )
    client.ping()  # fail fast at startup if Redis isn't reachable
    return client


def _history_key(session_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}chat_history:{session_id}"


def _meta_key(session_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}session_meta:{session_id}"


def _user_sessions_key(user_name: str) -> str:
    return f"{REDIS_KEY_PREFIX}user_sessions:{user_name}"


# ─────────────────────────────────────────────────────────────────────────
# REDIS-BACKED HISTORY
# ─────────────────────────────────────────────────────────────────────────


def load_history(session_id: str) -> list:
    r = get_redis_client()
    raw = r.get(_history_key(session_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def save_history(session_id: str, history: list) -> None:
    r = get_redis_client()
    r.setex(_history_key(session_id), CHAT_HISTORY_TTL_SECONDS, json.dumps(history))


def append_turn(session_id: str, role: str, content: str) -> list:
    """Append a single turn to a session's history in Redis and return the
    updated history."""
    history = load_history(session_id)
    history.append({"role": role, "content": content})
    save_history(session_id, history)
    return history


def register_session(user_name: str, session_id: str, first_query: str = "") -> None:
    """Record this session under the user's session list, with a short
    title (first query) so it's recognizable in the sidebar."""
    r = get_redis_client()
    meta = {
        "user": user_name,
        "title": (first_query[:60] + "…") if len(first_query) > 60 else first_query,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    r.hset(_meta_key(session_id), mapping=meta)
    r.expire(_meta_key(session_id), CHAT_HISTORY_TTL_SECONDS)
    r.lpush(_user_sessions_key(user_name), session_id)
    r.ltrim(_user_sessions_key(user_name), 0, 49)  # keep last 50 sessions
    r.expire(_user_sessions_key(user_name), CHAT_HISTORY_TTL_SECONDS)


def get_user_sessions(user_name: str) -> list:
    r = get_redis_client()
    session_ids = r.lrange(_user_sessions_key(user_name), 0, -1)
    sessions = []
    for sid in session_ids:
        meta = r.hgetall(_meta_key(sid))
        if meta:
            sessions.append({"session_id": sid, **meta})
    return sessions


# ─────────────────────────────────────────────────────────────────────────
# RAG API CALL
# ─────────────────────────────────────────────────────────────────────────


def call_rag_api(query: str, user_id: str, session_id: str, history: list) -> dict:
    """
    POSTs to the RAG API. Payload mirrors GroqRetrievalService.answer_query()'s
    signature exactly. Trims history to the last MAX_HISTORY_TURNS_SENT turns
    so the prompt doesn't grow unbounded over a long session.
    """
    url = f"{RAG_API_BASE_URL}{RAG_API_QUERY_PATH}"
    payload = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "timeout_seconds": RAG_ANSWER_TIMEOUT_SECONDS,
        "history": history[-MAX_HISTORY_TURNS_SENT:],
    }
    response = requests.post(url, json=payload, timeout=RAG_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# ---- session_state defaults ----
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # local render copy, mirrors Redis history

# ---- registration gate ----
if not st.session_state.user_name:
    st.title("💬 RAG Chat")
    st.subheader("Register to start chatting")
    with st.form("register_form"):
        name_input = st.text_input("Your name")
        submitted = st.form_submit_button("Start chatting")
        if submitted:
            name_clean = name_input.strip()
            if not name_clean:
                st.error("Please enter a name.")
            else:
                try:
                    get_redis_client()  # verify Redis is reachable up front
                except redis.exceptions.RedisError as e:
                    st.error(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT} — {e}")
                    st.stop()
                st.session_state.user_name = name_clean
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
    st.stop()

# ---- sidebar: user info, session controls, past sessions ----
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.user_name}")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.user_name = None
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Past sessions**")
    try:
        past_sessions = get_user_sessions(st.session_state.user_name)
    except redis.exceptions.RedisError as e:
        past_sessions = []
        st.caption(f"Couldn't load history: {e}")

    for sess in past_sessions:
        label = sess.get("title") or "(empty chat)"
        is_current = sess["session_id"] == st.session_state.session_id
        if st.button(
            f"{'👉 ' if is_current else ''}{label}",
            key=f"sess_{sess['session_id']}",
            use_container_width=True,
            disabled=is_current,
        ):
            st.session_state.session_id = sess["session_id"]
            st.session_state.messages = load_history(sess["session_id"])
            st.rerun()

# ---- main chat area ----
st.title("💬 RAG Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**[{i}]** {src}")

user_query = st.chat_input("Ask something…")

if user_query:
    session_id = st.session_state.session_id
    user_name = st.session_state.user_name

    # Show + persist the user turn immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        history_before = append_turn(session_id, "user", user_query)
        is_first_turn = len(history_before) == 1
        if is_first_turn:
            register_session(user_name, session_id, first_query=user_query)
    except redis.exceptions.RedisError as e:
        st.error(f"Redis error while saving history: {e}")
        history_before = st.session_state.messages  # degrade gracefully

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Send history *excluding* the just-added user turn, since
                # the current query is passed separately to the API.
                context_history = history_before[:-1]
                result = call_rag_api(
                    query=user_query,
                    user_id=user_name,
                    session_id=session_id,
                    history=context_history,
                )
                answer = result.get("answer", "(no answer returned)")
                sources = result.get("sources", [])
            except requests.exceptions.Timeout:
                answer = "The request timed out. Please try again."
                sources = []
            except requests.exceptions.RequestException as e:
                answer = f"Couldn't reach the RAG API: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[{i}]** {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    try:
        append_turn(session_id, "assistant", answer)
    except redis.exceptions.RedisError as e:
        st.warning(f"Couldn't cache assistant turn to Redis: {e}")
