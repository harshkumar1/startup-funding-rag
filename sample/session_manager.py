import uuid
import redis
from typing import List, Tuple
from src.retrieve.config.retrieval_config import RetrievalConfig


class SessionManager:
    def __init__(
        self, cfg: RetrievalConfig
    ):  # redis_host="localhost", redis_port=6379, redis_db=0, ttl_seconds=86400*30):
        self._r = redis.StrictRedis(
            host=cfg.redis_host,
            port=cfg.redis_port,
            db=cfg.redis_db,
            decode_responses=True,
        )
        self.ttl = cfg.session_ttl_seconds

    def _user_key(self, user_id: str) -> str:
        return f"user:{user_id}:sessions"

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def create_user(self) -> str:
        return str(uuid.uuid4())

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self._r.sadd(self._user_key(user_id), session_id)
        self._r.expire(self._user_key(user_id), self.ttl)
        return session_id

    def get_user_sessions(self, user_id: str) -> List[str]:
        return list(self._r.smembers(self._user_key(user_id)))

    def append_turn(self, session_id: str, role: str, text: str):
        key = self._session_key(session_id)
        self._r.rpush(key, f"{role}:::{text}")
        self._r.expire(key, self.ttl)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        key = self._session_key(session_id)
        items = self._r.lrange(key, 0, -1)
        history = []
        for item in items:
            try:
                role, text = item.split(":::", 1)
                history.append((role, text))
            except:
                history.append(("unknown", item))
        return history
