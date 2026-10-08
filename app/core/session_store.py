"""
会话历史存储（内存版）。

为什么需要多轮对话历史？
- 用户问"宿舍几人间"→ AI 答"6 人间"
- 用户再问"那有空调吗"→ AI 得知道"那"指宿舍
- 所以要把历史对话存下来，下次提问时一起塞给 LLM

为什么用内存不用数据库？
- 历史对话是临时数据，服务重启丢了没关系
- 内存读写极快，不需要外部依赖
- 后期要持久化再换 Redis / DB

注意事项：
- 内存 dict 不是进程安全的，生产环境多 worker 要换 Redis
- 有两层清理：①每个会话最多保留 N 轮，防止上下文窗口爆掉
              ②总会话数有上限 + 超时淘汰，防止内存无限增长
"""

import threading
import time
import uuid
from collections import OrderedDict

from app.core.logging import get_logger

logger = get_logger(__name__)


# 每个会话最多保留的"对话轮数"（1 轮 = 用户问 1 次 + AI 答 1 次）
_MAX_HISTORY_TURNS = 10
# 单个会话存活时间（秒），超时自动清理
_SESSION_TTL = 3600  # 1 小时
# 最多同时保留多少个会话（防止内存被无限增长的 session_id 撑爆）
_MAX_SESSIONS = 1000


class SessionStore:
    """会话历史的内存存储（线程安全）。"""

    def __init__(
        self,
        max_turns: int = _MAX_HISTORY_TURNS,
        ttl: int = _SESSION_TTL,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        # OrderedDict 方便实现"超量时淘汰最久未用"（类似 LRU）
        # 结构：session_id -> {"messages": list[dict], "last_active": float}
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max_turns = max_turns
        self._ttl = ttl
        self._max_sessions = max_sessions

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史（list of {"role": "user"|"assistant", "content": ...}）。"""
        with self._lock:
            self._prune_expired_locked()
            entry = self._store.get(session_id)
            if not entry:
                return []
            entry["last_active"] = time.time()
            # 标记为最近使用，让 LRU 淘汰时不会先删它
            self._store.move_to_end(session_id)
            return list(entry["messages"])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息到会话历史。"""
        with self._lock:
            self._prune_expired_locked()
            entry = self._store.get(session_id)
            if entry is None:
                entry = {"messages": [], "last_active": time.time()}
                self._store[session_id] = entry

            entry["messages"].append({"role": role, "content": content})
            # 按"轮"裁剪：一轮 = 2 条消息（user + assistant）。
            # 注意不能用 deque(maxlen=N)：N 条消息只是 N/2 轮，
            # 而且会把一轮拆开（只剩 assistant 没有对应的 user），语义就乱了。
            max_messages = self._max_turns * 2
            if len(entry["messages"]) > max_messages:
                entry["messages"] = entry["messages"][-max_messages:]

            entry["last_active"] = time.time()
            self._store.move_to_end(session_id)

            # 会话总数超上限：淘汰最久未使用的
            while len(self._store) > self._max_sessions:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug("会话数超上限，淘汰最久未用会话: %s", evicted_id)

    def clear(self, session_id: str) -> None:
        """清空某个会话的历史。"""
        with self._lock:
            self._store.pop(session_id, None)

    def _prune_expired_locked(self) -> int:
        """清理超时会话（调用方必须已持有锁）。"""
        now = time.time()
        expired = [
            sid for sid, entry in self._store.items() if now - entry["last_active"] > self._ttl
        ]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.info("清理了 %d 个过期会话", len(expired))
        return len(expired)

    def cleanup_expired(self) -> int:
        """清理超时会话，返回清理数量。

        对外暴露的公开方法（方便定时任务/测试调用）；
        读写历史时内部也会顺带清理一次。
        """
        with self._lock:
            return self._prune_expired_locked()

    def stats(self) -> dict:
        """返回当前会话数量等统计信息（便于监控和测试）。"""
        with self._lock:
            return {
                "active_sessions": len(self._store),
                "max_sessions": self._max_sessions,
                "max_turns": self._max_turns,
                "ttl_seconds": self._ttl,
            }


# 全局单例
_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _session_store


def ensure_session_id(session_id: str | None) -> str:
    """前端没传 session_id 时生成一个新的。

    放在这里而不是各个路由里：三个路由（chat / qa / qa/stream）
    都需要"没传就新建"这个逻辑，之前每个文件都抄了一份，
    改动时容易漏改某一个。
    """
    return session_id or uuid.uuid4().hex
