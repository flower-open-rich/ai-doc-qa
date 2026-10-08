"""测试会话存储：轮数裁剪、TTL 过期、LRU 淘汰、线程安全。"""

import threading
import time

from app.core.session_store import SessionStore, ensure_session_id


def test_ensure_session_id_generates_when_missing():
    """没传 session_id 时应生成一个新的（32 位 hex）。"""
    sid = ensure_session_id(None)
    assert isinstance(sid, str)
    assert len(sid) == 32
    assert sid.isalnum()


def test_ensure_session_id_keeps_existing():
    """传了 session_id 时必须原样保留，否则多轮对话会断掉。"""
    assert ensure_session_id("my-session") == "my-session"


def test_ensure_session_id_is_unique():
    """连续生成的 ID 不能重复。"""
    ids = {ensure_session_id(None) for _ in range(200)}
    assert len(ids) == 200


def test_new_session_history_is_empty():
    store = SessionStore()
    assert store.get_history("nobody") == []


def test_add_and_get_history():
    store = SessionStore()
    store.add_message("s1", "user", "宿舍几人间？")
    store.add_message("s1", "assistant", "6 人间。")
    history = store.get_history("s1")
    assert history == [
        {"role": "user", "content": "宿舍几人间？"},
        {"role": "assistant", "content": "6 人间。"},
    ]


def test_history_trimmed_by_turns_not_messages():
    """按"轮"裁剪：保留的消息数应为 轮数 × 2。

    回归测试：早期版本用 deque(maxlen=N) 直接把 N 当成消息条数，
    结果 20 条只等于 10 轮，而且会把一轮从中间截断
    （只剩 assistant 回复、丢掉对应的 user 提问），上下文语义就乱了。
    """
    store = SessionStore(max_turns=3)
    for i in range(10):
        store.add_message("s1", "user", f"问题{i}")
        store.add_message("s1", "assistant", f"回答{i}")

    history = store.get_history("s1")
    assert len(history) == 6, "3 轮应该保留 6 条消息"

    # 必须保留最近的 3 轮
    assert history[0] == {"role": "user", "content": "问题7"}
    assert history[-1] == {"role": "assistant", "content": "回答9"}

    # 关键：不能出现"孤儿" assistant（一轮被从中间截断）
    assert history[0]["role"] == "user", "历史必须以 user 提问开头"


def test_clear_removes_session():
    store = SessionStore()
    store.add_message("s1", "user", "hi")
    store.clear("s1")
    assert store.get_history("s1") == []


def test_clear_missing_session_is_noop():
    store = SessionStore()
    store.clear("never-existed")  # 不应抛异常


def test_cleanup_expired_removes_old_sessions():
    """TTL 过期后 cleanup_expired 要真的清掉会话。

    回归测试：早期版本定义了 cleanup_expired()，
    但代码里没有任何地方调用它，_SESSION_TTL 等于一段死代码。
    """
    store = SessionStore(ttl=0)  # 立刻过期
    store.add_message("s1", "user", "hi")
    time.sleep(0.01)

    removed = store.cleanup_expired()
    assert removed == 1
    assert store.get_history("s1") == []


def test_expired_sessions_pruned_on_read():
    """就算没人显式调用 cleanup，读取时也应该顺带清理过期会话。"""
    store = SessionStore(ttl=0)
    store.add_message("old", "user", "hi")
    time.sleep(0.01)

    # 触发另一次写入 -> 内部自动清理
    store.add_message("new", "user", "hello")
    assert store.stats()["active_sessions"] == 1


def test_lru_eviction_when_too_many_sessions():
    """会话数超过上限时，应淘汰最久未使用的那个，防止内存无限增长。"""
    store = SessionStore(max_sessions=2)
    store.add_message("s1", "user", "a")
    store.add_message("s2", "user", "b")
    # 访问 s1，让它变成"最近使用"
    store.get_history("s1")
    # 加入 s3，触发淘汰：最久未用的 s2 应该被踢掉
    store.add_message("s3", "user", "c")

    assert store.stats()["active_sessions"] == 2
    assert store.get_history("s2") == []
    assert store.get_history("s1") != []
    assert store.get_history("s3") != []


def test_history_returned_is_a_copy():
    """拿到的历史应该是副本，外部修改不能影响 store 内部状态。"""
    store = SessionStore()
    store.add_message("s1", "user", "hi")
    history = store.get_history("s1")
    history.append({"role": "user", "content": "注入的假数据"})
    assert len(store.get_history("s1")) == 1


def test_concurrent_writes_are_thread_safe():
    """多线程并发写入不应丢消息或抛异常。"""
    store = SessionStore(max_turns=1000)
    threads_count = 8
    per_thread = 50

    def worker(idx: int) -> None:
        for i in range(per_thread):
            store.add_message("shared", "user", f"t{idx}-{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.get_history("shared")) == threads_count * per_thread
