"""测试 API 层：路由、参数校验、SSE 事件协议、安全校验、CORS。"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

FAKE_DOCS = [
    Document(
        page_content="本科生宿舍以 6 人间为主，配有空调和暖气。",
        metadata={"source": "campus_life.md"},
    ),
    Document(
        page_content="学校共设 4 个学生食堂。",
        metadata={"source": "campus_life.md"},
    ),
]


class FakeLLM:
    """假的 LLM：不联网，返回固定内容。"""

    def invoke(self, messages):  # noqa: ANN001
        class _Resp:
            content = "【测试回答】宿舍以 6 人间为主。"

        return _Resp()

    async def astream(self, messages):  # noqa: ANN001
        for piece in ["宿舍", "以 6", " 人间为主。"]:

            class _Chunk:
                content = piece

            yield _Chunk()


@pytest.fixture
def client(monkeypatch, fake_embeddings):
    """构造测试客户端。

    关键点：
    - 打桩检索，避免真的去查向量库（也就不会调 Embedding）
    - 打桩 LLM，避免真的发请求
    """
    # 打桩检索层
    monkeypatch.setattr(
        "app.services.qa_service.search_documents",
        lambda query, k=None: list(FAKE_DOCS),
    )
    # 打桩 LLM 工厂
    monkeypatch.setattr("app.core.llm._get_llm_cached", lambda streaming: FakeLLM())

    from app.main import app

    # 用 with 触发 lifespan（启动钩子），保证走的是真实启动路径
    with TestClient(app) as c:
        yield c


def _sse_events(text: str) -> list[dict]:
    """把 SSE 原始响应解析成事件列表。"""
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------- 基础路由


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_root_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "中北大学" in r.text
    assert "<!DOCTYPE html>" in r.text.upper() or "<!doctype html>" in r.text.lower()


def test_openapi_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    # 文档里承诺的接口必须真实存在
    for path in ("/health", "/chat", "/qa", "/qa/stream", "/documents/upload", "/documents/list"):
        assert path in paths, f"OpenAPI 里缺少 {path}"


def test_unknown_route_404(client):
    assert client.get("/api/v1/nonexistent").status_code == 404


# ---------------------------------------------------------------- 参数校验


@pytest.mark.parametrize(
    "payload",
    [
        {},  # 缺 question
        {"question": ""},  # 空字符串
        {"question": "x" * 501},  # 超长
    ],
)
def test_qa_request_validation(client, payload):
    """请求体不合法时应该由 FastAPI 自动返回 422。"""
    assert client.post("/qa", json=payload).status_code == 422


def test_qa_get_requires_question(client):
    assert client.get("/qa").status_code == 422


# ---------------------------------------------------------------- 纯对话


def test_chat_get(client):
    r = client.get("/chat", params={"question": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "你好"
    assert body["answer"]
    assert body["provider"] == "zhipu"


def test_chat_post_returns_session_id(client):
    r = client.post("/chat", json={"question": "你好"})
    assert r.status_code == 200
    assert r.json()["session_id"]


def test_chat_multi_turn_reuses_session(client):
    """同一个 session_id 的第二轮应该能拿到历史。"""
    first = client.post("/chat", json={"question": "第一轮"}).json()
    sid = first["session_id"]

    second = client.post("/chat", json={"question": "第二轮", "session_id": sid}).json()
    assert second["session_id"] == sid

    from app.core.session_store import get_session_store

    history = get_session_store().get_history(sid)
    assert any(h["content"] == "第一轮" for h in history)


# ---------------------------------------------------------------- RAG 问答


def test_qa_post_returns_sources(client):
    r = client.post("/qa", json={"question": "宿舍几人间？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"] == ["campus_life.md"]


def test_qa_sources_are_deduplicated(client):
    """两条检索结果来自同一文件时，sources 里只应出现一次。"""
    body = client.post("/qa", json={"question": "宿舍几人间？"}).json()
    assert len(body["sources"]) == len(set(body["sources"]))


# ---------------------------------------------------------------- SSE 流式


def test_qa_stream_event_protocol(client):
    """SSE 事件顺序必须是 session_id -> sources -> token... -> done。"""
    with client.stream("POST", "/qa/stream", json={"question": "宿舍几人间？"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(r.iter_text()))

    types = [e["type"] for e in events]
    assert types[0] == "session_id"
    assert types[1] == "sources"
    assert types[-1] == "done"
    assert "token" in types
    assert "error" not in types

    # token 拼起来应该等于完整回答
    full = "".join(e["data"] for e in events if e["type"] == "token")
    assert full == "宿舍以 6 人间为主。"


def test_qa_stream_retrieves_only_once(client, monkeypatch):
    """回归测试：流式路径曾经检索两遍（先拿 sources 再生成回答）。

    这里统计 search_documents 的调用次数，必须恰好为 1。
    """
    calls = {"n": 0}

    def counting_search(query, k=None):
        calls["n"] += 1
        return list(FAKE_DOCS)

    monkeypatch.setattr("app.services.qa_service.search_documents", counting_search)

    with client.stream("POST", "/qa/stream", json={"question": "宿舍几人间？"}) as r:
        "".join(r.iter_text())

    assert calls["n"] == 1, f"检索了 {calls['n']} 次，应该只检索 1 次"


def test_qa_stream_saves_history(client):
    with client.stream("POST", "/qa/stream", json={"question": "宿舍几人间？"}) as r:
        events = _sse_events("".join(r.iter_text()))

    sid = next(e["data"] for e in events if e["type"] == "session_id")

    from app.core.session_store import get_session_store

    history = get_session_store().get_history(sid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "宿舍几人间？"}
    assert history[1]["role"] == "assistant"


# ---------------------------------------------------------------- 文档上传


def test_upload_rejects_unsupported_extension(client):
    r = client.post("/documents/upload", files={"file": ("evil.exe", b"MZ")})
    assert r.status_code == 400
    # 前端依赖 detail 字段展示错误，不能只返回 message
    assert "detail" in r.json()


def test_upload_rejects_path_traversal(client, tmp_path):
    """文件名里的目录部分必须被丢弃，不能写到 raw_docs 之外。

    安全回归测试：客户端可控的 filename 若直接拼进保存路径，
    构造 ../../xxx 就能覆盖项目里的任意文件。
    """
    from fastapi import HTTPException

    from app.api.routes_documents import _safe_filename

    assert _safe_filename("../../etc/passwd") == "passwd"
    assert _safe_filename("..\\..\\windows\\system32\\a.txt") == "a.txt"
    assert _safe_filename("/absolute/path/file.md") == "file.md"

    with pytest.raises(HTTPException):
        _safe_filename("..")
    with pytest.raises(HTTPException):
        _safe_filename(None)
    with pytest.raises(HTTPException):
        _safe_filename("")


def test_upload_rejects_empty_file(client):
    r = client.post("/documents/upload", files={"file": ("a.md", b"")})
    assert r.status_code == 400


def test_upload_and_ingest_markdown(client, tmp_path, monkeypatch):
    """上传合法 Markdown 应能完成切片入库。"""
    # 把 raw_docs 重定向到临时目录，避免污染仓库
    from app.api import routes_documents

    monkeypatch.setattr(routes_documents, "_RAW_DOCS_DIR", tmp_path)

    content = "# 测试文档\n\n## 小节\n\n这是一段测试内容，用于验证入库流程。\n".encode()
    r = client.post("/documents/upload", files={"file": ("测试.md", content)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "测试.md"
    assert body["chunks_added"] >= 1
    assert (tmp_path / "测试.md").exists()


# ---------------------------------------------------------------- 文档列表


def test_documents_list_shape(client):
    r = client.get("/documents/list")
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body
    assert "total_chunks" in body
    assert isinstance(body["documents"], list)
    assert isinstance(body["total_chunks"], int)


def test_documents_list_only_shows_supported_types(client, monkeypatch, tmp_path):
    """列表里不应出现不支持的文件类型。"""
    from app.api import routes_documents

    (tmp_path / "good.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "bad.exe").write_bytes(b"MZ")
    monkeypatch.setattr(routes_documents, "_RAW_DOCS_DIR", tmp_path)

    body = client.get("/documents/list").json()
    assert "good.md" in body["documents"]
    assert "bad.exe" not in body["documents"]


# ---------------------------------------------------------------- CORS


def test_cors_does_not_use_wildcard(client):
    """CORS 响应头不能是通配符。

    本服务开了 allow_credentials=True，
    通配符来源 + 携带凭证属于非法组合，浏览器会拒绝该响应，
    所以必须回显具体来源。
    """
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    allow_origin = r.headers.get("access-control-allow-origin")
    assert allow_origin is not None
    assert allow_origin != "*"
    assert allow_origin == "http://localhost:5173"
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_cors_rejects_unknown_origin(client):
    """不在白名单里的来源不应被授予跨域权限。"""
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"
