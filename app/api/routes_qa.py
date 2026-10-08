"""RAG 智能问答路由（核心功能），支持多轮对话。"""

import uuid

from fastapi import APIRouter, Query

from app.core.logging import get_logger
from app.core.session_store import get_session_store
from app.models.schemas import QARequest, QAResponse
from app.services.chat_service import build_history
from app.services.qa_service import answer_question

logger = get_logger(__name__)
router = APIRouter(tags=["RAG 问答"])


def _ensure_session_id(session_id: str | None) -> str:
    return session_id or uuid.uuid4().hex


@router.get("/qa", response_model=QAResponse, summary="RAG 问答（GET，便于测试）")
async def qa_get(
    question: str = Query(..., min_length=1, max_length=500, description="用户问题"),
) -> QAResponse:
    """
    先检索中北大学资料库 → 把资料塞进 prompt → LLM 看着资料回答。
    适合问"宿舍几人间""科研经费多少"等学校相关信息。
    """
    return answer_question(question)


@router.post("/qa", response_model=QAResponse, summary="RAG 问答（POST，支持多轮）")
async def qa_post(body: QARequest) -> QAResponse:
    """
    POST 版本：支持多轮对话。

    - 传 session_id：继续上一轮对话（历史会喂给 LLM 理解指代）
    - 不传 session_id：开启新对话，返回新的 session_id
    """
    store = get_session_store()
    session_id = _ensure_session_id(body.session_id)

    # 取历史 → 转 LangChain 消息
    history = build_history(store.get_history(session_id))

    # RAG 问答（带历史）
    result = answer_question(body.question, history=history)

    # 存历史
    store.add_message(session_id, "user", body.question)
    store.add_message(session_id, "assistant", result.answer)

    result.session_id = session_id
    return result
