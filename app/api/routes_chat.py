"""纯 LLM 对话路由（无 RAG，对照组/闲聊用），支持多轮对话。"""

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.session_store import ensure_session_id, get_session_store
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import answer_plain, build_history

logger = get_logger(__name__)
router = APIRouter(tags=["对话"])


@router.get("/chat", response_model=ChatResponse, summary="纯对话（GET，便于浏览器测试）")
async def chat_get(
    question: str = Query(..., min_length=1, max_length=500, description="用户问题"),
) -> ChatResponse:
    """直接把问题转发给 LLM，AI 凭训练知识回答（不查学校资料）。"""
    answer = answer_plain(question)
    settings = get_settings()
    return ChatResponse(
        question=question,
        answer=answer,
        model=settings.llm_model,
        provider=settings.chat_provider,
    )


@router.post("/chat", response_model=ChatResponse, summary="纯对话（POST，支持多轮）")
async def chat_post(body: ChatRequest) -> ChatResponse:
    """
    POST 版本：支持多轮对话。

    - 传 session_id：继续上一轮对话
    - 不传 session_id：开启新对话，返回新的 session_id
    """
    settings = get_settings()
    store = get_session_store()
    session_id = ensure_session_id(body.session_id)

    # 取历史 → 转 LangChain 消息
    history = build_history(store.get_history(session_id))

    # 调用 LLM
    answer = answer_plain(body.question, history=history)

    # 把这轮对话存进历史
    store.add_message(session_id, "user", body.question)
    store.add_message(session_id, "assistant", answer)

    return ChatResponse(
        question=body.question,
        answer=answer,
        model=settings.llm_model,
        provider=settings.chat_provider,
        session_id=session_id,
    )
