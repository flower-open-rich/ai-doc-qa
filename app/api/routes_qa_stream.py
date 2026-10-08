"""RAG 问答 - 流式响应（SSE），支持多轮对话。"""

import json
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.core.session_store import get_session_store
from app.models.schemas import QARequest
from app.services.chat_service import build_history
from app.services.qa_service import answer_question_stream

logger = get_logger(__name__)
router = APIRouter(tags=["RAG 问答"])


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/qa/stream", summary="RAG 问答（SSE 流式，支持多轮）")
async def qa_stream(body: QARequest) -> StreamingResponse:
    """
    流式返回回答，支持多轮对话 session_id。

    事件类型：
    - session_id: 会话 ID（首轮返回）
    - sources:    引用来源
    - token:      增量文本
    - done:       结束
    - error:      错误
    """
    store = get_session_store()
    session_id = body.session_id or uuid.uuid4().hex
    history = build_history(store.get_history(session_id))

    async def event_generator():
        full_answer: list[str] = []
        try:
            # 先推 session_id，前端后续请求都带上它
            yield _sse_event({"type": "session_id", "data": session_id})

            # answer_question_stream 内部只检索一次，
            # 依次产出 sources 事件和 token 事件
            async for event in answer_question_stream(body.question, history=history):
                if event["type"] == "token":
                    full_answer.append(event["data"])
                yield _sse_event(event)

            # 存历史
            store.add_message(session_id, "user", body.question)
            store.add_message(session_id, "assistant", "".join(full_answer))

            yield _sse_event({"type": "done", "data": None})

        except Exception as e:
            logger.exception("流式问答失败: %s", e)
            yield _sse_event({"type": "error", "data": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
