"""
RAG 问答服务：把检索 + LLM 组合起来。

核心流程：
1. 用户提问 → 向量库检索 top K 资料片段
2. 把片段拼成 context → 塞进 prompt
3. LLM "看着资料"回答 → 返回答案 + 引用来源

设计要点：检索只做一次。
早期版本里流式路由先调 get_sources_for_question() 拿来源，
再调 answer_question_stream() 生成回答，同一句问题被检索了两遍
（等于把 Embedding API 调了两次，既慢又费钱）。
现在统一走 _retrieve()，一次检索同时喂给 prompt 和来源列表。
"""

from collections.abc import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_settings
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.core.messages import extract_text
from app.core.vectorstore import search_documents
from app.models.schemas import QAResponse

logger = get_logger(__name__)


# ---------- 提示词模板 ----------
SYSTEM_PROMPT = """你是中北大学智能问答助手。请严格根据下面提供的资料回答用户问题。

回答要求：
1. 只根据资料中的事实回答，不要编造、不要用资料外的知识
2. 资料里没有相关信息时，直接回复："抱歉，我手头的资料里没有这个信息。"
3. 答案要简洁清晰，必要时可以分点说明
4. 结合对话历史理解用户问题中的指代（如"它""那"等）

资料：
{context}"""

# 多轮对话的 prompt：带历史槽位
PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        # 历史消息占位符：会被 [HumanMessage, AIMessage, ...] 填充
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 首轮对话的 prompt：没有历史
PROMPT_FIRST_TURN = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_context(docs: list[Document]) -> str:
    """把检索到的文档列表格式化成 prompt 中的 context 字段。"""
    blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "未知")
        content = d.page_content.strip()
        blocks.append(f"[资料{i}] (来源: {source})\n{content}")
    return "\n\n".join(blocks)


def _unique_sources(docs: list[Document]) -> list[str]:
    """从检索结果里提取去重后的来源文件名，保持检索顺序。"""
    seen: list[str] = []
    for d in docs:
        source = d.metadata.get("source", "未知")
        if source not in seen:
            seen.append(source)
    return seen


def _retrieve(question: str) -> tuple[list[Document], str]:
    """检索 + 格式化上下文（整个服务只在这里碰向量库一次）。"""
    settings = get_settings()
    docs = search_documents(question, k=settings.retrieval_top_k)
    logger.debug("检索到 %d 个片段", len(docs))
    return docs, _format_context(docs)


def _build_prompt_messages(
    question: str,
    history: list[BaseMessage] | None = None,
) -> tuple[list[Document], list]:
    """
    检索 + 构造 prompt（同步和流式共用）。

    Returns:
        (检索到的文档列表, LangChain 消息列表)
    """
    docs, context = _retrieve(question)

    if history:
        messages = PROMPT_WITH_HISTORY.format_messages(
            context=context,
            history=history,
            question=question,
        )
    else:
        messages = PROMPT_FIRST_TURN.format_messages(
            context=context,
            question=question,
        )

    return docs, messages


def answer_question(question: str, history: list[BaseMessage] | None = None) -> QAResponse:
    """
    完整的 RAG 问答流程（支持多轮历史）。

    Args:
        question: 用户问题
        history: 历史消息列表（LangChain Message 对象），None 表示首轮

    Returns:
        QAResponse 对象，包含答案和引用来源
    """
    settings = get_settings()
    logger.info("RAG 问答开始: %s", question[:50])

    retrieved_docs, messages = _build_prompt_messages(question, history=history)

    llm = get_llm()
    response = llm.invoke(messages)
    answer = extract_text(response)
    logger.info("LLM 回答完成，长度 %d 字", len(answer))

    return QAResponse(
        question=question,
        answer=answer,
        sources=_unique_sources(retrieved_docs),
        model=settings.llm_model,
    )


async def answer_question_stream(
    question: str,
    history: list[BaseMessage] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    RAG 问答的流式版本：先产出 sources 事件，再逐字产出 token 事件。

    产出的事件（dict）由路由层包成 SSE：
        {"type": "sources", "data": [...]}
        {"type": "token",   "data": "文本片段"}

    这样检索只发生一次，来源和回答共用同一批文档。
    """
    logger.info("RAG 流式问答开始: %s", question[:50])
    retrieved_docs, messages = _build_prompt_messages(question, history=history)

    # 先给前端来源，让用户马上看到"在查哪些资料"
    yield {"type": "sources", "data": _unique_sources(retrieved_docs)}

    llm = get_llm(streaming=True)

    # astream 返回异步迭代器，每次产出 AIMessageChunk（增量内容）
    total = 0
    async for chunk in llm.astream(messages):
        text = chunk.content
        if text:
            total += len(text)
            yield {"type": "token", "data": text}

    logger.info("LLM 流式回答完成，长度 %d 字", total)


def get_sources_for_question(question: str) -> list[str]:
    """只拿引用来源（不生成回答），用于前端单独展示来源。"""
    docs, _ = _retrieve(question)
    return _unique_sources(docs)


if __name__ == "__main__":
    # 单独运行此模块可以测试 RAG 流程
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "中北大学宿舍几人间？"
    print(f"问题: {q}\n")
    result = answer_question(q)
    print(f"回答: {result.answer}\n")
    print(f"引用来源: {result.sources}")
