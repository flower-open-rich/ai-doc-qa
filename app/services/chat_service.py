"""
纯 LLM 对话服务（不带 RAG）。

和 qa_service 的区别：
- chat_service：直接问 LLM，AI 凭训练知识答（适合闲聊/通用问题）
- qa_service：先检索学校资料再答（适合问中北大学相关信息）

为什么单独建 service？
- 后面加多轮对话历史时，只改这个文件
- 路由层保持薄，只做参数接收/返回，不写业务
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.llm import get_llm
from app.core.logging import get_logger
from app.core.messages import extract_text

logger = get_logger(__name__)


# 系统提示词：定义 AI 的身份和回答风格
SYSTEM_PROMPT = (
    "你是中北大学智能问答助手，一个友好、专业的校园 AI。"
    "回答简洁清晰，涉及中北大学的具体信息时请提醒用户：'建议在问答模式获取更准确的校内信息'。"
)


# 带历史槽位的 prompt 模板
# MessagesPlaceholder(variable_name="history") 会被历史消息列表填充
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def answer_plain(
    question: str,
    history: list[BaseMessage] | None = None,
) -> str:
    """
    纯 LLM 回答。

    Args:
        question: 当前问题
        history: 历史消息列表（HumanMessage / AIMessage），None 表示首轮

    Returns:
        AI 回答文本
    """
    llm = get_llm()
    history = history or []

    logger.info("Chat 对话: %s (历史 %d 条)", question[:50], len(history))

    messages = PROMPT.format_messages(history=history, question=question)
    response = llm.invoke(messages)
    # response.content 的类型是 str | list，统一转成字符串再返回
    return extract_text(response)


def build_history(turns: list[dict]) -> list[BaseMessage]:
    """
    把字典格式的对话历史转成 LangChain 消息对象。

    Args:
        turns: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        [HumanMessage, AIMessage, ...]
    """
    messages: list[BaseMessage] = []
    for turn in turns:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages
