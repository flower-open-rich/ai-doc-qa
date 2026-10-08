"""
处理 LLM 返回内容的小工具。

为什么需要它？
LangChain 的 `BaseMessage.content` 类型是 `str | list[str | dict]`：
- 普通文本模型返回 `str`
- 带多模态 / 工具调用的模型会返回内容块列表，比如
  `[{"type": "text", "text": "你好"}]`

如果我们直接把 `content` 塞进 Pydantic 的 `answer: str` 字段，
类型对不上（mypy 会报错），运行时还可能返回一个前端读不懂的结构。
所以统一在出口处转成纯字符串。
"""

from langchain_core.messages import BaseMessage


def extract_text(message: BaseMessage) -> str:
    """把 LLM 返回的消息内容统一转成纯文本字符串。"""
    return normalize_content(message.content)


def normalize_content(content: str | list) -> str:
    """
    把消息内容规范化为字符串。

    - str  -> 原样返回
    - list -> 逐个元素提取文本后拼接（跳过无法识别的块）
    - 其他 -> 转成 str 兜底，保证不会返回 None 或异常结构
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # OpenAI / LangChain 风格的文本块
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content)
