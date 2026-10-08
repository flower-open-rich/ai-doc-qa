"""
LLM 客户端封装。

为什么封装？
- 统一接口：无论用智谱、OpenAI、通义，调用方都只调 get_llm().invoke()
- 切换方便：改 .env 里的 LLM_PROVIDER 就能换厂商，不用改业务代码
- 单例缓存：LLM 客户端初始化有开销（建连、加载配置），别每次调用都重建
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings


def get_llm(streaming: bool = False) -> BaseChatModel:
    """
    根据配置创建 LLM 客户端。

    Args:
        streaming: 是否启用流式输出。流式用 .astream() 逐字返回，非流式用 .invoke() 等全部生成。

    注意：@lru_cache 不能直接装饰带参数的工厂函数，
    所以用闭包+缓存字典实现"按 streaming 标志分别缓存"。
    """
    return _get_llm_cached(streaming)


@lru_cache(maxsize=4)
def _get_llm_cached(streaming: bool) -> BaseChatModel:
    """内部缓存：按 streaming=True/False 各缓存一个实例。"""
    settings = get_settings()
    provider = settings.chat_provider

    if provider == "zhipu":
        # 智谱 GLM
        if not settings.zhipu_api_key:
            raise ValueError(
                "智谱 API Key 未配置！请在 .env 文件中设置 ZHIPU_API_KEY\n"
                "  → 没注册的话去 https://open.bigmodel.cn/ 注册\n"
                "  → 控制台拿 API Key 后填到 .env 文件里"
            )
        from langchain_community.chat_models import ChatZhipuAI

        # 注意：ChatZhipuAI 把 zhipu_api_base 当成"完整的请求 URL"用，
        # 不会自动拼 /chat/completions，所以要手动补上后缀。
        full_chat_url = settings.zhipu_api_base.rstrip("/") + "/chat/completions"
        return ChatZhipuAI(
            model=settings.llm_model,
            api_key=settings.zhipu_api_key,
            api_base=full_chat_url,
            temperature=0.7,
            streaming=streaming,
        )

    if provider == "openai":
        # OpenAI（或兼容 OpenAI 协议的接口，如 DeepSeek / 通义 / 本地 vLLM）
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API Key 未配置！请在 .env 中设置 OPENAI_API_KEY\n"
                "  → 如果用兼容 OpenAI 协议的第三方服务，"
                "同时把 OPENAI_API_BASE 改成对应的地址"
            )
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        return ChatOpenAI(
            model=settings.llm_model,
            # langchain-openai 的 api_key 字段类型是 SecretStr，
            # 传裸字符串 mypy 会报类型不兼容，这里显式包一层。
            api_key=SecretStr(settings.openai_api_key),
            base_url=settings.openai_api_base,
            temperature=0.7,
            streaming=streaming,
        )

    # chat_provider 是 Literal 类型，理论上到不了这里；留作兜底防御
    raise ValueError(f"不支持的对话提供商: {provider}，可选值: zhipu | openai")
