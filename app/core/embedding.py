"""
Embedding 模型封装：把文字转成向量。

为什么单独封装？
- 统一接口：调用方只调 get_embeddings()
- 单例缓存：Embedding 客户端有初始化开销（本地模型还要加载权重）
- 切换方便：换厂商只改 .env，不动业务代码

支持三种提供商（由 EMBEDDING_PROVIDER 决定）：
- local  ：本地开源模型（BAAI/bge-small-zh-v1.5），免费、离线、无需 API Key
- zhipu  ：智谱 embedding-3 / embedding-2，按量计费
- openai ：OpenAI text-embedding-3-small，按量计费

为什么默认用 local？
对话模型可以白嫖（智谱 glm-4-flash 是免费的），但 Embedding 是按量计费的，
学生党不一定愿意为它充值。本地模型让整个 RAG 链路零成本跑通，
且离线可用；等有预算了改一行 .env 就能切回云端。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    """根据配置创建 Embedding 客户端单例。"""
    settings = get_settings()
    provider = settings.embedding_provider
    model = settings.resolved_embedding_model

    # ---------- 本地免费模型 ----------
    if provider == "local":
        # 延迟导入：torch 很重（几百 MB），不用本地模型时不该拖慢启动
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise ValueError(
                "使用本地 Embedding 需要额外依赖，请先安装：\n"
                "  pip install -r requirements.txt\n"
                "（需要 sentence-transformers 和 langchain-huggingface）"
            ) from e

        # 中文场景用归一化后的余弦相似度，效果比默认点积更稳
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ---------- 智谱 Embedding ----------
    if provider == "zhipu":
        if not settings.zhipu_api_key:
            raise ValueError("智谱 API Key 未配置！请在 .env 中设置 ZHIPU_API_KEY")

        from langchain_community.embeddings import ZhipuAIEmbeddings

        kwargs: dict = {
            "model": model,
            "api_key": settings.zhipu_api_key,
        }
        # embedding-3 支持自定义维度；embedding-2 固定 1024 维，不能传 dimensions
        if model == "embedding-3":
            kwargs["dimensions"] = 2048
        return ZhipuAIEmbeddings(**kwargs)

    # ---------- OpenAI Embedding ----------
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API Key 未配置！请在 .env 中设置 OPENAI_API_KEY")

        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        return OpenAIEmbeddings(
            model=model,
            # 字段类型是 SecretStr，传裸字符串 mypy 会报类型不兼容
            api_key=SecretStr(settings.openai_api_key),
            base_url=settings.openai_api_base,
        )

    # embedding_provider 是 Literal 类型，理论上到不了这里；留作兜底防御
    raise ValueError(f"不支持的 Embedding 提供商: {provider}，可选值: local | zhipu | openai")
