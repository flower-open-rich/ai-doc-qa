"""
项目配置：所有可调参数都从 .env 文件读取。

为什么用 pydantic-settings？
- 类型安全：.env 里都是字符串，pydantic 自动转成 Python 类型
- 缺值报错：必填字段没填，启动时直接报错，避免运行时才崩
- IDE 提示：访问 Settings 对象的属性有补全
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：app/core/config.py 往上两级就是项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------- Embedding 提供商的默认模型名 ----------
# 不同厂商的模型名完全不同，所以默认值必须跟着提供商走，
# 不能用一个全局默认值（否则切 provider 时会拿错模型名）。
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    # 本地开源模型：中文效果好、免费、离线可用（首次运行会下载约 90MB）
    "local": "BAAI/bge-small-zh-v1.5",
    "zhipu": "embedding-3",
    "openai": "text-embedding-3-small",
}

# 对话提供商允许的取值
ChatProvider = Literal["zhipu", "openai"]
# 向量化提供商允许的取值（比对话多一个 local）
EmbeddingProvider = Literal["local", "zhipu", "openai"]


class Settings(BaseSettings):
    """全局配置，所有地方都通过 get_settings() 拿这个单例。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # 读取这个文件
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里多余的字段忽略，不报错
    )

    # ---------- LLM 配置 ----------
    # 智谱 GLM（推荐学生党，注册送免费额度：https://open.bigmodel.cn/）
    zhipu_api_key: str = Field(default="", description="智谱 API Key")
    zhipu_api_base: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="智谱 API 基础地址",
    )

    # OpenAI（或任何兼容 OpenAI 协议的接口）
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API 基础地址",
    )

    # 对话用哪个提供商：zhipu | openai
    # 用 Literal 而不是 str：写错值会在启动时立刻报错，而不是运行到一半才崩
    chat_provider: ChatProvider = Field(default="zhipu", description="对话 LLM 提供商")

    # 默认模型名（glm-4-flash 是智谱免费模型，适合学生开发调试）
    llm_model: str = Field(default="glm-4-flash", description="对话模型名")

    # ---------- Embedding 配置 ----------
    # 为什么要和 chat_provider 分开？
    # 对话模型可以白嫖（智谱 glm-4-flash 免费），但 Embedding 是按量计费的，
    # 两者解耦后就能做到：对话走云端免费模型 + 向量化走本地免费模型 = 全程零成本。
    embedding_provider: EmbeddingProvider = Field(
        default="local",
        description="向量化提供商：local（本地免费）/ zhipu / openai",
    )
    # 留空表示用该提供商的默认模型（见 DEFAULT_EMBEDDING_MODELS）
    embedding_model: str = Field(default="", description="向量嵌入模型名，留空用提供商默认值")

    # ---------- 应用配置 ----------
    app_name: str = Field(default="中北大学智能问答助手", description="应用名")
    app_version: str = Field(default="0.1.0", description="应用版本")
    debug: bool = Field(default=True, description="是否调试模式")

    # ---------- 服务配置 ----------
    host: str = Field(default="127.0.0.1", description="服务监听地址")
    port: int = Field(default=8000, description="服务监听端口")

    # ---------- CORS 配置 ----------
    # 逗号分隔的来源白名单。留空表示开发模式（放开 localhost，不带通配符）
    cors_origins: str = Field(
        default="",
        description="允许跨域的来源，逗号分隔；留空则用开发默认值",
    )

    # ---------- 向量数据库配置 ----------
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "chroma"),
        description="Chroma 向量库持久化目录",
    )
    chroma_collection_name: str = Field(
        default="nuc_qa_docs",
        description="Chroma 集合名（可理解为数据库里的表名）",
    )

    # ---------- RAG 配置 ----------
    chunk_size: int = Field(default=300, description="文档切片大小（字符数）")
    chunk_overlap: int = Field(default=50, description="切片重叠大小")
    retrieval_top_k: int = Field(default=5, description="检索时返回最相关的 K 条片段")
    # 相关性阈值：低于该分数的片段会被丢弃，避免把无关内容塞进 prompt 干扰 LLM。
    # 实测：真正相关的片段得分 0.3~0.7，无关内容 0.05~0.2（甚至为负）。
    # 设为 0 表示关闭过滤（保留全部 top_k）。
    retrieval_score_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="检索相关性阈值（0 表示不过滤）",
    )

    @property
    def resolved_embedding_model(self) -> str:
        """返回实际生效的 Embedding 模型名。

        逻辑：EMBEDDING_MODEL 显式配置了就优先用它，
        否则按当前 embedding_provider 取对应厂商的默认模型名。
        """
        if self.embedding_model.strip():
            return self.embedding_model.strip()
        return DEFAULT_EMBEDDING_MODELS[self.embedding_provider]

    @property
    def resolved_cors_origins(self) -> list[str]:
        """返回实际生效的 CORS 白名单。

        注意：绝对不允许出现 "*" —— 因为应用同时开了 allow_credentials=True，
        按 CORS 规范通配符 + 携带凭证是非法组合，浏览器会直接拒绝该响应。
        """
        if self.cors_origins.strip():
            origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        else:
            # 开发默认值：把常见的本地端口列全，而不是图省事写 "*"
            origins = [
                "http://localhost:5173",  # Vite 默认端口
                "http://127.0.0.1:5173",
                "http://localhost:8000",  # 同源访问（前端由本服务托管）
                "http://127.0.0.1:8000",
            ]
        return [o for o in origins if o != "*"]


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例。

    @lru_cache 保证只实例化一次，整个进程共享同一个 Settings 对象。
    为什么用单例？.env 读取一次就够，反复读 IO 浪费。
    """
    return Settings()
