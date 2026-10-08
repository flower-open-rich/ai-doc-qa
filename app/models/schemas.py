"""
HTTP 接口的请求/响应数据模型（Pydantic Schema）。

为什么要单独建 models 层？
- 接口"长什么样"集中管理，前后端联调看这一个文件就够
- FastAPI 自动用这些模型生成 OpenAPI 文档
- 请求体校验：字段缺失/类型错直接返回 422，不用业务代码手动判
"""

from pydantic import BaseModel, Field


# ---------- 健康检查 ----------
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    app: str
    version: str


# ---------- 纯 LLM 对话 ----------
class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题",
        examples=["你好"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="会话 ID（多轮对话时传同一个值，首次可不传）",
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    model: str
    provider: str
    session_id: str | None = None


# ---------- RAG 问答 ----------
class QARequest(BaseModel):
    """POST /qa 的请求体。"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题",
        examples=["中北大学宿舍几人间？"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="会话 ID（多轮对话用）",
    )


class QAResponse(BaseModel):
    """RAG 问答返回结构。"""

    question: str = Field(..., description="用户的问题")
    answer: str = Field(..., description="AI 基于资料的回答")
    sources: list[str] = Field(
        default_factory=list,
        description="引用的资料文件名列表",
    )
    model: str = Field(..., description="使用的 LLM 模型名")
    session_id: str | None = None


# ---------- 文件上传 ----------
class IngestResponse(BaseModel):
    """文档入库结果。"""

    filename: str = Field(..., description="上传的文件名")
    chunks_added: int = Field(..., ge=0, description="新增的 chunk 数")
    total_chunks: int = Field(..., ge=0, description="向量库当前总 chunk 数")
    message: str


# ---------- 错误响应 ----------
class ErrorResponse(BaseModel):
    """统一错误返回结构。"""

    detail: str = Field(..., description="错误详情")
    error_code: str = Field(default="INTERNAL_ERROR", description="错误码")
