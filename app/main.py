"""
FastAPI 应用入口。

职责：
1. 创建 app、配置日志
2. 注册中间件（CORS）
3. 注册全局异常处理
4. 挂载路由（health / chat / qa）
5. （后续）挂载静态前端页面

启动：
    conda run -n nuc_qa uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    routes_chat,
    routes_documents,
    routes_health,
    routes_qa,
    routes_qa_stream,
)
from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import get_logger, setup_logging


# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子。"""
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    logger.info("=" * 50)
    logger.info("启动 %s v%s", settings.app_name, settings.app_version)
    logger.info("调试模式: %s", settings.debug)
    logger.info("对话模型: %s / %s", settings.chat_provider, settings.llm_model)
    logger.info(
        "向量模型: %s / %s",
        settings.embedding_provider,
        settings.resolved_embedding_model,
    )
    logger.info("=" * 50)

    # 启动时检查 Embedding 维度是否和库里已有向量一致。
    # 换过 Embedding 模型却没重建库的话，检索会报一句很难懂的底层异常，
    # 这里提前给出人话提示。检查失败不阻断启动（比如库是空的）。
    try:
        from app.core.vectorstore import check_embedding_consistency

        # 本地模型加载 / 远程 API 探测都是阻塞操作，丢到线程池里避免卡住事件循环
        ok, message = await run_in_threadpool(check_embedding_consistency)
        if ok:
            logger.info("向量库自检: %s", message)
        else:
            logger.error("向量库自检未通过: %s", message)
    except Exception as e:
        logger.warning("向量库自检跳过: %s: %s", type(e).__name__, e)

    yield
    get_logger(__name__).info("服务已停止")


# ---------- 创建应用 ----------
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="基于 RAG 的中北大学智能问答助手 - 后端 API",
    version=settings.app_version,
    lifespan=lifespan,
)


# ---------- 中间件 ----------
# CORS：浏览器跨域访问控制。前后端分离时前端（不同端口/域名）必须在后端白名单里。
#
# 关键点：allow_origins 里绝对不能出现 "*"。
# 因为下面开了 allow_credentials=True，而按 CORS 规范，
# "通配符来源 + 携带凭证" 是非法组合，浏览器会直接拒绝该响应——
# 配了等于没配。生产环境请用 CORS_ORIGINS 环境变量显式列出域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.resolved_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 全局异常处理 ----------
# 把底层异常转成统一的 JSON 错误结构，避免直接抛 500 堆栈给前端
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """业务错误（配置缺失等）→ 400。"""
    get_logger(__name__).warning("业务错误: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_code": "BUSINESS_ERROR"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预期异常 → 500（生产环境不暴露堆栈细节）。"""
    get_logger(__name__).exception("未处理异常: %s", exc)
    detail = str(exc) if settings.debug else "服务器内部错误"
    return JSONResponse(
        status_code=500,
        content={"detail": detail, "error_code": "INTERNAL_ERROR"},
    )


# ---------- 注册路由 ----------
app.include_router(routes_health.router)
app.include_router(routes_chat.router)
app.include_router(routes_qa.router)
app.include_router(routes_qa_stream.router)
app.include_router(routes_documents.router)


# ---------- 静态资源 ----------
# 前端页面目录
STATIC_DIR = PROJECT_ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["系统"], summary="前端页面")
async def root():
    """根路径直接返回前端 HTML 页面（用户打开浏览器就能用）。"""
    return FileResponse(str(STATIC_DIR / "index.html"))
