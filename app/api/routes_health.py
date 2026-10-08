"""健康检查路由。"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["系统"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    """用于监控/负载均衡探测服务是否存活。"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
    )
