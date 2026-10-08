"""
日志配置。

为什么不用 print？
- print 无法关闭（上线后调试信息也全打出来）
- print 没有时间/级别/模块名，出问题查不到源头
- logging 可以分级（DEBUG/INFO/WARNING/ERROR），可以写文件，可以对接监控

用法：
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("启动完成")
    logger.error("调用失败: %s", e)
"""

import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """
    配置全局日志（应用启动时调用一次）。

    格式：时间 | 级别 | 模块名 | 消息
    级别：DEBUG 调试模式打印 DEBUG 以上，生产只打印 INFO 以上
    """
    settings = get_settings()
    level = logging.DEBUG if settings.debug else logging.INFO

    # 日志格式
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 用 root logger 统一配置（避免 uvicorn / 第三方库日志格式不一致）
    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler（uvicorn --reload 会触发多次）
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root.addHandler(handler)

    # 降低常见噪音库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain_community").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取一个带模块名的 logger。"""
    return logging.getLogger(name)
