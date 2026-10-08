"""文档管理路由：上传文件 → 自动切片入库。"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import PROJECT_ROOT
from app.core.document_loader import load_single_file, split_documents
from app.core.logging import get_logger
from app.core.vectorstore import add_documents_to_vectorstore, count_documents
from app.models.schemas import IngestResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["文档管理"])

# 允许的文件类型
_ALLOWED_EXT = {".md", ".txt", ".pdf", ".docx"}
# 上传文件大小上限 10MB
_MAX_FILE_SIZE = 10 * 1024 * 1024
# 资料目录
_RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"


def _safe_filename(filename: str | None) -> str:
    """
    从上传的 filename 里取出安全的文件名。

    为什么必须做这一步？
    上传的 filename 完全由客户端控制，可能长这样：../../app/main.py
    如果直接拼进保存路径，就能覆盖项目里的任意文件（路径穿越漏洞）。
    这里用 Path(...).name 只保留最后一段，把目录部分全部丢掉。
    """
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"非法文件名: {filename}")
    return name


@router.post("/upload", response_model=IngestResponse, summary="上传文档并入库")
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档（md/txt/pdf/docx）"),
) -> IngestResponse:
    """
    上传一个文档，系统自动加载、切片、向量化并入库。

    - 支持 .md / .txt / .pdf / .docx
    - 单文件最大 10MB
    - 上传后可立即用 /qa 提问
    """
    _RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 校验文件名与扩展名
    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext or '(无扩展名)'}，"
            f"仅支持 {', '.join(sorted(_ALLOWED_EXT))}",
        )

    # 2. 读取内容（校验大小）
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content)} bytes，上限 {_MAX_FILE_SIZE} bytes (10MB)",
        )

    # 3. 保存到 raw_docs 目录
    save_path = _RAW_DOCS_DIR / filename
    save_path.write_bytes(content)
    logger.info("文件已保存: %s (%d bytes)", save_path, len(content))

    # 4. 加载 + 切片 + 入库
    #    这些步骤都是同步的（还要调 Embedding），
    #    放在 async 函数里直接调用会阻塞整个事件循环，
    #    并发请求全部卡住。所以丢进线程池执行。
    def _ingest() -> tuple[int, int]:
        docs = load_single_file(save_path)
        for d in docs:
            d.metadata["source"] = filename
        if not docs:
            return 0, count_documents()
        chunks = split_documents(docs)
        added = add_documents_to_vectorstore(chunks)
        return added, count_documents()

    try:
        added, total = await run_in_threadpool(_ingest)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("文件入库失败: %s", e)
        raise HTTPException(status_code=400, detail=f"文件处理失败: {e}") from e

    if added == 0:
        raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

    logger.info("入库 %d 个 chunk，向量库共 %d 条", added, total)
    return IngestResponse(
        filename=filename,
        chunks_added=added,
        total_chunks=total,
        message=f"成功！{filename} 已入库，可立即提问",
    )


@router.get("/list", summary="列出已入库的文档")
async def list_documents() -> dict:
    """列出 raw_docs 目录下所有文件，并附带向量库总条数。"""
    if not _RAW_DOCS_DIR.exists():
        return {"documents": [], "total_chunks": 0}

    files = [
        f.name
        for f in sorted(_RAW_DOCS_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in _ALLOWED_EXT
    ]
    # count_documents 内部已处理异常，失败返回 0
    total = await run_in_threadpool(count_documents)
    return {"documents": files, "total_chunks": total}
