"""
离线建库脚本：把 data/raw_docs/ 下的资料灌进向量库。

使用方式：
    python -m app.services.ingest_service
    （或者 make ingest）

为什么需要单独的"建库"步骤？
- 资料更新不需要每次启动服务都重灌
- 灌库慢（每段文字都要算一次向量），一次性做完后服务直接查就行
- 数据持久化到磁盘，重启服务数据不丢

注意：换过 Embedding 模型（EMBEDDING_PROVIDER / EMBEDDING_MODEL）
必须重跑本脚本重建，否则新旧向量维度不一致，检索会报错。
"""

import shutil
from pathlib import Path

from app.core.config import get_settings
from app.core.document_loader import load_documents, split_documents
from app.core.vectorstore import add_documents_to_vectorstore


def clear_vectorstore() -> None:
    """
    清空向量库。

    方案：直接物理删除整个持久化目录，让下次创建时是全新空库。
    - 比调 chromadb API 删除 collection 更可靠（chromadb 内部缓存多）
    - 删除目录后所有 client 都会失效，但反正是离线脚本，无所谓
    """
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)

    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"[清空] 已删除目录: {persist_dir}")
    else:
        print(f"[清空] 目录不存在，跳过: {persist_dir}")


def ingest_documents(directory: str | None = None) -> int:
    """
    主流程：加载 → 切片 → 清空旧库 → 灌入新数据。

    Args:
        directory: 资料目录，默认 data/raw_docs/

    Returns:
        入库的 chunk 数量
    """
    print("=" * 60)
    print("开始建库流程")
    print("=" * 60)

    # 1. 加载 + 切片
    docs = load_documents(directory)
    if not docs:
        print("[错误] 没有加载到任何文档，建库终止")
        return 0

    chunks = split_documents(docs)
    if not chunks:
        print("[错误] 切片后无内容，建库终止")
        return 0

    # 2. 清空旧库（避免重复）
    clear_vectorstore()

    # 3. 入库：这一步会逐段计算向量，是整个流程最慢的环节。
    #    本地模型首次运行还需要下载权重（约 90MB），之后走缓存。
    added = add_documents_to_vectorstore(chunks)

    print("=" * 60)
    print(f"建库完成！共入库 {added} 个 chunk")
    print("=" * 60)
    return added


if __name__ == "__main__":
    # 直接运行此脚本即可建库
    ingest_documents()
