"""
Chroma 向量库封装。

Chroma 是一个轻量级本地向量数据库：
- 持久化到磁盘（不需要单独跑服务，开发期足够用）
- 自带 Embedding 算子（但我们让它用我们的 Embeddings，保持提供商可切换）
- 支持元数据过滤（比如只检索某个来源的资料）

为什么不用 Pinecone/Milvus？
- 那些需要单独部署服务，对开发期来说太重
- Chroma 数据直接落在本地一个目录，部署简单
- 量级到了几十万条以上再考虑迁移
"""

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.embedding import get_embeddings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_vectorstore(
    collection_name: str | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    """
    获取 Chroma 向量库实例。

    不加 @lru_cache：
    - 因为每次添加文档 / 检索都用新实例更安全（避免并发问题）
    - Chroma 客户端初始化开销很小（就是连一个本地文件夹）

    Args:
        collection_name: 集合名（类似数据库表名），默认从配置读
        persist_directory: 持久化目录，默认 data/chroma/
    """
    settings = get_settings()
    collection = collection_name or settings.chroma_collection_name
    persist_dir = persist_directory or settings.chroma_persist_dir

    # 确保目录存在（Chroma 不会自动创建）
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def check_embedding_consistency() -> tuple[bool, str]:
    """
    校验"当前 Embedding 模型"和"库里已有向量"的维度是否一致。

    为什么需要这个检查？
    向量库里每条向量是固定维度（比如本地 bge-small 是 512 维，智谱 embedding-3 是 2048 维）。
    如果换了 EMBEDDING_PROVIDER / EMBEDDING_MODEL 却没重新建库，
    Chroma 会抛出一段很难懂的底层异常。提前检测就能给出人话提示。

    Returns:
        (是否一致, 提示信息)
    """
    settings = get_settings()
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        if count == 0:
            return True, "向量库为空，尚未建库（首次建库会自动使用当前 Embedding 模型）"

        # 取一条已有向量，拿到它的维度
        sample = vectorstore._collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return True, "无法读取历史向量维度，跳过检查"

        stored_dim = len(embeddings[0])
        current_dim = len(get_embeddings().embed_query("维度探测"))
    except Exception as e:  # 探测失败不应阻断主流程
        return True, f"维度检查跳过（{type(e).__name__}: {e}）"

    if stored_dim != current_dim:
        return False, (
            f"Embedding 维度不一致：库里已有向量是 {stored_dim} 维，"
            f"但当前配置（{settings.embedding_provider} / "
            f"{settings.resolved_embedding_model}）产出 {current_dim} 维。\n"
            "  换过 Embedding 模型后必须重建向量库，否则检索会报错。\n"
            "  修复：python -m app.services.ingest_service"
        )
    return True, f"维度一致（{stored_dim} 维）"


def add_documents_to_vectorstore(documents: list[Document]) -> int:
    """
    把文档列表灌进向量库。

    会自动调用 Embedding 模型把每段文字转成向量，
    然后连同元数据一起存进 Chroma。

    Returns:
        添加的文档数量
    """
    if not documents:
        logger.warning("文档列表为空，跳过入库")
        return 0

    vectorstore = get_vectorstore()
    # Chroma.add_documents 会自动生成唯一 ID（UUID）
    vectorstore.add_documents(documents)
    logger.info("入库 %d 个 chunk 到向量库", len(documents))
    return len(documents)


def count_documents() -> int:
    """
    返回向量库里的 chunk 总数。

    为什么写在这里？
    Chroma 0.5.x 没有公开的 count() 接口，只能通过底层 collection 拿，
    所以把"碰私有属性"这件事收敛到一个地方，别让路由层到处伸手。
    出错时返回 0，避免统计信息把主流程带崩。
    """
    try:
        # _collection.count() 在 chromadb 里没有类型标注，mypy 视为 Any
        return int(get_vectorstore()._collection.count())
    except Exception as e:
        logger.warning("读取向量库条数失败: %s: %s", type(e).__name__, e)
        return 0


def search_documents(query: str, k: int | None = None) -> list[Document]:
    """
    在向量库里检索和 query 最相关的 K 个 chunk。

    带相关性过滤：得分低于 retrieval_score_threshold 的片段会被丢掉。
    为什么需要？
    Chroma 的 Top-K 检索一定会返回 K 条结果，哪怕库里根本没有相关内容。
    实测"今天天气怎么样"这种完全无关的问题，Top-1 得分只有 0.057，
    不加过滤时这 5 条噪声会照样塞进 prompt 干扰 LLM。

    但要清醒地认识到这个过滤的能力边界：
    相似度分数只有"相对"意义，没有绝对阈值。实测"学校前身是什么"
    这种能答对的问题，命中片段也只有 0.31 分，而一堆不相干的片段能到 0.27 分。
    所以阈值只能滤掉明显不相关的噪声，无法做到精准判定"这段到底相不相关"。
    真正想提升精度，应该上 rerank 模型（比如 bge-reranker）对候选做二次排序，
    这已列入 README 的路线图。

    安全兜底：如果全部片段都低于阈值，仍然保留得分最高的那一条，
    避免因为阈值设得太高导致"什么都查不到"。

    Args:
        query: 用户问题
        k: 返回前 K 个，默认从配置读 retrieval_top_k

    Returns:
        Document 列表，按相似度从高到低排序
    """
    settings = get_settings()
    top_k = k or settings.retrieval_top_k
    threshold = settings.retrieval_score_threshold

    vectorstore = get_vectorstore()

    if threshold <= 0:
        return vectorstore.similarity_search(query, k=top_k)

    scored = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    if not scored:
        return []

    # 注意：Chroma 返回的是"距离"，而 LangChain 会把它换算成相关性分数，
    # 在本项目的配置下分数越大越相关（也可能是负数）。
    above = [doc for doc, score in scored if score >= threshold]
    if above:
        return above

    # 全部低于阈值：保留最高分那条，让 LLM 自己判断能否回答
    logger.debug(
        "检索结果全部低于阈值 %.2f（最高 %.4f），退化为保留 Top-1",
        threshold,
        scored[0][1],
    )
    return [scored[0][0]]
