"""测试向量库封装：相关性阈值过滤、维度自检、计数。

这些测试都通过 stub 替身替换掉真实的 Chroma / Embedding，
只验证 search_documents 的过滤策略本身是否正确。

注意：这里验证的是"过滤逻辑"，不是"检索质量"。
真实场景下相似度分数只有相对意义，阈值滤不掉所有不相关片段，
想提升精度需要 rerank 模型（见 README 路线图）。
"""

import pytest
from langchain_core.documents import Document


class FakeVectorStore:
    """假的向量库：返回预设的 (文档, 分数) 列表。"""

    def __init__(self, scored: list[tuple[Document, float]]) -> None:
        self._scored = scored
        self.search_calls = 0
        self.scored_calls = 0

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.search_calls += 1
        return [doc for doc, _ in self._scored[:k]]

    def similarity_search_with_relevance_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        self.scored_calls += 1
        return self._scored[:k]


def _doc(text: str, source: str = "a.md") -> Document:
    return Document(page_content=text, metadata={"source": source})


@pytest.fixture
def patch_vectorstore(monkeypatch):
    """把 get_vectorstore 换成可控的假实现，并固定阈值。"""

    def _install(scored, threshold: float = 0.15):
        from app.core import vectorstore

        fake = FakeVectorStore(scored)
        monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: fake)

        # 固定一份配置，避免受本机 .env 影响
        class _Settings:
            retrieval_top_k = 5
            retrieval_score_threshold = threshold

        monkeypatch.setattr(vectorstore, "get_settings", lambda: _Settings())
        return fake

    return _install


def test_filters_out_low_score_chunks(patch_vectorstore):
    """低于阈值的片段必须被丢掉。"""
    scored = [
        (_doc("宿舍以 6 人间为主"), 0.69),
        (_doc("学校简介"), 0.45),
        (_doc("完全无关的内容"), 0.05),
        (_doc("负分内容"), -0.02),
    ]
    patch_vectorstore(scored, threshold=0.15)

    from app.core.vectorstore import search_documents

    results = search_documents("宿舍几人间")
    texts = [d.page_content for d in results]
    assert "宿舍以 6 人间为主" in texts
    assert "完全无关的内容" not in texts
    assert "负分内容" not in texts


def test_keeps_top1_when_everything_below_threshold(patch_vectorstore):
    """全部低于阈值时保留 Top-1，避免"什么都查不到"。

    否则阈值设高了会让系统对所有问题都回答"资料里没有"。
    """
    scored = [
        (_doc("勉强相关"), 0.10),
        (_doc("更不相关"), 0.02),
    ]
    patch_vectorstore(scored, threshold=0.5)

    from app.core.vectorstore import search_documents

    results = search_documents("随便问问")
    assert len(results) == 1
    assert results[0].page_content == "勉强相关"


def test_threshold_zero_disables_filtering(patch_vectorstore):
    """阈值为 0 时应走普通检索，返回全部 Top-K。"""
    scored = [
        (_doc("相关"), 0.9),
        (_doc("不相关"), 0.01),
        (_doc("负分"), -0.5),
    ]
    fake = patch_vectorstore(scored, threshold=0.0)

    from app.core.vectorstore import search_documents

    results = search_documents("问题")
    assert len(results) == 3
    # 阈值关闭时应走 similarity_search 而不是带分数的接口
    assert fake.search_calls == 1
    assert fake.scored_calls == 0


def test_uses_scored_api_when_threshold_enabled(patch_vectorstore):
    """开启阈值时应该走带分数的检索接口（否则拿不到分数）。"""
    fake = patch_vectorstore([(_doc("a"), 0.9)], threshold=0.15)

    from app.core.vectorstore import search_documents

    search_documents("问题")
    assert fake.scored_calls == 1
    assert fake.search_calls == 0


def test_empty_vectorstore_returns_empty(patch_vectorstore):
    """空向量库不应崩，返回空列表。"""
    patch_vectorstore([], threshold=0.15)

    from app.core.vectorstore import search_documents

    assert search_documents("问题") == []


def test_results_keep_original_order(patch_vectorstore):
    """过滤后必须保持"相似度从高到低"的顺序。"""
    scored = [
        (_doc("最相关"), 0.9),
        (_doc("次相关"), 0.6),
        (_doc("第三"), 0.3),
        (_doc("噪声"), 0.01),
    ]
    patch_vectorstore(scored, threshold=0.15)

    from app.core.vectorstore import search_documents

    results = search_documents("问题")
    assert [d.page_content for d in results] == ["最相关", "次相关", "第三"]


def test_count_documents_swallows_errors(monkeypatch):
    """统计条数失败时应返回 0，而不是把主流程带崩。"""
    from app.core import vectorstore

    def boom():
        raise RuntimeError("chroma 挂了")

    monkeypatch.setattr(vectorstore, "get_vectorstore", boom)
    assert vectorstore.count_documents() == 0


def test_threshold_config_validation():
    """阈值必须在 0~1 之间。"""
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(retrieval_score_threshold=-0.1)
    with pytest.raises(ValidationError):
        Settings(retrieval_score_threshold=1.5)
