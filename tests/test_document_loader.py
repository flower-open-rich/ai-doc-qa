"""测试文档加载与切片：Markdown 清洗、多级标题、切片质量。"""

import re

from langchain_core.documents import Document

from app.core.document_loader import (
    _strip_markdown,
    load_documents,
    split_documents,
)


class TestStripMarkdown:
    """Markdown 清洗逻辑。"""

    def test_heading_merged_with_body(self, sample_markdown):
        """二级标题要和它的正文粘成一句，否则"宿舍"标题是孤立的、检索不到。"""
        cleaned = _strip_markdown(sample_markdown)
        assert "宿舍：本科生宿舍以 6 人间为主。" in cleaned

    def test_no_residual_hash_marks(self, sample_markdown):
        """清洗后不能残留任何 # 标记。

        回归测试：早期版本用一个 re.sub 处理所有标题，
        因为 re.sub 不会回头匹配被替换文本的开头，
        会留下 "校园生活：## 宿舍" 这种半截标题。
        """
        cleaned = _strip_markdown(sample_markdown)
        assert "##" not in cleaned
        assert not re.search(r"^#{1,6}\s", cleaned, re.MULTILINE)

    def test_nested_headings_all_converted(self, sample_markdown):
        """三级标题同样要被处理，不能只处理前两级。"""
        cleaned = _strip_markdown(sample_markdown)
        assert "用电：每间宿舍每月 200 度免费电。" in cleaned

    def test_document_title_kept(self, sample_markdown):
        """一级标题是文档标题，要保留下来（但不能单独成段变成垃圾 chunk）。"""
        cleaned = _strip_markdown(sample_markdown)
        assert cleaned.startswith("中北大学校园生活")

    def test_inline_formatting_removed(self):
        """粗体 / 斜体 / 行内代码 / 链接都要还原成纯文本。"""
        text = "# 标题\n\n**粗体** 和 *斜体* 以及 `代码` 和 [链接](http://a.com)"
        cleaned = _strip_markdown(text)
        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "http://a.com" not in cleaned
        assert "粗体" in cleaned
        assert "斜体" in cleaned
        assert "链接" in cleaned

    def test_plain_text_unchanged(self):
        """没有 Markdown 标记的纯文本不应被改动。"""
        text = "这是一段普通的文本。\n\n第二段。"
        assert _strip_markdown(text).strip() == text.strip()

    def test_empty_input(self):
        """空输入不应抛异常。"""
        assert _strip_markdown("") == ""


class TestLoadAndSplit:
    """加载与切片。"""

    def test_load_real_documents(self):
        """能加载仓库里自带的示例文档。"""
        docs = load_documents()
        assert docs, "应该能加载到 data/raw_docs 下的示例文档"
        for d in docs:
            assert isinstance(d, Document)
            assert d.page_content.strip()
            # source 必须被改写为文件名（方便引用溯源展示）
            assert d.metadata["source"]
            assert "/" not in d.metadata["source"]
            assert "\\" not in d.metadata["source"]

    def test_load_missing_directory_returns_empty(self, tmp_path):
        """目录不存在时应返回空列表而不是抛异常。"""
        assert load_documents(tmp_path / "not-exist") == []

    def test_split_respects_chunk_size(self):
        """切片结果不应明显超过配置的 chunk_size。"""
        from app.core.config import get_settings

        size = get_settings().chunk_size
        chunks = split_documents(load_documents())
        assert chunks
        for c in chunks:
            # RecursiveCharacterTextSplitter 在极端情况下会有少量溢出，给点余量
            assert len(c.page_content) <= size * 2, (
                f"chunk 过长: {len(c.page_content)} > {size * 2}"
            )

    def test_split_preserves_metadata(self):
        """切片后 metadata（尤其是 source）必须跟着走，否则无法溯源。"""
        chunks = split_documents(load_documents())
        assert all("source" in c.metadata for c in chunks)

    def test_split_empty_input(self):
        """空输入返回空列表。"""
        assert split_documents([]) == []

    def test_every_chunk_is_searchable_text(self):
        """每个 chunk 都应该是有内容的文本（不能出现全空白 chunk）。"""
        chunks = split_documents(load_documents())
        assert all(c.page_content.strip() for c in chunks)
