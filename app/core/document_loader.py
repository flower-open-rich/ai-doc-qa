"""
文档加载与切片。

两个职责：
1. load_documents()：从 data/raw_docs/ 加载所有 .md / .txt / .pdf / .docx 文件
2. split_documents()：把长文档切成 chunk（大小由 config 的 chunk_size 决定）

为什么用 RecursiveCharacterTextSplitter？
- 按 ["\n\n", "\n", "。", " "] 优先级递归切分
- 先尝试按段落切（语义完整），段落太长再按句子切
- overlap=50：相邻 chunk 重叠 50 字，避免把一句话切成两段丢失上下文
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import PROJECT_ROOT, get_settings

# 支持的文件扩展名 → 加载器
_SUPPORTED_EXT = {".md", ".txt", ".pdf", ".docx"}

# 匹配行首的 Markdown 标题：## 宿舍
_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")


def _strip_markdown(text: str) -> str:
    r"""
    清洗 Markdown，让标题和正文处于同一个语义块。

    原因：embedding 模型按段落理解语义，
    如果 '## 宿舍' 和下面 '6 人间为主' 是分开的两段，
    那么"宿舍几人间"这个问题就检索不到这段内容——标题是孤立的、没信息量。

    所以把每一级的标题都"下沉"成它所属内容的第一句话：

        原始：
            # 中北大学校园生活
            ## 宿舍
            本科生宿舍以 6 人间为主...

        转换后：
            中北大学校园生活
            宿舍：本科生宿舍以 6 人间为主...

    注意：早期版本用一句 `re.sub` 想把所有标题一次性拼接，但
    re.sub 不会回头匹配"被替换文本"的开头（`# 中北大学校园生活` 替换掉后，
    `## 宿舍` 刚好顶到行首，却已经错过了这一轮的匹配位置），
    结果留下 `中北大学校园生活：## 宿舍` 这种半截标题。这里改成逐行状态机处理。
    """
    # 1. 先去掉行内标记，避免它们干扰标题识别
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"`([^`]+)`", r"\1", text)  # `code`
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url)

    lines = text.split("\n")
    first_heading: str | None = None
    sections: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        """把当前累积的段落收尾。"""
        if buf:
            block = "\n".join(buf).strip()
            if block:
                sections.append(block)
            buf.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if not m:
            buf.append(line)
            continue

        # 遇到标题：先把上一段收尾
        flush()
        title = m.group(2)
        if m.group(1) == "#" and first_heading is None:
            # 一级标题是文档标题，单独成段，不往下粘（它统管全文）
            first_heading = title
        else:
            # 其他层级标题：预置到 buffer 里，等下面正文来了拼成
            # "标题：正文第一句" —— 别忘了把标题自身的 '#' 去掉
            buf.append(f"{title}：")

    flush()

    parts: list[str] = []
    if first_heading:
        parts.append(first_heading)

    for block in sections:
        # 只合并"标题："和它后面正文之间的那个换行，
        # 让标题和正文落在同一行。
        cleaned = re.sub(r"：\s*\n+\s*", "：", block)
        # 注意：正文内部的空行必须保留（\n\n）。
        # RecursiveCharacterTextSplitter 的首选分隔符就是 "\n\n"，
        # 如果把空行压成单个换行，段落边界就消失了，
        # 切片器只能退化成按句号切，容易把一句话切断。
        cleaned = cleaned.strip()
        if cleaned:
            parts.append(cleaned)

    # 文档标题单独成段的话会变成一个没有信息量的小 chunk，
    # 所以把它和第一段正文合并成一个段落。
    if len(parts) >= 2 and first_heading:
        merged = [f"{parts[0]}\n{parts[1]}", *parts[2:]]
        return "\n\n".join(merged)
    return "\n\n".join(parts)


def load_single_file(file_path: Path) -> list[Document]:
    """加载单个文件，返回 Document 列表。"""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        # PDF 用 pypdf 加载
        from langchain_community.document_loaders import PyPDFLoader

        pdf_loader = PyPDFLoader(str(file_path))
        return pdf_loader.load()

    if ext == ".docx":
        # Word 文档用 docx2txt 抽纯文本（依赖 python-docx）
        from langchain_community.document_loaders import Docx2txtLoader

        docx_loader = Docx2txtLoader(str(file_path))
        return docx_loader.load()

    if ext in (".md", ".txt"):
        # 文本类用 TextLoader
        from langchain_community.document_loaders import TextLoader

        text_loader = TextLoader(str(file_path), encoding="utf-8")
        docs = text_loader.load()
        # 对 markdown 文件做预处理：去掉 # 标题标记，提升检索效果
        if ext == ".md":
            for d in docs:
                d.page_content = _strip_markdown(d.page_content)
        return docs

    print(f"[警告] 不支持的文件类型 {ext}，跳过: {file_path}")
    return []


def load_documents(directory: str | Path | None = None) -> list[Document]:
    """
    加载目录下所有支持格式的文档。

    Args:
        directory: 目录路径，默认 data/raw_docs/

    Returns:
        Document 列表，每个 Document 有 page_content (str) 和 metadata (dict)
        metadata 包含 source 字段（文件路径），方便后面回答时引用来源
    """
    docs_dir = Path(directory) if directory else PROJECT_ROOT / "data" / "raw_docs"

    if not docs_dir.exists():
        print(f"[警告] 文档目录不存在: {docs_dir}")
        return []

    all_docs: list[Document] = []
    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _SUPPORTED_EXT:
            continue
        print(f"[加载] {file_path.name}")
        docs = load_single_file(file_path)
        # 给每个 Document 加上 source 元数据（用于回答时引用来源）
        for d in docs:
            # TextLoader 默认把 source 设成完整路径，覆盖为文件名更友好
            d.metadata["source"] = file_path.name
        all_docs.extend(docs)

    print(f"[完成] 共加载 {len(all_docs)} 个文档段")
    return all_docs


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    把长文档切成小 chunk。

    Args:
        documents: load_documents 返回的 Document 列表
        chunk_size: 切片大小，默认从配置读 chunk_size（见 config.py）
        chunk_overlap: 重叠大小，默认从配置读 chunk_overlap
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        # 切分优先级：先按段落（双换行），再按单换行，再按中文句号，再按空格
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[切片] {len(documents)} 个文档 → {len(chunks)} 个 chunk (size={size}, overlap={overlap})"
    )
    return chunks


if __name__ == "__main__":
    # 直接运行此文件可以单独测试加载 + 切片
    docs = load_documents()
    chunks = split_documents(docs)
    if chunks:
        print("\n--- 第一个 chunk 示例 ---")
        print(f"来源: {chunks[0].metadata.get('source')}")
        print(f"长度: {len(chunks[0].page_content)} 字符")
        print(f"内容预览: {chunks[0].page_content[:150]}...")
