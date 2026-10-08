"""
pytest 全局配置与共享 fixture。

测试的核心原则：**不依赖真实的外部服务**。
- 不打真实 LLM API（没 Key、没余额也能跑）
- 不加载真实 Embedding 模型（避免下载权重、拖慢 CI）
- 不污染真实的向量库（重定向到临时目录）

这样 CI 里任何机器上都能稳定跑绿，这也是 CI 里敢直接执行 pytest 的前提。
"""

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

# 让测试能 import app 包（在没装成 package 的情况下也能跑）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试用的临时目录统一放在工作区内部的 .pytest_tmp/ 下。
# 为什么不直接用 tmp_path / tmp_path_factory？
# 它们依赖系统临时目录（Windows 上是 %TEMP%），而受限沙箱环境下
# 系统临时目录可能不可写，会让整个测试套件在 setup 阶段就全部报错。
# 放在工作区内则任何环境都能跑。
_TMP_ROOT = PROJECT_ROOT / ".pytest_tmp"

# ------------------------------------------------------------------
# 在导入任何 app 模块之前先注入测试环境变量。
#
# 为什么要在导入之前？
# app.core.config 用 pydantic-settings 读取 .env 文件，
# 而环境变量的优先级高于 .env 文件，所以在导入前设好变量，
# 就能让测试完全脱离开发者本机的 .env 内容。
# ------------------------------------------------------------------
os.environ.update(
    {
        "ZHIPU_API_KEY": "test-zhipu-key",
        "OPENAI_API_KEY": "test-openai-key",
        "CHAT_PROVIDER": "zhipu",
        "LLM_MODEL": "glm-4-flash",
        "EMBEDDING_PROVIDER": "local",
        "EMBEDDING_MODEL": "",
        "DEBUG": "true",
        "CHUNK_SIZE": "300",
        "CHUNK_OVERLAP": "50",
        "RETRIEVAL_TOP_K": "5",
        "RETRIEVAL_SCORE_THRESHOLD": "0.15",
        "CORS_ORIGINS": "",
    }
)


@pytest.fixture(autouse=True)
def workspace_tmp():
    """提供一个一定可写的临时目录（放在工作区内）。

    不暴露给测试直接使用：测试请请求 ``tmp_path``，
    由下面这个 fixture 把它指向同一个工作区目录。
    """
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def tmp_path(workspace_tmp):
    """覆盖 pytest 内置的 tmp_path，改成工作区内的目录。

    原因：内置 tmp_path 基于系统临时目录，受限沙箱下不可写，
    会让所有测试在 setup 阶段就报 PermissionError。
    """
    return workspace_tmp


@pytest.fixture(autouse=True)
def _isolate_vectorstore(workspace_tmp, monkeypatch):
    """把向量库重定向到临时目录，避免测试弄脏开发者的 data/chroma。"""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(workspace_tmp / "chroma"))

    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_embeddings(monkeypatch):
    """
    用确定性假向量替换真实 Embedding 模型。

    假向量用"文本长度 + 字符哈希"生成，保证：
    - 同一段文本永远得到同一个向量（可复现）
    - 不触发任何模型下载或网络请求
    """

    class FakeEmbeddings:
        def __init__(self, dim: int = 8) -> None:
            self.dim = dim
            self.embed_calls = 0

        def _vec(self, text: str) -> list[float]:
            self.embed_calls += 1
            # 简单的确定性 hash：同一文本 -> 同一向量
            base = sum(ord(c) for c in text) % 97
            return [((base + i) % 13) / 13.0 for i in range(self.dim)]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vec(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._vec(text)

    fake = FakeEmbeddings()
    monkeypatch.setattr("app.core.embedding.get_embeddings", lambda: fake)
    # vectorstore 模块内部也是通过这个符号拿 embedding 的
    monkeypatch.setattr("app.core.vectorstore.get_embeddings", lambda: fake)
    return fake


@pytest.fixture
def sample_markdown() -> str:
    """一段带多级标题的 Markdown，用于验证清洗逻辑。"""
    return (
        "# 中北大学校园生活\n"
        "\n"
        "## 宿舍\n"
        "\n"
        "本科生宿舍以 6 人间为主。\n"
        "\n"
        "### 用电\n"
        "\n"
        "每间宿舍每月 200 度免费电。\n"
    )
