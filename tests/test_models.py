"""测试模型工厂：Embedding 与 LLM 的 provider 分发逻辑。

这里只验证"工厂按配置选对了实现、传递了对的参数"，
不做任何网络调用（构造客户端对象本身不联网）。
"""

import pytest

from app.core.config import Settings


def _fresh_settings(**overrides):
    """构造一份独立的 Settings 实例，避免污染全局单例。

    只显式传我们关心的字段（没有额外字段，Settings 开了 extra="ignore" 也无所谓）。
    """
    base = {
        "chat_provider": "zhipu",
        "embedding_provider": "local",
        "embedding_model": "",
        "zhipu_api_key": "test-key",
        "zhipu_api_base": "https://open.bigmodel.cn/api/paas/v4",
        "openai_api_key": "test-key",
        "openai_api_base": "https://api.openai.com/v1",
        "llm_model": "glm-4-flash",
    }
    base.update(overrides)
    return Settings(**base)


class TestEmbeddingFactory:
    """get_embeddings() 的 provider 分发。"""

    def test_local_provider_returns_huggingface_embeddings(self, monkeypatch):
        from app.core import embedding

        settings = _fresh_settings(embedding_provider="local")
        monkeypatch.setattr(embedding, "get_settings", lambda: settings)
        embedding.get_embeddings.cache_clear()

        emb = embedding.get_embeddings()
        assert type(emb).__name__ == "HuggingFaceEmbeddings"
        # 默认模型应该是本地中文小模型
        assert "bge-small-zh" in settings.resolved_embedding_model

    def test_zhipu_provider_requires_api_key(self, monkeypatch):
        from app.core import embedding

        settings = _fresh_settings(embedding_provider="zhipu", zhipu_api_key="")
        monkeypatch.setattr(embedding, "get_settings", lambda: settings)
        embedding.get_embeddings.cache_clear()

        with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
            embedding.get_embeddings()

    def test_openai_provider_requires_api_key(self, monkeypatch):
        """回归测试：OpenAI 分支之前访问了不存在的配置字段会抛 AttributeError，
        现在应该给出明确的 ValueError 提示。"""
        from app.core import embedding

        settings = _fresh_settings(embedding_provider="openai", openai_api_key="")
        monkeypatch.setattr(embedding, "get_settings", lambda: settings)
        embedding.get_embeddings.cache_clear()

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            embedding.get_embeddings()

    def test_openai_provider_builds_client(self, monkeypatch):
        from app.core import embedding

        settings = _fresh_settings(embedding_provider="openai", openai_api_key="sk-test")
        monkeypatch.setattr(embedding, "get_settings", lambda: settings)
        embedding.get_embeddings.cache_clear()

        emb = embedding.get_embeddings()
        assert type(emb).__name__ == "OpenAIEmbeddings"


class TestLLMFactory:
    """get_llm() 的 provider 分发。"""

    def test_zhipu_provider_requires_api_key(self, monkeypatch):
        from app.core import llm

        settings = _fresh_settings(chat_provider="zhipu", zhipu_api_key="")
        monkeypatch.setattr(llm, "get_settings", lambda: settings)
        llm._get_llm_cached.cache_clear()

        with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
            llm.get_llm()

    def test_openai_provider_requires_api_key(self, monkeypatch):
        """回归测试：之前这里是 AttributeError: 'Settings' object has no attribute
        'openai_api_key'，属于一用就崩的 bug。"""
        from app.core import llm

        settings = _fresh_settings(chat_provider="openai", openai_api_key="")
        monkeypatch.setattr(llm, "get_settings", lambda: settings)
        llm._get_llm_cached.cache_clear()

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            llm.get_llm()

    def test_openai_provider_builds_client(self, monkeypatch):
        from app.core import llm

        settings = _fresh_settings(
            chat_provider="openai",
            openai_api_key="sk-test",
            llm_model="gpt-4o-mini",
        )
        monkeypatch.setattr(llm, "get_settings", lambda: settings)
        llm._get_llm_cached.cache_clear()

        model = llm.get_llm()
        assert type(model).__name__ == "ChatOpenAI"
        assert model.model_name == "gpt-4o-mini"
        # api_key 字段是 SecretStr，要确认密钥真的传进去了而不是空值
        assert model.openai_api_key.get_secret_value() == "sk-test"

    def test_zhipu_provider_builds_client(self, monkeypatch):
        """智谱分支必须把 api_base 补成完整的 /chat/completions 地址。

        这是一个容易踩的坑：ChatZhipuAI 把 zhipu_api_base 当成完整请求 URL 用，
        不会自动拼路径后缀，所以代码里要手动补。
        """
        from app.core import llm

        settings = _fresh_settings(
            chat_provider="zhipu",
            zhipu_api_key="test-key",
            zhipu_api_base="https://open.bigmodel.cn/api/paas/v4",
        )
        monkeypatch.setattr(llm, "get_settings", lambda: settings)
        llm._get_llm_cached.cache_clear()

        model = llm.get_llm()
        assert type(model).__name__ == "ChatZhipuAI"
        # 注意字段名是 zhipuai_api_base（不是 zhipu_api_base）
        assert str(model.zhipuai_api_base).endswith("/chat/completions")
        assert model.model_name == "glm-4-flash"

    def test_streaming_flag_is_cached_separately(self, monkeypatch):
        """streaming=True/False 必须各缓存一个实例，不能混用。"""
        from app.core import llm

        settings = _fresh_settings(chat_provider="zhipu", zhipu_api_key="test-key")
        monkeypatch.setattr(llm, "get_settings", lambda: settings)
        llm._get_llm_cached.cache_clear()

        plain = llm.get_llm(streaming=False)
        stream = llm.get_llm(streaming=True)
        assert plain is not stream
        # 同一个 flag 应该命中缓存
        assert llm.get_llm(streaming=True) is stream
