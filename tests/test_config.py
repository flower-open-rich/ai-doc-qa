"""测试配置层：provider 解析、CORS 白名单合法性、Embedding 模型默认值。"""

import pytest
from pydantic import ValidationError

from app.core.config import (
    DEFAULT_EMBEDDING_MODELS,
    Settings,
    get_settings,
)


def test_openai_fields_exist():
    """回归测试：OpenAI 相关的配置字段必须存在。

    之前 llm.py / embedding.py 访问 settings.openai_api_key，
    但 config.py 里根本没定义这两个字段，一设 CHAT_PROVIDER=openai
    就抛 AttributeError。这个测试专门防止该 bug 复现。
    """
    settings = get_settings()
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "openai_api_base")
    assert "openai_api_key" in type(settings).model_fields
    assert "openai_api_base" in type(settings).model_fields


def test_provider_field_is_separate():
    """对话提供商和 Embedding 提供商必须是两个独立字段。"""
    fields = Settings.model_fields
    assert "chat_provider" in fields
    assert "embedding_provider" in fields
    # 旧字段名应该已经被移除，避免两套配置并存产生歧义
    assert "llm_provider" not in fields


def test_resolved_embedding_model_uses_provider_default():
    """EMBEDDING_MODEL 留空时，应按 provider 取对应厂商的默认模型名。"""
    for provider, expected in DEFAULT_EMBEDDING_MODELS.items():
        settings = Settings(embedding_provider=provider, embedding_model="")
        assert settings.resolved_embedding_model == expected


def test_resolved_embedding_model_respects_explicit_value():
    """显式配置了 EMBEDDING_MODEL 时，必须优先使用它。"""
    settings = Settings(embedding_provider="zhipu", embedding_model="embedding-2")
    assert settings.resolved_embedding_model == "embedding-2"


def test_cors_origins_never_contain_wildcard():
    """CORS 白名单里绝不能出现 "*"。

    本服务开启了 allow_credentials=True，
    按 CORS 规范通配符来源 + 携带凭证是非法组合，浏览器会直接拒绝，
    所以配置层必须把它过滤掉。
    """
    settings = Settings(cors_origins="*")
    assert "*" not in settings.resolved_cors_origins

    settings2 = Settings(cors_origins="*,https://example.com")
    assert "*" not in settings2.resolved_cors_origins
    assert "https://example.com" in settings2.resolved_cors_origins


def test_cors_origins_parses_comma_separated():
    """逗号分隔的白名单要能正确切分并去掉空白。"""
    settings = Settings(cors_origins="https://a.com, https://b.com ,")
    assert settings.resolved_cors_origins == ["https://a.com", "https://b.com"]


def test_cors_origins_default_is_localhost_only():
    """留空时应回落到 localhost 开发默认值，而不是放开一切。"""
    settings = Settings(cors_origins="")
    origins = settings.resolved_cors_origins
    assert origins, "默认白名单不应为空"
    assert all("localhost" in o or "127.0.0.1" in o for o in origins)


def test_invalid_provider_rejected_at_startup():
    """provider 写错时应该在实例化阶段就报错，而不是运行到一半才崩。"""
    with pytest.raises(ValidationError):
        Settings(embedding_provider="not-a-provider")

    with pytest.raises(ValidationError):
        Settings(chat_provider="not-a-provider")


def test_settings_is_cached_singleton():
    """get_settings() 必须是单例（@lru_cache），避免反复读 .env。"""
    get_settings.cache_clear()
    assert get_settings() is get_settings()
