import pytest

from config import Settings
from llm import _model_kwargs


def _settings(**kw):
    return Settings(_env_file=None, edgar_email="a@b.co", **kw)


def test_groq_is_default_and_needs_groq_key():
    s = _settings(groq_api="g")
    assert s.llm_provider == "groq"
    assert s.generation_model == "openai/gpt-oss-120b"
    with pytest.raises(ValueError):
        _settings()


def test_openai_provider_switches_default_models():
    s = _settings(llm_provider="openai", openai_api_key="k")
    assert s.generation_model == "gpt-4.1"
    assert s.routing_model == "gpt-4.1-mini"


def test_openai_provider_respects_explicit_models():
    s = _settings(llm_provider="openai", openai_api_key="k", generation_model="gpt-4.1-mini")
    assert s.generation_model == "gpt-4.1-mini"


def test_openai_provider_requires_key():
    with pytest.raises(ValueError):
        _settings(llm_provider="openai")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        _settings(groq_api="g", llm_provider="bedrock")


def test_model_kwargs_per_model_family():
    assert _model_kwargs("openai/gpt-oss-20b", 10, 0.0)["reasoning_effort"] == "low"
    assert "max_completion_tokens" in _model_kwargs("gpt-5-mini", 10, 0.0)
    plain = _model_kwargs("gpt-4.1", 10, 0.1)
    assert plain == {"max_tokens": 10, "temperature": 0.1}
