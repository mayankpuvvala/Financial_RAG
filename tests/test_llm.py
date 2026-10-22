from types import SimpleNamespace

import pytest
from groq import RateLimitError

import llm
from llm import clean_citations, retry_after_seconds


def test_fullwidth_citations_become_brackets():
    assert clean_citations("Sales were $1 billion【1】 and $2 billion【2†L9-L12】.") == \
        "Sales were $1 billion[1] and $2 billion[2]."


def test_plain_text_untouched():
    assert clean_citations("No citations [3] here.") == "No citations [3] here."


def test_retry_after_parsing():
    assert retry_after_seconds("Please try again in 877.5ms.") == pytest.approx(0.8775)
    assert retry_after_seconds("try again in 12.3s") == pytest.approx(12.3)
    assert retry_after_seconds("try again in 2m3.5s") == pytest.approx(123.5)
    assert retry_after_seconds("no hint here") is None


def _resp(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish)])


def _rate_limit(msg):
    import httpx
    req = httpx.Request("POST", "https://api.groq.com/x")
    return RateLimitError(msg, response=httpx.Response(429, request=req), body=None)


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def _use(monkeypatch, script):
    fake = FakeClient(script)
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    return fake


def test_empty_length_reply_retried_with_bigger_budget(monkeypatch):
    fake = _use(monkeypatch, [_resp("", "length"), _resp('{"ok": 1}')])
    assert llm.complete("openai/gpt-oss-20b", [], max_tokens=600) == '{"ok": 1}'
    assert [c["max_tokens"] for c in fake.calls] == [600, 1200]
    assert fake.calls[0]["reasoning_effort"] == "low"


def test_budget_growth_is_bounded(monkeypatch):
    fake = _use(monkeypatch, [_resp("", "length")] * 10)
    assert llm.complete("openai/gpt-oss-20b", [], max_tokens=1500) == ""
    assert max(c["max_tokens"] for c in fake.calls) == llm.MAX_TOKEN_CEILING
    assert len(fake.calls) <= 4


def test_truncated_but_nonempty_reply_is_returned(monkeypatch):
    fake = _use(monkeypatch, [_resp("partial answer", "length")])
    assert llm.complete("openai/gpt-oss-120b", [], max_tokens=100) == "partial answer"
    assert len(fake.calls) == 1


def test_per_minute_rate_limit_is_retried(monkeypatch):
    fake = _use(monkeypatch, [_rate_limit("tokens per minute (TPM). Please try again in 877.5ms."), _resp("hello")])
    assert llm.complete("openai/gpt-oss-20b", [], max_tokens=100) == "hello"
    assert len(fake.calls) == 2


def test_daily_rate_limit_is_not_retried(monkeypatch):
    fake = _use(monkeypatch, [_rate_limit("tokens per day (TPD): Limit 100000. try again in 5m1s")])
    with pytest.raises(RateLimitError):
        llm.complete("openai/gpt-oss-20b", [], max_tokens=100)
    assert len(fake.calls) == 1


def test_exhausted_billing_balance_is_not_retried(monkeypatch):
    msg = ("Error code: 429 - You have no credits remaining. Add credits to continue. "
           "type: insufficient_quota, code: credit_balance_exhausted. Please try again in 1s")
    fake = _use(monkeypatch, [_rate_limit(msg)])
    with pytest.raises(RateLimitError):
        llm.complete("gpt-4.1-mini", [], max_tokens=100)
    assert len(fake.calls) == 1


def test_rate_limit_gives_up_after_retries(monkeypatch):
    fake = _use(monkeypatch, [_rate_limit("TPM. try again in 1s")] * 10)
    with pytest.raises(RateLimitError):
        llm.complete("openai/gpt-oss-20b", [], max_tokens=100)
    assert len(fake.calls) == llm.MAX_RATE_LIMIT_RETRIES + 1
