"""
Single entry point for chat completions (Groq or OpenAI, per settings.llm_provider).

Groq retired llama-3.1-8b-instant / llama-3.3-70b-versatile; the replacements
(openai/gpt-oss-20b, openai/gpt-oss-120b) are reasoning models. Three things
follow, and they live here so no caller has to know about them:

  1. `reasoning_effort` must be low and the reasoning text excluded.
  2. Even so, the hidden reasoning is counted against max_tokens: some prompts
     burn the whole budget thinking and come back with finish_reason="length"
     and EMPTY content (observed: ~5 in 6 classifier calls on one query).
     An empty reply is retried with a doubled budget.
  3. The free tier's per-minute token limit (8000 TPM on gpt-oss-20b) clears
     within seconds, so a 429 is retried after the wait Groq asks for. A
     per-DAY limit won't clear, so that one is re-raised immediately.
"""

import re
import time
from functools import lru_cache
from typing import Optional

import groq
from loguru import logger

from config import settings

# Exception classes for the configured provider(s), so callers can catch API
# failures without caring which one is active (an `except` accepts a tuple).
# The OpenAI SDK is a heavy import and only needed when it is the provider, so
# it is not loaded at all on the default Groq setup (matters on small hosts).
def _error_classes(name: str) -> tuple:
    classes = [getattr(groq, name)]
    if settings.llm_provider == "openai":
        import openai
        classes.append(getattr(openai, name))
    return tuple(classes)


RateLimitError     = _error_classes("RateLimitError")
APIConnectionError = _error_classes("APIConnectionError")
APITimeoutError    = _error_classes("APITimeoutError")
APIStatusError     = _error_classes("APIStatusError")
# One FLAT tuple for `except`: Python 3 rejects a tuple that contains tuples.
TransientErrors    = APIConnectionError + APITimeoutError + APIStatusError

# An empty-reply retry doubles max_tokens. On Groq's free tier a request
# (prompt + max_tokens) must stay under the 8000 tokens/minute limit or it is
# rejected outright with a 413, so cap the budget there.
MAX_TOKEN_CEILING = 6000 if settings.llm_provider == "openai" else 2000
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_WAIT = 30.0

# gpt-oss emits citations as 【1】 / 【1†L9-L12】; the rest of the pipeline (and
# the UI) speaks [1].
_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+)(?:\s*[†‡][^】]*)?】")
_RETRY_IN = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", re.IGNORECASE)


def clean_citations(text: str) -> str:
    return _FULLWIDTH_CITATION.sub(lambda m: f"[{m.group(1)}]", text)


def retry_after_seconds(message: str) -> Optional[float]:
    """Seconds Groq says to wait ('try again in 877.5ms' / '2m3.4s'), if stated."""
    m = _RETRY_IN.search(message)
    if not m or not any(m.groups()):
        return None
    hours, minutes, num, unit = m.groups()
    total = int(hours or 0) * 3600 + int(minutes or 0) * 60
    if num:
        total += float(num) / (1000 if unit.lower() == "ms" else 1)
    return total


def _is_daily_limit(message: str) -> bool:
    """A limit that won't clear within seconds: a daily cap, or an exhausted
    billing balance (OpenAI answers that with a 429 too: insufficient_quota)."""
    low = message.lower()
    return any(s in low for s in (
        "per day", "(tpd)", "(rpd)", "insufficient_quota", "no credits", "credit_balance", "billing",
    ))


@lru_cache(maxsize=1)
def get_client():
    """Both SDKs expose the same client.chat.completions.create(...)."""
    if settings.llm_provider == "openai":
        import openai
        return openai.OpenAI(api_key=settings.openai_api_key)
    return groq.Groq(api_key=settings.groq_api)


def _model_kwargs(model: str, max_tokens: int, temperature: float) -> dict:
    if model.startswith("openai/gpt-oss"):      # Groq-hosted reasoning model
        return {"max_tokens": max_tokens, "temperature": temperature,
                "reasoning_effort": "low", "include_reasoning": False}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):   # OpenAI reasoning models
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": temperature}


def _create(model: str, messages: list, max_tokens: int, temperature: float):
    """One API call, retrying per-minute rate limits."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return get_client().chat.completions.create(
                model=model, messages=messages, **_model_kwargs(model, max_tokens, temperature),
            )
        except RateLimitError as exc:   # tuple: Groq or OpenAI
            message = str(exc)
            wait = retry_after_seconds(message)
            if _is_daily_limit(message) or attempt == MAX_RATE_LIMIT_RETRIES or (wait or 0) > MAX_RATE_LIMIT_WAIT:
                raise
            wait = min((wait if wait is not None else 2.0 * (attempt + 1)) + 0.5, MAX_RATE_LIMIT_WAIT)
            logger.warning(f"Rate limit ({model}); retrying in {wait:.1f}s")
            time.sleep(wait)


def complete(
    model:       str,
    messages:    list,
    max_tokens:  int,
    temperature: float = 0.0,
) -> str:
    """Run one chat completion and return the stripped text of the reply."""
    budget = max_tokens
    while True:
        response = _create(model, messages, budget, temperature)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if text or choice.finish_reason != "length" or budget >= MAX_TOKEN_CEILING:
            return clean_citations(text)
        budget = min(budget * 2, MAX_TOKEN_CEILING)
        logger.warning(f"{model} spent its whole token budget reasoning (empty reply); retrying with max_tokens={budget}")
