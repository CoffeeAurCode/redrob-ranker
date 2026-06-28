"""The shared offline-LLM provider layer (used by every precompute LLM CLI).

Two precompute stages call a hosted LLM: Session 04's structured extraction
(``03_extract_signals.py``) and Session 09's reasoning generation
(``05_generate_reasoning.py``). Both need the *same* network plumbing — pick a
free-tier backend, build a ``call_fn(prompt) -> str``, wrap it in retry/backoff,
and read the key from ``.env`` — so it lives here once rather than being copied
into each digit-prefixed CLI (which cannot import each other anyway).

The backend is pluggable (``--provider``) because no single free tier's daily cap
carries a whole stage alone: Gemini-flash's free RPD turned out to be ~20, Groq's
llama-70B ~100k tokens/day, and Cerebras (~1M tokens/day) is the workhorse. The
chosen model is recorded in each artifact's provenance.

Golden rule: this module is **precompute only** — it is imported by the
``src/precompute/*`` CLIs, never by ``rank.py``. The heavy client SDKs
(``google.generativeai`` / ``openai``) are imported lazily inside the call
builders so that importing this module (e.g. for ``--dry-run`` or under pytest)
needs neither the packages nor a key.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Transient-error backoff for a single call (rate limit / 5xx): 5 tries, doubling.
_CALL_MAX_ATTEMPTS = 5
_CALL_BASE_BACKOFF = 8.0


class Provider(NamedTuple):
    """One offline LLM backend: where its key lives, its default model, its endpoint.

    ``base_url is None`` selects the native Gemini SDK; any other value is an
    OpenAI-compatible chat endpoint (Groq, Cerebras, OpenRouter, …) driven by the
    ``openai`` client. Switching providers is the project's escape hatch from a
    single free tier's daily cap — the cached artifact records which model produced
    each line.
    """

    env_key: str
    default_model: str
    base_url: str | None
    # For reasoning models (e.g. gpt-oss): cap hidden reasoning so a batch stays
    # well under the tokens/minute limit. ``None`` omits the param entirely (plain
    # chat models like Llama reject it). "low" preserves judgment quality at a
    # fraction of the tokens for these extraction/summarisation tasks.
    reasoning_effort: str | None = None


# Free-tier providers blessed by ``02_LLM_API_GUIDE.md``. Gemini-flash's free RPD
# is only ~20 and Groq's llama-70B is ~100k tokens/day, so neither carries a full
# stage alone; Cerebras (~1M tokens/day) does. Keys live in ``.env`` (gitignored).
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider("GEMINI_API_KEY", "gemini-2.5-flash", None),
    "groq": Provider("GROQ_API_KEY", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    # Cerebras free tier is ~1M tokens/day. gpt-oss-120b is a reasoning model, so we
    # pin reasoning_effort=low to keep each batch predictable (under the 30k/min cap).
    "cerebras": Provider(
        "CEREBRAS_API_KEY", "gpt-oss-120b", "https://api.cerebras.ai/v1", reasoning_effort="low"
    ),
    "openrouter": Provider(
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct",
        "https://openrouter.ai/api/v1",
    ),
}
DEFAULT_PROVIDER = "cerebras"


def make_call_fn(
    *, provider: Provider, model_name: str, api_key: str, label: str
) -> Callable[[str], str]:
    """Build the real ``call_fn`` for the chosen provider, wrapped in backoff retry.

    The heavy client imports are local so the rest of the module works without the
    packages or a key. Transient failures (rate limit / 5xx) are retried with
    exponential backoff *inside* the call, so the callers stay network-agnostic.
    """
    raw = (
        _gemini_call(model_name, api_key)
        if provider.base_url is None
        else _openai_compatible_call(
            base_url=provider.base_url,
            api_key=api_key,
            model_name=model_name,
            reasoning_effort=provider.reasoning_effort,
        )
    )
    return _with_backoff(raw, label)


def _gemini_call(model_name: str, api_key: str) -> Callable[[str], str]:
    """Native Gemini SDK call: JSON-mode, temperature 0."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    model = genai.GenerativeModel(  # type: ignore[attr-defined]
        model_name,
        generation_config={"temperature": 0, "response_mime_type": "application/json"},
    )
    return lambda prompt: str(model.generate_content(prompt).text)


def _openai_compatible_call(
    *, base_url: str, api_key: str, model_name: str, reasoning_effort: str | None = None
) -> Callable[[str], str]:
    """OpenAI-compatible chat call (Groq / Cerebras / OpenRouter): temperature 0.

    JSON-object response mode is intentionally *not* forced: the prompts ask for a
    JSON *array*, and ``io_utils.parse_json_safe`` already strips any fences/prose,
    so leaving the format free avoids the array-vs-object friction of strict mode.
    ``reasoning_effort`` is forwarded only when set (reasoning models); plain chat
    models reject the param, so it stays absent for them.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    # Forwarded as extra request-body fields; ``None`` sends nothing (plain chat models).
    extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else None

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    return call


def _with_backoff(call: Callable[[str], str], label: str) -> Callable[[str], str]:
    """Wrap a raw call with exponential backoff on any exception (rate limit / 5xx)."""

    def wrapped(prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(_CALL_MAX_ATTEMPTS):
            try:
                return call(prompt)
            except Exception as exc:  # provider SDKs raise varied error types
                last_exc = exc
                wait = _CALL_BASE_BACKOFF * (2**attempt)
                reason = str(exc).splitlines()[0][:200]
                logger.warning(
                    "%s call failed (attempt %d): %s -- backing off %.0fs",
                    label,
                    attempt + 1,
                    reason,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"{label} call failed after {_CALL_MAX_ATTEMPTS} attempts") from last_exc

    return wrapped


def require_api_key(env_var: str, *, env_path: Path) -> str:
    """Load ``.env`` from ``env_path`` and return ``env_var`` or fail with a clear message."""
    import os

    from dotenv import load_dotenv

    load_dotenv(env_path)
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(
            f"{env_var} is not set. Add it to .env (this is precompute only; "
            "rank.py never reads it)."
        )
    return api_key
