"""Structured LLM signal extraction — the testable core of Session 04 (precompute 03).

For each shortlisted candidate the LLM reads the *career history* (not the noise
``skills`` list) and returns a small, validated JSON of reasoned signals —
archetype, domain, product-vs-services, seniority fit, whether they built a
ranking/search system, an evidence span, and disqualifier flags. Those signals
feed the Session-06 score (``role_match``/``domain_match``/``product_ratio``/
``seniority_fit``/``built_ranking`` + per-flag penalties).

The digit-prefixed CLI (``src/precompute/03_extract_signals.py``) owns the side
effects — the network calls and reading the 100k pool — and delegates every pure
decision to the functions here so they unit-test without a network or a model:

* :func:`build_rubric_summary` / :func:`build_batch_prompt` — the exact prompt.
* :func:`parse_signals_batch` / :func:`coerce_signal` — defensive validation of
  the model's JSON into typed, clamped :class:`LLMSignals` records.
* :func:`load_signal_cache` / :func:`append_signals` / :func:`pending_ids` — the
  crash-safe JSONL cache so re-runs make **zero** new API calls.
* :func:`extract_signals` — the batched loop with retry + per-candidate fallback,
  parameterized over an injected ``call_fn`` so the whole orchestration is
  testable with a fake client (and so this module imports **no** network code).

Golden rule: this is the heaviest LLM user in the project. Nothing here is
imported by ``rank.py`` — the only handoff to rank time is the cached JSONL file.

Intentionally light: only the standard library is imported, so validation and the
extraction loop carry no numpy/model/network weight.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:  # only for annotations — avoids pulling numpy in via jd_reference.
    from src.jd_reference import Rubric

logger = logging.getLogger(__name__)

# The artifact this stage produces; rank-time features (Session 06) read it by name.
LLM_SIGNALS_FILE = "llm_signals.jsonl"
# Ids that still failed after retry + per-candidate fallback are logged here.
FAILURES_FILE = "llm_signals_failures.json"

# Controlled vocabularies. The model is constrained to these so the downstream
# score keys off a closed set (off-vocabulary labels are rare at temperature 0 and
# fall back to the generic bucket — see :func:`_normalize_enum`).
ROLE_ARCHETYPES: tuple[str, ...] = (
    "ml_engineer",
    "ai_engineer",
    "data_scientist",
    "recsys_search",
    "swe_generic",
    "cv_speech",
    "data_eng",
    "non_tech",
)
DOMAINS: tuple[str, ...] = (
    "nlp_ir",
    "recsys_search",
    "generic_swe",
    "cv_speech",
    "data_eng",
    "non_tech",
)
DISQUALIFIER_FLAGS: tuple[str, ...] = (
    "pure_research",
    "langchain_only_recent",
    "consulting_only",
    "cv_primary",
    "stale_coding",
    "job_hopper",
)

# Fallbacks for off-vocabulary / unparseable fields. A parse glitch should never
# silently penalize a candidate, so a missing numeric judgment defaults to the
# neutral midpoint rather than 0.0; an unknown archetype/domain falls back to the
# generic bucket (logged by the CLI report via the distribution histogram).
_DEFAULT_ARCHETYPE = "swe_generic"
_DEFAULT_DOMAIN = "generic_swe"
_NEUTRAL_SCORE = 0.5
_MAX_EVIDENCE_CHARS = 400


class LLMSignals(TypedDict):
    """One candidate's validated, clamped extraction record (a JSONL cache line)."""

    candidate_id: str
    role_archetype: str
    domain: str
    product_vs_services: float
    seniority_band_fit: float
    built_ranking_or_search: bool
    evidence_span: str
    disqualifier_flags: list[str]


# --------------------------------------------------------------------------- #
# Prompt.                                                                       #
# --------------------------------------------------------------------------- #
_SYSTEM_TEMPLATE = """\
You are a senior technical recruiter screening candidates against the job rubric \
below. For EACH candidate, judge what their CAREER HISTORY actually shows — not \
which buzzwords appear. This dataset deliberately punishes keyword matching: a \
profile can list every AI keyword and still be a weak fit, and a strong fit may \
never say "RAG" or "Pinecone" yet clearly built retrieval, ranking, search, or \
recommendation systems at a product company.

JOB RUBRIC (what the role means):
{rubric_summary}

Return ONLY a JSON array — one object per candidate, in the same order — with \
EXACTLY these keys and nothing else (no prose, no markdown fences):

{{
  "candidate_id": "<echo the exact id given for the candidate>",
  "role_archetype": one of [{archetypes}],
  "domain": one of [{domains}],
  "product_vs_services": number 0.0-1.0,   \
// 1.0 = career at product companies; 0.0 = pure IT services/consulting
  "seniority_band_fit": number 0.0-1.0,    \
// 1.0 = squarely in the target band; lower the further it deviates
  "built_ranking_or_search": true or false, \
// career shows shipping a ranking/search/recommendation/retrieval system to real users
  "evidence_span": \
"<= one short quote or paraphrase from THIS candidate's career history justifying the judgment",
  "disqualifier_flags": subset of [{flags}]  // [] if none clearly apply
}}

Set a disqualifier flag ONLY when the career history clearly supports it:
- pure_research: only academic / research-lab roles, no production deployment.
- langchain_only_recent: AI experience is only recent (<12 months) LLM-framework \
glue, with no pre-LLM-era ML production depth.
- consulting_only: set ONLY if EVERY role is at an IT services/consulting firm \
(e.g. TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) AND no role is at a \
product company. If even one role is at a product company, do NOT set this flag \
(a services-heavy career with some product time is not consulting_only).
- cv_primary: ML expertise primarily computer vision / speech / robotics, \
with little NLP / information-retrieval work.
- stale_coding: no hands-on production coding in roughly the last 18 months \
(shifted to pure architecture / tech-lead).
- job_hopper: frequent ~1-1.5 year job changes mainly for title progression.
"""

_CANDIDATE_TEMPLATE = "[{n}] candidate_id: {cid}\n{profile}"


def build_rubric_summary(rubric: Rubric) -> str:
    """Compact the JD rubric into a few prompt lines (must-haves, disqualifiers, band).

    Pure and deterministic. Drops the embedding-oriented prose and keeps just the
    judgment-relevant structure so the batch prompt stays within context limits.
    """
    lines: list[str] = []
    archetypes = rubric.get("role_archetypes") or []
    if archetypes:
        lines.append("Fitting role archetypes: " + ", ".join(archetypes))

    domains = rubric.get("domains") or []
    if domains:
        lines.append("Relevant domains: " + ", ".join(domains))

    must_haves = rubric.get("must_haves") or []
    if must_haves:
        lines.append("Must-have evidence to look for in the career history:")
        lines.extend(f"  - {item}" for item in must_haves)

    disqualifiers = rubric.get("hard_disqualifiers") or []
    if disqualifiers:
        lines.append("Hard disqualifiers (the JD explicitly does NOT want):")
        lines.extend(f"  - {item}" for item in disqualifiers)

    band = rubric.get("seniority_band") or {}
    lo, hi = band.get("min_years"), band.get("max_years")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        notes = band.get("notes")
        suffix = f" — {notes}" if isinstance(notes, str) and notes else ""
        lines.append(f"Seniority band: {lo:g}-{hi:g} years total experience{suffix}")

    return "\n".join(lines)


def build_batch_prompt(rubric_summary: str, batch: Sequence[tuple[str, str]]) -> str:
    """Assemble the full extraction prompt for one batch of ``(id, profile_text)``.

    Pure and deterministic so the prompt is reviewable and unit-testable. The
    caller sets ``temperature=0`` and a JSON response type.
    """
    system = _SYSTEM_TEMPLATE.format(
        rubric_summary=rubric_summary.strip(),
        archetypes=", ".join(ROLE_ARCHETYPES),
        domains=", ".join(DOMAINS),
        flags=", ".join(DISQUALIFIER_FLAGS),
    )
    candidates = "\n\n".join(
        _CANDIDATE_TEMPLATE.format(n=i, cid=cid, profile=profile.strip())
        for i, (cid, profile) in enumerate(batch, start=1)
    )
    return f"{system}\nCANDIDATES:\n\n{candidates}\n"


# --------------------------------------------------------------------------- #
# Response validation (defensive: clamp, coerce, drop unknowns).                #
# --------------------------------------------------------------------------- #
def parse_signals_batch(payload: Any, expected_ids: Sequence[str]) -> dict[str, LLMSignals]:
    """Validate a parsed LLM response into ``{candidate_id: LLMSignals}``.

    Accepts a JSON array (or a single-key object wrapping one). Each object is
    coerced/clamped by :func:`coerce_signal`; only records whose ``candidate_id``
    is in ``expected_ids`` are kept (a hallucinated id is dropped → the caller
    treats it as missing and falls back). As a single-candidate robustness aid,
    when exactly one id is expected and exactly one record parses, the record is
    bound to that id even if the model echoed it slightly wrong.
    """
    records = _as_record_list(payload)
    expected = set(expected_ids)

    if len(expected_ids) == 1 and len(records) == 1:
        only = coerce_signal(records[0], default_id=expected_ids[0])
        if only is None:
            return {}
        only["candidate_id"] = expected_ids[0]  # trust our id over the model's echo
        return {expected_ids[0]: only}

    result: dict[str, LLMSignals] = {}
    for raw in records:
        signal = coerce_signal(raw)
        if signal is not None and signal["candidate_id"] in expected:
            result[signal["candidate_id"]] = signal  # last echo of a dup id wins
    return result


def _as_record_list(payload: Any) -> list[Any]:
    """Normalize a payload to a list of candidate objects.

    Handles the bare array, and the common deviation where a model wraps the array
    in a single-key object such as ``{"candidates": [...]}`` or ``{"results": [...]}``.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if all(k in payload for k in ("candidate_id", "role_archetype")):
            return [payload]  # a single candidate object, unbatched
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


def coerce_signal(raw: Any, *, default_id: str | None = None) -> LLMSignals | None:
    """Coerce one raw object into a validated :class:`LLMSignals`, or ``None``.

    Returns ``None`` only when there is no usable ``candidate_id`` (and no
    ``default_id`` to bind to). Every other field is coerced defensively: enums are
    normalized to the controlled vocabulary (unknown → generic bucket), numbers are
    clamped to ``[0, 1]`` (missing/garbled → neutral 0.5), the boolean is coerced,
    unknown disqualifier flags are dropped, and the evidence span is truncated.
    """
    if not isinstance(raw, dict):
        return None
    candidate_id = _clean_text(raw.get("candidate_id")) or (default_id or "")
    if not candidate_id:
        return None
    return {
        "candidate_id": candidate_id,
        "role_archetype": _normalize_enum(
            raw.get("role_archetype"), ROLE_ARCHETYPES, _DEFAULT_ARCHETYPE
        ),
        "domain": _normalize_enum(raw.get("domain"), DOMAINS, _DEFAULT_DOMAIN),
        "product_vs_services": _clamp01(raw.get("product_vs_services")),
        "seniority_band_fit": _clamp01(raw.get("seniority_band_fit")),
        "built_ranking_or_search": _coerce_bool(raw.get("built_ranking_or_search")),
        "evidence_span": _clean_text(raw.get("evidence_span"))[:_MAX_EVIDENCE_CHARS],
        "disqualifier_flags": _clean_flags(raw.get("disqualifier_flags")),
    }


def _normalize_enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """Lower/underscore-normalize a label and return it if known, else ``default``."""
    if not isinstance(value, str):
        return default
    key = "_".join(value.strip().lower().replace("-", " ").replace("/", " ").split())
    return key if key in allowed else default


def _clamp01(value: Any) -> float:
    """Parse a number and clamp to ``[0, 1]``; missing/non-numeric → neutral 0.5."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Some models emit "0.8" as a string — accept that, but nothing else.
        try:
            value = float(value)
        except (TypeError, ValueError):
            return _NEUTRAL_SCORE
    return max(0.0, min(1.0, float(value)))


def _coerce_bool(value: Any) -> bool:
    """Coerce a model's boolean (real bool, or 'true'/'yes'/'1' strings)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return False


def _clean_flags(value: Any) -> list[str]:
    """Keep only known disqualifier flags, de-duplicated and sorted (deterministic)."""
    if not isinstance(value, list):
        return []
    seen = {
        normalized
        for item in value
        if (normalized := _normalize_enum(item, DISQUALIFIER_FLAGS, "")) != ""
    }
    return sorted(seen)


def _clean_text(value: Any) -> str:
    """Normalize a value to a single-spaced, stripped string ('' if None/empty)."""
    if value is None:
        return ""
    return " ".join(str(value).split())


# --------------------------------------------------------------------------- #
# Crash-safe JSONL cache.                                                       #
# --------------------------------------------------------------------------- #
def load_signal_cache(path: Path) -> dict[str, LLMSignals]:
    """Load ``llm_signals.jsonl`` into ``{candidate_id: LLMSignals}`` (last wins).

    Tolerant by design — the file is our own append log, so a half-written final
    line from a crashed run is skipped (with a warning) rather than fatal. Each
    line is re-coerced through :func:`coerce_signal` so older entries conform to
    the current schema. A missing file yields an empty cache.
    """
    cache: dict[str, LLMSignals] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d: skipping malformed cache line (%s)", path.name, lineno, exc)
                continue
            signal = coerce_signal(obj)
            if signal is not None:
                cache[signal["candidate_id"]] = signal
    return cache


def append_signals(path: Path, records: Iterable[LLMSignals]) -> None:
    """Append records as one JSON object per line, flushing so a crash keeps them.

    Opened with ``newline=""`` so line endings stay ``\\n`` on every platform (a
    stable, re-parseable JSONL artifact). Called once per batch by
    :func:`extract_signals`, so progress survives an interrupted run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def pending_ids(shortlist: Sequence[str], cache: Mapping[str, Any]) -> list[str]:
    """Shortlist ids not yet in the cache, sorted ascending (deterministic order)."""
    return sorted(cid for cid in shortlist if cid not in cache)


def chunk(seq: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive slices of ``seq`` of length ``size`` (last may be shorter)."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


# --------------------------------------------------------------------------- #
# Batched extraction loop (network injected via ``call_fn`` — testable).        #
# --------------------------------------------------------------------------- #
def extract_signals(
    *,
    profiles: Mapping[str, str],
    rubric_summary: str,
    call_fn: Callable[[str], str],
    on_results: Callable[[list[LLMSignals]], None],
    batch_size: int = 8,
    max_retries: int = 1,
    sleep_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[int, list[str]]:
    """Extract signals for every id in ``profiles`` (assumed already uncached).

    Batches ``batch_size`` candidates per ``call_fn`` prompt. A batch that fails to
    parse or comes back missing candidates is retried up to ``max_retries`` times;
    any still-missing ids then fall back to one-candidate-per-call. Each batch's
    successful records are handed to ``on_results`` immediately (crash-safe append)
    so an interrupted run loses nothing. Ids that fail even the per-candidate
    fallback are returned (the CLI logs them).

    ``call_fn`` is the only side-effecting dependency (the real Gemini call lives in
    the CLI), which keeps this module network-free and lets tests drive it with a
    fake client. Returns ``(num_new_records, failed_ids)``.
    """
    ordered = sorted(profiles)
    total_new = 0
    failed: list[str] = []

    for batch_ids in chunk(ordered, batch_size):
        got = _process_batch(
            batch_ids, profiles, rubric_summary, call_fn, max_retries, sleep_seconds, sleep_fn
        )
        missing = [cid for cid in batch_ids if cid not in got]
        if missing:
            logger.warning(
                "batch left %d/%d unparsed -- per-candidate fallback", len(missing), len(batch_ids)
            )
            for cid in missing:
                one = _process_batch(
                    [cid], profiles, rubric_summary, call_fn, max_retries, sleep_seconds, sleep_fn
                )
                if cid in one:
                    got[cid] = one[cid]
                else:
                    failed.append(cid)

        results = [got[cid] for cid in batch_ids if cid in got]
        if results:
            on_results(results)
            total_new += len(results)

    return total_new, failed


def _process_batch(
    ids: Sequence[str],
    profiles: Mapping[str, str],
    rubric_summary: str,
    call_fn: Callable[[str], str],
    max_retries: int,
    sleep_seconds: float,
    sleep_fn: Callable[[float], None],
) -> dict[str, LLMSignals]:
    """Call the model for one batch, retrying until fully covered or retries exhausted.

    Imports :func:`io_utils.parse_json_safe` lazily to strip code fences / prose from
    the raw response before validation. Returns whatever was parsed — possibly a
    partial dict or empty (the caller handles fallback / logging).
    """
    from src.io_utils import parse_json_safe

    prompt = build_batch_prompt(rubric_summary, [(cid, profiles[cid]) for cid in ids])
    got: dict[str, LLMSignals] = {}
    for attempt in range(max_retries + 1):
        if sleep_seconds > 0:
            sleep_fn(sleep_seconds)  # gentle pacing to respect the free-tier rate limit
        try:
            raw = call_fn(prompt)
            got = parse_signals_batch(parse_json_safe(raw), ids)
        except Exception as exc:
            logger.warning(
                "batch [%s..%s] attempt %d failed: %s", ids[0], ids[-1], attempt + 1, exc
            )
            got = {}
        if all(cid in got for cid in ids):
            return got
        if attempt < max_retries:
            logger.info("retrying batch (got %d/%d) ...", len(got), len(ids))
    return got
