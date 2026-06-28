"""Grounded reasoning generation — the testable core of Session 09 (precompute 05).

For the final top 100, every CSV row carries a one-to-two-sentence justification a
recruiter (and a Stage-5 reviewer) reads next to the candidate. The text is graded
for grounding and honesty: a single hallucinated employer or skill undermines the
whole submission's credibility. So the requirements are strict — each reasoning must

* cite **specific facts** from *this* candidate (title, years, what they built per
  their own evidence span, the leading score terms);
* connect to a **JD requirement** (a senior AI/ML engineer for retrieval / ranking /
  search / recommendation at a product company);
* **honestly name gaps** when relevant (a 120-day notice, a services background, a
  CV-leaning profile, limited recent activity);
* **never invent** an employer, tool, metric or skill not present in the profile;
* match the **rank's tone** (rank 1 should read stronger than rank 95) and be varied.

This module holds the pure, deterministic pieces so they unit-test without a network
or a model:

* :func:`build_facts` — distil a candidate + its LLM signals + its score breakdown
  into the grounded :class:`ReasoningFacts` the prompt and the validator both use.
* :func:`build_reasoning_prompt` / :func:`parse_reasoning_batch` — the exact prompt
  and the defensive parse of the model's JSON.
* :func:`validate_reasoning` — the **grounding/hallucination gate**: every
  capitalized term (employer / tool / metric / acronym) and every number in the text
  must appear in the candidate's real data, or the reasoning is rejected.
* :func:`deterministic_reasoning` — a grounded-by-construction templated reasoning
  built only from the score breakdown and real facts. It is the fallback
  ``rank.py`` uses for any top-100 id missing a generated reasoning, which keeps
  **rank.py LLM-free** (the golden rule) while never shipping an empty cell.
* :func:`load_reasoning_cache` / :func:`append_reasoning` — the JSONL artifact keyed
  by ``candidate_id`` that ``rank.py`` joins in.

Golden rule: nothing here imports a network/LLM library (the digit-prefixed CLI owns
the call). :func:`deterministic_reasoning` and the cache loader are the only parts
``rank.py`` touches, and they are pure.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from src.io_utils import Candidate, Profile
from src.llm_signals import LLMSignals
from src.scoring import ScoreResult

logger = logging.getLogger(__name__)

# The artifact this stage produces; rank-time (Session 10) joins it by candidate_id.
REASONING_FILE = "reasoning.jsonl"

# Length/shape guards for one reasoning (1-2 sentences, with a little slack).
_MIN_CHARS = 30
_MAX_CHARS = 360
_MAX_SENTENCES = 3

# Number of leading score terms surfaced to the prompt (the dominant contributions).
_LEADING_TERMS = 3
# Behavioral thresholds that count as an honest "gap" worth naming.
_LONG_NOTICE_DAYS = 90
_LOW_AVAILABILITY = 0.85


class ReasoningRecord(TypedDict):
    """One line of ``reasoning.jsonl``: the grounded text plus its provenance."""

    candidate_id: str
    reasoning: str
    source: str  # "llm" (generated + validated) or "deterministic" (the fallback)


# --------------------------------------------------------------------------- #
# Grounding vocabulary: capitalized tokens the reasoning may use WITHOUT them   #
# being a candidate-specific claim. Only acronyms / camel-case / mid-sentence   #
# Title-case tokens are ever checked (see _is_named_token), so this set is just  #
# the generic role/domain acronyms and allowed proper terms — everything else    #
# capitalized must be grounded in the candidate's own text.                      #
# --------------------------------------------------------------------------- #
_GENERIC_VOCAB: frozenset[str] = frozenset(
    {
        # role / domain acronyms that describe the JD, not a candidate claim
        "ml",
        "ai",
        "nlp",
        "ir",
        "jd",
        "llm",
        "llms",
        "mle",
        # geography we may legitimately reference
        "india",
        "us",
        "usa",
        "uk",
        "eu",
        "apac",
    }
)

# Readable strength phrases for the deterministic fallback, keyed by role archetype.
# Lowercase mid-sentence so they are never treated as named claims (see the validator).
_ARCHETYPE_STRENGTH: Mapping[str, str] = {
    "recsys_search": "a search / recommendation specialist",
    "ml_engineer": "a hands-on applied-ML engineer",
    "ai_engineer": "an AI engineer with production exposure",
    "data_scientist": "a data scientist with modelling depth",
    "data_eng": "a data engineer who builds systems",
    "swe_generic": "a software engineer",
    "cv_speech": "a computer-vision / speech engineer",
    "non_tech": "outside the core engineering track",
}

# Readable, honest gap phrases for each disqualifier flag (lowercase, generic words).
_FLAG_GAP: Mapping[str, str] = {
    "consulting_only": "a services and consulting background rather than product work",
    "cv_primary": "a computer-vision-leaning profile with limited retrieval / ranking work",
    "pure_research": "a research-only track with little production deployment",
    "langchain_only_recent": "only recent llm-framework experience and limited pre-llm depth",
    "stale_coding": "limited hands-on production coding recently",
    "job_hopper": "frequent short tenures",
}


@dataclass(frozen=True)
class ReasoningFacts:
    """Everything grounded that one candidate's reasoning may draw on.

    Built by :func:`build_facts` from the candidate record, its ``llm_signals``
    record and its :class:`~src.scoring.ScoreResult`. ``grounding_terms`` /
    ``grounding_numbers`` are the candidate's *real* vocabulary — the validator
    rejects any capitalized term or number in a reasoning that is not found here.
    """

    candidate_id: str
    rank: int
    title: str
    years: float | None
    country: str
    role_archetype: str
    domain: str
    built_ranking: bool
    product_ratio: float
    seniority_fit: float
    evidence_span: str
    disqualifier_flags: tuple[str, ...]
    final: float
    base_fit: float
    availability: float
    leading_terms: tuple[str, ...]
    notice_period_days: int | None
    last_active_date: str | None
    open_to_work: bool | None
    willing_to_relocate: bool | None
    grounding_terms: frozenset[str]
    grounding_numbers: frozenset[float]


@dataclass(frozen=True)
class ValidationResult:
    """The verdict of :func:`validate_reasoning`: ``ok`` plus the issues that failed."""

    ok: bool
    issues: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Fact assembly.                                                                #
# --------------------------------------------------------------------------- #
def build_facts(
    candidate: Candidate,
    signals: LLMSignals | None,
    result: ScoreResult,
    rank: int,
) -> ReasoningFacts:
    """Distil a scored candidate into the grounded facts a reasoning may use.

    Pure and deterministic. The ``grounding_*`` sets are derived from the
    candidate's free text (title, summary, career descriptions, companies) and its
    own LLM evidence span — the closed world the reasoning is allowed to reference.
    """
    profile: Profile = candidate.get("profile") or Profile()
    redrob = candidate.get("redrob_signals") or {}

    years = _as_float(profile.get("years_of_experience"))
    title = (profile.get("current_title") or "").strip() or "(no title)"
    evidence_span = (signals or {}).get("evidence_span", "") if signals else ""

    corpus = _grounding_corpus(candidate, evidence_span)
    grounding_terms = frozenset(_WORD_RE.findall(corpus.lower()))
    grounding_numbers = _grounding_numbers(corpus, years, redrob, candidate)

    return ReasoningFacts(
        candidate_id=candidate.get("candidate_id") or "",
        rank=rank,
        title=title,
        years=years,
        country=(profile.get("country") or "").strip(),
        role_archetype=(signals or {}).get("role_archetype", "") if signals else "",
        domain=(signals or {}).get("domain", "") if signals else "",
        built_ranking=bool(signals["built_ranking_or_search"]) if signals else False,
        product_ratio=float(signals["product_vs_services"]) if signals else 0.5,
        seniority_fit=float(signals["seniority_band_fit"]) if signals else 0.5,
        evidence_span=evidence_span,
        disqualifier_flags=tuple(signals["disqualifier_flags"]) if signals else (),
        final=result.final,
        base_fit=result.base_fit,
        availability=result.availability,
        leading_terms=_leading_terms(result.contributions),
        notice_period_days=_as_int(redrob.get("notice_period_days")),
        last_active_date=_as_str(redrob.get("last_active_date")),
        open_to_work=_as_bool(redrob.get("open_to_work_flag")),
        willing_to_relocate=_as_bool(redrob.get("willing_to_relocate")),
        grounding_terms=grounding_terms,
        grounding_numbers=grounding_numbers,
    )


def _grounding_corpus(candidate: Candidate, evidence_span: str) -> str:
    """All of a candidate's real free text — the closed world a reasoning may cite.

    Deliberately permissive (title, company names, summary, headline, location and
    every career role's company/title/description, plus the LLM evidence span) so a
    legitimately quoted fact is never wrongly flagged as a hallucination.
    """
    profile: Profile = candidate.get("profile") or Profile()
    parts: list[str] = [
        str(profile.get("current_title") or ""),
        str(profile.get("current_company") or ""),
        str(profile.get("summary") or ""),
        str(profile.get("headline") or ""),
        str(profile.get("location") or ""),
        str(profile.get("country") or ""),
    ]
    for entry in candidate.get("career_history") or []:
        parts.append(str(entry.get("company") or ""))
        parts.append(str(entry.get("title") or ""))
        parts.append(str(entry.get("description") or ""))
    parts.append(evidence_span)
    return " ".join(part for part in parts if part)


def _grounding_numbers(
    corpus: str, years: float | None, redrob: Mapping[str, Any], candidate: Candidate
) -> frozenset[float]:
    """Numbers a reasoning may use: any number in the candidate's text + tenure facts.

    Includes every number appearing in the free text (so a metric quoted from the
    evidence span is allowed) plus the candidate's years-of-experience roundings,
    notice-period days, and each role's duration — the structured numbers that may
    not be spelled out in prose.
    """
    numbers: set[float] = {float(n) for n in _NUMBER_RE.findall(corpus)}
    if years is not None:
        numbers |= {years, math.floor(years), round(years), math.ceil(years)}
    notice = _as_int(redrob.get("notice_period_days"))
    if notice is not None:
        numbers.add(float(notice))
    for entry in candidate.get("career_history") or []:
        months = _as_int(entry.get("duration_months"))
        if months is not None:
            numbers |= {float(months), float(round(months / 12))}
    return frozenset(numbers)


def _leading_terms(contributions: Mapping[str, float]) -> tuple[str, ...]:
    """The names of the dominant base-fit contributions (descending), for the prompt."""
    ranked = sorted(contributions.items(), key=lambda kv: -kv[1])
    return tuple(name for name, value in ranked[:_LEADING_TERMS] if value > 0)


# --------------------------------------------------------------------------- #
# Prompt.                                                                       #
# --------------------------------------------------------------------------- #
_SYSTEM_TEMPLATE = """\
You are a senior technical recruiter writing the one- to two-sentence justification \
that appears next to each ranked candidate for this role. It is read by a hiring \
panel and graded for accuracy and honesty.

THE ROLE (what a good reason connects to):
{jd_summary}

Write ONE reasoning per candidate. Hard rules — a reasoning that breaks any of \
these is wrong:
- Ground every claim in the FACTS given for THAT candidate (their title, years, \
what they built per their own evidence, the leading score terms). Cite something \
specific.
- Connect the candidate to the role above (retrieval / ranking / search / \
recommendation at a product company, senior band).
- If the candidate has a listed GAP, name it honestly in the reasoning (e.g. a long \
notice period, a services/consulting background, a computer-vision lean, limited \
recent activity). Do not hide it; do not invent one either.
- NEVER mention a company, employer, tool, library, metric or skill that is not in \
that candidate's facts. Do not name specific company names at all — describe the \
work, not the brand.
- Higher-ranked candidates should read as stronger, more unreserved; lower-ranked \
ones more measured. Vary the wording — do not reuse a template sentence.
- One or two sentences. No preamble, no candidate id, no bullet points.

Return ONLY a JSON array — one object per candidate, in the same order — each \
exactly:
{{"candidate_id": "<echo the id>", "reasoning": "<the 1-2 sentence justification>"}}
"""

_CANDIDATE_TEMPLATE = """\
[{n}] candidate_id: {cid}
  rank: {rank} of 100
  role: {title}; {years}; based in {country}
  signal: archetype={archetype}, domain={domain}, built ranking/search system={built}
  score: final {final:.2f} ({tone}); leading fit terms: {terms}
  evidence (their own career text): "{evidence}"
  gaps to be honest about: {gaps}\
"""


def build_reasoning_prompt(jd_summary: str, facts_batch: Sequence[ReasoningFacts]) -> str:
    """Assemble the full reasoning prompt for one batch of candidates.

    Pure and deterministic so the prompt is reviewable and unit-testable. The caller
    sets ``temperature=0``. Each candidate is rendered with only its grounded facts
    and any honest gaps, so the model has the material to be specific without
    inventing.
    """
    system = _SYSTEM_TEMPLATE.format(jd_summary=jd_summary.strip())
    blocks = "\n\n".join(_format_candidate(facts) for facts in facts_batch)
    return f"{system}\nCANDIDATES:\n\n{blocks}\n"


def _format_candidate(facts: ReasoningFacts) -> str:
    """Render one candidate's grounded facts + gaps for the prompt."""
    years = f"{facts.years:g} yrs experience" if facts.years is not None else "experience n/a"
    gaps = gap_notes(facts)
    return _CANDIDATE_TEMPLATE.format(
        n=facts.rank,
        cid=facts.candidate_id,
        rank=facts.rank,
        title=facts.title,
        years=years,
        country=facts.country or "n/a",
        archetype=facts.role_archetype or "n/a",
        domain=facts.domain or "n/a",
        built="yes" if facts.built_ranking else "no",
        final=facts.final,
        tone=_rank_tone(facts.rank),
        terms=", ".join(facts.leading_terms) or "n/a",
        evidence=_truncate(facts.evidence_span, 240) or "(none provided)",
        gaps="; ".join(gaps) if gaps else "none",
    )


def gap_notes(facts: ReasoningFacts) -> list[str]:
    """Honest, human-readable gaps for a candidate (empty when there are none).

    Shared by the prompt (so the model is told what to disclose) and the
    deterministic fallback (so it discloses the same things) — one definition of
    "what is worth being honest about" for both paths.
    """
    notes: list[str] = []
    for flag in facts.disqualifier_flags:
        notes.append(_FLAG_GAP.get(flag, flag.replace("_", " ")))
    if facts.notice_period_days is not None and facts.notice_period_days >= _LONG_NOTICE_DAYS:
        notes.append(f"{facts.notice_period_days}-day notice period")
    if facts.availability < _LOW_AVAILABILITY:
        notes.append("limited recent activity / engagement")
    if facts.open_to_work is False:
        notes.append("not currently flagged open to work")
    return notes


# --------------------------------------------------------------------------- #
# Response parsing (defensive, mirrors llm_signals.parse_signals_batch).        #
# --------------------------------------------------------------------------- #
def parse_reasoning_batch(payload: Any, expected_ids: Sequence[str]) -> dict[str, str]:
    """Validate a parsed LLM response into ``{candidate_id: reasoning_text}``.

    Accepts a JSON array (or a single-key object wrapping one). Only records whose
    ``candidate_id`` is expected are kept (a hallucinated id is dropped → the caller
    treats it as missing). When exactly one id is expected and one record parses, the
    record is bound to that id even if the model echoed it slightly wrong.
    """
    records = _as_record_list(payload)
    expected = set(expected_ids)

    if len(expected_ids) == 1 and len(records) == 1:
        text = _clean_reasoning(records[0])
        return {expected_ids[0]: text} if text else {}

    result: dict[str, str] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        cid = _as_str(raw.get("candidate_id"))
        text = _clean_reasoning(raw)
        if cid in expected and text:
            result[cid] = text
    return result


def _as_record_list(payload: Any) -> list[Any]:
    """Normalize a payload to a list of reasoning objects (array, or a wrapped array)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "reasoning" in payload:
            return [payload]
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


def _clean_reasoning(raw: Any) -> str:
    """Extract and whitespace-normalize the ``reasoning`` text from a raw object."""
    if not isinstance(raw, dict):
        return ""
    return " ".join(str(raw.get("reasoning") or "").split())


# --------------------------------------------------------------------------- #
# Grounding validator — the hallucination gate.                                 #
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def validate_reasoning(text: str, facts: ReasoningFacts) -> ValidationResult:
    """Reject a reasoning that is mis-shaped or references facts outside the profile.

    The grounding gate: every *named* token (an acronym, a CamelCase tool, or a
    mid-sentence Title-case word — i.e. anything that reads as a specific employer /
    tool / metric claim) and every number in the text must appear in the candidate's
    own data (:attr:`ReasoningFacts.grounding_terms` / ``grounding_numbers``) or in
    the small generic role vocabulary. Ordinary lowercase description is never
    flagged. Pure and deterministic.
    """
    issues: list[str] = []
    cleaned = text.strip()
    if not cleaned:
        return ValidationResult(False, ("empty reasoning",))
    if len(cleaned) < _MIN_CHARS:
        issues.append(f"too short ({len(cleaned)} chars)")
    if len(cleaned) > _MAX_CHARS:
        issues.append(f"too long ({len(cleaned)} chars)")
    if _sentence_count(cleaned) > _MAX_SENTENCES:
        issues.append(f"too many sentences (>{_MAX_SENTENCES})")

    ungrounded_terms = _ungrounded_terms(cleaned, facts.grounding_terms)
    if ungrounded_terms:
        issues.append("ungrounded terms: " + ", ".join(sorted(set(ungrounded_terms))))

    ungrounded_numbers = _ungrounded_numbers(cleaned, facts.grounding_numbers)
    if ungrounded_numbers:
        issues.append("ungrounded numbers: " + ", ".join(sorted(set(ungrounded_numbers))))

    return ValidationResult(not issues, tuple(issues))


def _sentence_count(text: str) -> int:
    """Count non-empty sentences (split on terminal punctuation)."""
    return sum(1 for part in _SENTENCE_SPLIT_RE.split(text) if part.strip())


def _ungrounded_terms(text: str, grounding_terms: frozenset[str]) -> list[str]:
    """Named tokens (employer / tool / metric / acronym) absent from the candidate's data."""
    bad: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for idx, match in enumerate(_WORD_RE.finditer(sentence)):
            word = match.group()
            if not _is_named_token(word, is_sentence_start=idx == 0):
                continue
            norm = word.lower()
            if norm in _GENERIC_VOCAB or norm in grounding_terms:
                continue
            bad.append(word)
    return bad


def _is_named_token(word: str, *, is_sentence_start: bool) -> bool:
    """True if ``word`` reads as a specific named claim rather than ordinary prose.

    Acronyms (``RAG``, ``NDCG``) and CamelCase tools (``PyTorch``, ``OpenSearch``)
    count anywhere; a Title-case word counts only mid-sentence (sentence-initial
    capitalization is ordinary, not a proper-noun claim). Lowercase words never count.
    """
    if len(word) < 2:
        return False
    if word.isupper():  # acronym: RAG, NDCG, AWS, SQL
        return True
    if any(char.isupper() for char in word[1:]):  # CamelCase: PyTorch, OpenSearch
        return True
    if word[0].isupper() and word[1:].islower():  # Title-case
        return not is_sentence_start
    return False


def _ungrounded_numbers(text: str, grounding_numbers: frozenset[float]) -> list[str]:
    """Numbers in the text not matching any number in the candidate's data (±0.5)."""
    bad: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        value = float(match.group())
        if not any(abs(value - allowed) < 0.5 for allowed in grounding_numbers):
            bad.append(match.group())
    return bad


# --------------------------------------------------------------------------- #
# Variety (cross-candidate; used by the CLI to catch a templated batch).        #
# --------------------------------------------------------------------------- #
def reasoning_fingerprint(text: str) -> str:
    """A normalized skeleton (numbers/punctuation stripped) for near-duplicate detection."""
    skeleton = _NUMBER_RE.sub("#", text.lower())
    return " ".join(re.sub(r"[^a-z#]+", " ", skeleton).split())


def too_similar(text: str, others: Iterable[str], *, threshold: float = 0.85) -> bool:
    """True if ``text`` shares more than ``threshold`` of its word set with any of ``others``.

    A cheap Jaccard on fingerprints — catches a "template with swapped nouns" without
    needing embeddings (the reasonings are short and the bar is deliberately high).
    """
    words = set(reasoning_fingerprint(text).split())
    if not words:
        return False
    for other in others:
        other_words = set(reasoning_fingerprint(other).split())
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        if overlap >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# Deterministic fallback — grounded by construction, so rank.py stays LLM-free. #
# --------------------------------------------------------------------------- #
def deterministic_reasoning(facts: ReasoningFacts) -> str:
    """A grounded, rank-toned, gap-honest reasoning built only from real facts.

    Used by ``rank.py`` for any top-100 id missing a generated reasoning (and by the
    generator when an LLM reasoning fails validation), so no row ever ships empty and
    no row ever leaves the offline path. Built from the candidate's own title, years
    and archetype plus the score breakdown, with any honest gap appended — it passes
    :func:`validate_reasoning` by construction (a test asserts this).
    """
    years = (
        f"{facts.years:g} years' experience" if facts.years is not None else "relevant experience"
    )
    profile = _ARCHETYPE_STRENGTH.get(facts.role_archetype, "a relevant engineering profile")
    tone = _rank_tone(facts.rank).capitalize()
    lead = f"{tone} for the retrieval / ranking role: {facts.title} with {years}, {profile}"
    if facts.built_ranking:
        lead += ", having shipped a ranking or search system"
    lead += "."

    gaps = gap_notes(facts)
    if gaps:
        return f"{lead} Noted gap: {_join_readable(gaps)}."
    return lead


def _rank_tone(rank: int) -> str:
    """A tone label that gets stronger toward rank 1 (drives the rank-aware wording)."""
    if rank <= 10:
        return "top-tier match"
    if rank <= 30:
        return "strong match"
    if rank <= 60:
        return "solid match"
    return "reasonable match"


def _join_readable(items: Sequence[str]) -> str:
    """Join phrases as 'a', 'a and b', or 'a, b and c'."""
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


# --------------------------------------------------------------------------- #
# JSONL cache (mirrors llm_signals' crash-safe append log).                     #
# --------------------------------------------------------------------------- #
def load_reasoning_cache(path: Path) -> dict[str, ReasoningRecord]:
    """Load ``reasoning.jsonl`` into ``{candidate_id: ReasoningRecord}`` (last wins).

    Tolerant of a half-written final line from a crashed run (skipped with a
    warning). A missing file yields an empty cache.
    """
    cache: dict[str, ReasoningRecord] = {}
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
            cid = _as_str(obj.get("candidate_id")) if isinstance(obj, dict) else ""
            text = _clean_reasoning(obj)
            if cid and text:
                cache[cid] = {"candidate_id": cid, "reasoning": text, "source": _source(obj)}
    return cache


def append_reasoning(path: Path, records: Iterable[ReasoningRecord]) -> None:
    """Append records as one JSON object per line, flushing so a crash keeps them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _source(obj: Mapping[str, Any]) -> str:
    """The provenance tag of a cached record (defaults to 'llm' for older lines)."""
    source = obj.get("source")
    return source if source in {"llm", "deterministic"} else "llm"


# --------------------------------------------------------------------------- #
# Small typed accessors (defensive — never crash on an odd field).              #
# --------------------------------------------------------------------------- #
def _truncate(text: str, limit: int) -> str:
    """Trim to ``limit`` chars on a word boundary (for the prompt's evidence line)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_str(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""
