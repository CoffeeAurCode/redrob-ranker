"""Turn the job description into a rubric + an "ideal candidate" reference.

This is the testable core of Session 03's Part A. The digit-prefixed CLI
(``src/precompute/02_build_jd_reference.py``) owns the side effects — the one
Gemini call and the embedding pass — and delegates every pure decision to the
functions here so they can be unit-tested without a network or a model:

* :func:`build_rubric_prompt` — the exact prompt for the JD→rubric call.
* :func:`parse_jd_response`   — defensive validation of the LLM's JSON into a
  typed :class:`Rubric` + ideal-candidate paragraph.
* :func:`build_jd_reference` / :func:`save_jd_reference` / :func:`load_jd_reference`
  — assemble, persist and reload ``artifacts/jd_reference.json``.
* :func:`assert_reference_dim` — guards the Session-03 gotcha: the reference
  vector must share the candidates' embedding dimension or cosine is meaningless.

``artifacts/jd_reference.json`` is the single file ``career_sim``, ``role_match``
and ``domain_match`` all key off in later sessions, so its shape is part of the
contract — hence the strict validation here (precompute fails loudly).

Intentionally light: only the standard library is imported, so the validation
path carries no numpy/model/network weight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

from src.embedding import META_FILE

# The artifact this module produces; later sessions read it by this name.
JD_REFERENCE_FILE = "jd_reference.json"

# Keys the rubric must contain. Ordered as they appear to a reader of the JSON.
_REQUIRED_LIST_FIELDS: tuple[str, ...] = (
    "role_archetypes",
    "must_haves",
    "nice_to_haves",
    "hard_disqualifiers",
    "domains",
)
# Accepted keys for the ideal-candidate paragraph (models vary on the exact name).
_IDEAL_TEXT_KEYS: tuple[str, ...] = ("ideal_candidate_description", "ideal_text")


class SeniorityBand(TypedDict, total=False):
    """Experience window the JD targets (years). ``notes`` carries the soft ideal."""

    min_years: float
    max_years: float
    notes: str


class Rubric(TypedDict, total=False):
    """Structured reading of the JD — what the role *means*, not its keywords."""

    role_archetypes: list[str]
    must_haves: list[str]
    nice_to_haves: list[str]
    hard_disqualifiers: list[str]
    domains: list[str]
    seniority_band: SeniorityBand


class JDReference(TypedDict, total=False):
    """The full ``jd_reference.json`` payload (rubric + reference embedding + meta)."""

    rubric: Rubric
    ideal_text: str
    reference_embedding: list[float]
    embedding_dim: int
    model_id: str
    query_prefix: str
    llm_model: str
    jd_source: str
    created: str


class JDReferenceError(ValueError):
    """The LLM response or a loaded reference file did not meet the contract."""


# --------------------------------------------------------------------------- #
# Prompt.                                                                       #
# --------------------------------------------------------------------------- #
_PROMPT_TEMPLATE = """\
You are screening candidates for the role below. Convert the job description into \
a precise screening rubric and a single "ideal candidate" paragraph.

This dataset deliberately punishes keyword matching: a profile can list every AI \
buzzword and still be a poor fit, while a great fit may never use words like "RAG" \
or "Pinecone" yet show — in their career history — that they built retrieval, \
ranking, search or recommendation systems at a product company. Reason about what \
the JD MEANS, not which keywords appear.

Return ONLY a JSON object with exactly this shape (no prose, no markdown). The
// comments explain each field; do NOT include them in your output:

{{
  "rubric": {{
    // job families that fit, e.g. ML engineer, search/retrieval, recsys, applied scientist
    "role_archetypes": [string, ...],
    // non-negotiable capabilities, each phrased as evidence to look for in a career history
    "must_haves": [string, ...],
    // pluses that do not reject a candidate if absent
    "nice_to_haves": [string, ...],
    // the JD's explicit "do NOT want" / disqualifier conditions
    "hard_disqualifiers": [string, ...],
    // relevant problem domains, e.g. retrieval, ranking, NLP/IR, recommendation, evaluation
    "domains": [string, ...],
    "seniority_band": {{
      "min_years": number,   // lower bound of acceptable total experience
      "max_years": number,   // upper bound (use the JD's stated range)
      "notes": string        // nuance: the softer "ideal" sub-range; a guide not a gate
    }}
  }},
  // one dense paragraph describing the ideal hire's career, written for semantic
  // matching against candidate career text (no bullet lists, no skill keyword dumps)
  "ideal_candidate_description": string
}}

Job description:
\"\"\"
{jd_text}
\"\"\"
"""


def build_rubric_prompt(jd_text: str) -> str:
    """The exact prompt sent to the LLM for the single JD→rubric call.

    Pure and deterministic so the prompt itself is testable (and reviewable at
    Stage 5). The caller sets ``temperature=0`` and a JSON response type.
    """
    return _PROMPT_TEMPLATE.format(jd_text=jd_text.strip())


# --------------------------------------------------------------------------- #
# Response validation.                                                          #
# --------------------------------------------------------------------------- #
def parse_jd_response(payload: Any) -> tuple[Rubric, str]:
    """Validate the parsed LLM JSON into a ``(rubric, ideal_text)`` pair.

    Loud on anything malformed (precompute should fail before writing a bad
    artifact). Lists are cleaned to non-empty stripped strings; the seniority band
    is coerced to numeric bounds. Accepts the ideal paragraph under either
    ``ideal_candidate_description`` or ``ideal_text``.
    """
    if not isinstance(payload, dict):
        raise JDReferenceError(f"expected a JSON object, got {type(payload).__name__}")

    rubric_raw = payload.get("rubric")
    if not isinstance(rubric_raw, dict):
        raise JDReferenceError("response is missing a 'rubric' object")

    rubric: Rubric = {
        "role_archetypes": _clean_str_list(rubric_raw.get("role_archetypes"), "role_archetypes"),
        "must_haves": _clean_str_list(rubric_raw.get("must_haves"), "must_haves"),
        "nice_to_haves": _clean_str_list(rubric_raw.get("nice_to_haves"), "nice_to_haves"),
        "hard_disqualifiers": _clean_str_list(
            rubric_raw.get("hard_disqualifiers"), "hard_disqualifiers"
        ),
        "domains": _clean_str_list(rubric_raw.get("domains"), "domains"),
        "seniority_band": _clean_seniority_band(rubric_raw.get("seniority_band")),
    }

    ideal_text = _extract_ideal_text(payload)
    return rubric, ideal_text


def _clean_str_list(value: Any, field: str) -> list[str]:
    """Coerce ``value`` to a non-empty list of stripped, non-empty strings."""
    if not isinstance(value, list):
        raise JDReferenceError(f"rubric.{field} must be a list, got {type(value).__name__}")
    cleaned = [" ".join(str(item).split()) for item in value]
    cleaned = [item for item in cleaned if item]
    if not cleaned:
        raise JDReferenceError(f"rubric.{field} is empty")
    return cleaned


def _clean_seniority_band(value: Any) -> SeniorityBand:
    """Coerce the seniority band to numeric ``min_years``/``max_years`` (+ notes)."""
    if not isinstance(value, dict):
        raise JDReferenceError("rubric.seniority_band must be an object")
    band: SeniorityBand = {
        "min_years": _as_float(value.get("min_years"), "seniority_band.min_years"),
        "max_years": _as_float(value.get("max_years"), "seniority_band.max_years"),
    }
    notes = value.get("notes")
    if notes is not None:
        band["notes"] = " ".join(str(notes).split())
    return band


def _as_float(value: Any, field: str) -> float:
    """Parse a JSON number to float, or fail loudly with the offending field name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JDReferenceError(f"{field} must be a number, got {value!r}")
    return float(value)


def _extract_ideal_text(payload: dict[str, Any]) -> str:
    """Pull the ideal-candidate paragraph out of the response, under either key."""
    for key in _IDEAL_TEXT_KEYS:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return " ".join(raw.split())
    keys = " / ".join(_IDEAL_TEXT_KEYS)
    raise JDReferenceError(f"response is missing a non-empty ideal-candidate paragraph ({keys})")


# --------------------------------------------------------------------------- #
# Assemble / persist / load.                                                    #
# --------------------------------------------------------------------------- #
def build_jd_reference(
    *,
    rubric: Rubric,
    ideal_text: str,
    reference_embedding: list[float],
    model_id: str,
    query_prefix: str,
    llm_model: str,
    jd_source: str,
    created: str,
) -> JDReference:
    """Assemble the ``jd_reference.json`` payload with full provenance.

    ``embedding_dim`` is recorded so :func:`assert_reference_dim` (and Session-06
    loaders) can cheaply verify the reference shares the candidate vector space.
    """
    return {
        "rubric": rubric,
        "ideal_text": ideal_text,
        "reference_embedding": reference_embedding,
        "embedding_dim": len(reference_embedding),
        "model_id": model_id,
        "query_prefix": query_prefix,
        "llm_model": llm_model,
        "jd_source": jd_source,
        "created": created,
    }


def save_jd_reference(out_dir: Path, reference: JDReference) -> Path:
    """Write ``jd_reference.json`` (pretty, UTF-8) and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / JD_REFERENCE_FILE
    path.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_jd_reference(artifacts_dir: Path) -> JDReference:
    """Load and minimally validate ``jd_reference.json`` from ``artifacts_dir``."""
    path = artifacts_dir / JD_REFERENCE_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise JDReferenceError(f"{path} is not a JSON object")
    embedding = data.get("reference_embedding")
    if not isinstance(embedding, list) or not embedding:
        raise JDReferenceError(f"{path} has no reference_embedding")
    return cast(JDReference, data)  # validated shape above; expose the TypedDict view


def assert_reference_dim(reference: JDReference, artifacts_dir: Path) -> None:
    """Fail loudly unless the reference vector matches the candidate embedding dim.

    The Session-03 gotcha: a reference embedded with a different model/dimension
    makes every downstream cosine meaningless. Compares the vector length against
    ``embeddings_meta.json``'s recorded ``dim``.
    """
    meta_path = artifacts_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = int(meta["dim"])
    actual = len(reference.get("reference_embedding") or [])
    if actual != expected:
        raise JDReferenceError(
            f"reference embedding dim {actual} != candidate embedding dim {expected} "
            f"(from {meta_path.name}); reference must use the same model"
        )
