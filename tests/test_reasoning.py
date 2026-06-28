"""Tests for the reasoning-generation core (``src/reasoning.py``).

Two layers, mirroring ``test_llm_signals.py``:

* **Grounding** — the hallucination gate: a reasoning that names an employer / tool /
  metric or a number absent from the candidate's data is rejected, while one that
  only references real facts is accepted.
* **Fallback + plumbing** — the deterministic grounded reasoning passes its own
  validator for every archetype/rank (so ``rank.py`` always has a valid string), plus
  the defensive parser, the JSONL cache round-trip, and the variety check.

The digit-prefixed CLI owns the real network call, so it is not imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.io_utils import Candidate
from src.llm_signals import DISQUALIFIER_FLAGS, LLMSignals
from src.reasoning import (
    _ARCHETYPE_STRENGTH,  # internal map: every archetype must round-trip
    ReasoningFacts,
    ReasoningRecord,
    append_reasoning,
    build_facts,
    deterministic_reasoning,
    gap_notes,
    load_reasoning_cache,
    parse_reasoning_batch,
    reasoning_fingerprint,
    too_similar,
    validate_reasoning,
)
from src.scoring import score
from tests.factories import make_features


def _signals(**overrides: Any) -> LLMSignals:
    """A well-formed LLM signal record for one candidate, with optional overrides."""
    base: dict[str, Any] = {
        "candidate_id": "CAND_0000042",
        "role_archetype": "ml_engineer",
        "domain": "nlp_ir",
        "product_vs_services": 1.0,
        "seniority_band_fit": 1.0,
        "built_ranking_or_search": True,
        "evidence_span": "Built a ranking and retrieval system evaluated with NDCG.",
        "disqualifier_flags": [],
    }
    base.update(overrides)
    return cast(LLMSignals, base)


def _make_facts(
    *,
    title: str = "ML Engineer",
    summary: str = "",
    evidence: str = "Built a ranking and retrieval system.",
    years: float = 6.0,
    country: str = "India",
    flags: list[str] | None = None,
    role: str = "ml_engineer",
    built: bool = True,
    rank: int = 5,
    notice: int = 30,
    availability: float = 0.90,
    open_to_work: bool = True,
    career: list[dict[str, Any]] | None = None,
) -> ReasoningFacts:
    """Build :class:`ReasoningFacts` from a synthetic candidate + signals + score."""
    candidate = cast(
        Candidate,
        {
            "candidate_id": "CAND_0000042",
            "profile": {
                "current_title": title,
                "summary": summary,
                "country": country,
                "years_of_experience": years,
            },
            "career_history": career or [],
            "redrob_signals": {
                "notice_period_days": notice,
                "last_active_date": "2026-06-01",
                "open_to_work_flag": open_to_work,
                "willing_to_relocate": True,
            },
        },
    )
    signals = _signals(
        evidence_span=evidence,
        role_archetype=role,
        built_ranking_or_search=built,
        disqualifier_flags=flags or [],
    )
    result = score(make_features("CAND_0000042", availability=availability))
    return build_facts(candidate, signals, result, rank)


# --------------------------------------------------------------------------- #
# build_facts                                                                   #
# --------------------------------------------------------------------------- #
def test_build_facts_collects_grounding_from_real_text() -> None:
    facts = _make_facts(
        title="Search Engineer",
        evidence="Built embedding retrieval with NDCG.",
        years=7.6,
        notice=120,
    )
    # Terms come from the candidate's own text (lower-cased word tokens).
    assert "search" in facts.grounding_terms and "ndcg" in facts.grounding_terms
    # Numbers include the year roundings and the notice period.
    assert 8.0 in facts.grounding_numbers and 120.0 in facts.grounding_numbers
    assert facts.leading_terms  # the dominant contributions were captured


# --------------------------------------------------------------------------- #
# validate_reasoning — grounding gate                                           #
# --------------------------------------------------------------------------- #
def test_grounded_reasoning_passes() -> None:
    facts = _make_facts(title="Search Engineer", evidence="Built embedding retrieval with NDCG.")
    text = (
        "Strong search engineer who built embedding retrieval evaluated with NDCG, "
        "squarely matching the senior ranking role."
    )
    assert validate_reasoning(text, facts).ok


def test_hallucinated_employer_is_rejected() -> None:
    """A reasoning naming an employer absent from the profile is rejected."""
    facts = _make_facts()
    result = validate_reasoning("Excellent ranking engineer who built systems at Google.", facts)
    assert not result.ok
    assert any("Google" in issue for issue in result.issues)


def test_hallucinated_tool_or_skill_is_rejected() -> None:
    facts = _make_facts(evidence="Built a ranking system in production.")
    result = validate_reasoning(
        "Solid engineer with deep PyTorch and Pinecone experience on the ranking stack.", facts
    )
    assert not result.ok
    issues = " ".join(result.issues)
    assert "PyTorch" in issues and "Pinecone" in issues


def test_acronym_present_in_evidence_is_grounded() -> None:
    """An acronym the candidate actually used (RAG/NDCG) is not a hallucination."""
    facts = _make_facts(evidence="Shipped a RAG pipeline evaluated with NDCG in production.")
    text = "Strong engineer who shipped a RAG pipeline evaluated with NDCG for the ranking role."
    assert validate_reasoning(text, facts).ok


def test_sentence_initial_capitalization_is_not_flagged() -> None:
    """Capitalizing the first word of a sentence is ordinary, not a proper-noun claim."""
    facts = _make_facts()
    text = "Strong applied engineer. Their ranking and retrieval work fits the senior role well."
    assert validate_reasoning(text, facts).ok


def test_wrong_number_is_rejected_but_real_number_passes() -> None:
    facts = _make_facts(years=6.0)
    assert validate_reasoning("Engineer with 6 years on ranking and retrieval systems.", facts).ok
    bad = validate_reasoning("Engineer with 14 years on ranking and retrieval systems.", facts)
    assert not bad.ok
    assert any("14" in issue for issue in bad.issues)


def test_empty_and_overlong_and_run_on_are_rejected() -> None:
    facts = _make_facts()
    assert not validate_reasoning("", facts).ok
    assert not validate_reasoning("   ", facts).ok
    too_long = "ranking " * 80  # all lowercase (grounded) but way over the char cap
    assert not validate_reasoning(too_long, facts).ok
    run_on = (
        "the engineer built ranking systems. the engineer shipped search. "
        "the engineer evaluated relevance. the engineer owned the work."
    )
    result = validate_reasoning(run_on, facts)
    assert not result.ok
    assert any("sentences" in issue for issue in result.issues)


# --------------------------------------------------------------------------- #
# deterministic fallback — grounded by construction                             #
# --------------------------------------------------------------------------- #
def test_deterministic_reasoning_is_grounded_for_every_archetype_and_rank() -> None:
    """rank.py's fallback must always pass the validator (guards vocab completeness)."""
    for role in _ARCHETYPE_STRENGTH:
        for rank in (1, 15, 45, 95):
            facts = _make_facts(role=role, rank=rank, title="ML Engineer")
            text = deterministic_reasoning(facts)
            assert text
            verdict = validate_reasoning(text, facts)
            assert verdict.ok, f"{role}@{rank}: {text!r} -> {verdict.issues}"


def test_deterministic_reasoning_discloses_a_real_gap() -> None:
    facts = _make_facts(flags=["consulting_only"], notice=120)
    text = deterministic_reasoning(facts).lower()
    assert "consult" in text or "services" in text
    assert "120" in text  # the long notice period is named honestly


def test_deterministic_reasoning_tone_tracks_rank() -> None:
    top = deterministic_reasoning(_make_facts(rank=1))
    low = deterministic_reasoning(_make_facts(rank=95))
    assert top != low
    assert top.lower().startswith("top-tier")
    assert low.lower().startswith("reasonable")


def test_deterministic_reasoning_is_grounded_with_a_disqualifier_flag_present() -> None:
    for flag in DISQUALIFIER_FLAGS:
        facts = _make_facts(flags=[flag], notice=120, availability=0.80, open_to_work=False)
        assert validate_reasoning(deterministic_reasoning(facts), facts).ok


# --------------------------------------------------------------------------- #
# gap_notes                                                                     #
# --------------------------------------------------------------------------- #
def test_gap_notes_surface_flags_and_behavioural_gaps() -> None:
    notes = gap_notes(
        _make_facts(flags=["cv_primary"], notice=120, availability=0.70, open_to_work=False)
    )
    joined = " ".join(notes).lower()
    assert "vision" in joined  # the cv_primary flag
    assert "120-day" in joined  # the long notice period
    assert any("activity" in note for note in notes)  # low availability
    assert any("open to work" in note for note in notes)


def test_gap_notes_empty_for_a_clean_strong_candidate() -> None:
    assert gap_notes(_make_facts(notice=15, availability=0.93, open_to_work=True)) == []


# --------------------------------------------------------------------------- #
# parse_reasoning_batch                                                         #
# --------------------------------------------------------------------------- #
def test_parse_reasoning_batch_array_and_whitespace() -> None:
    payload = [
        {"candidate_id": "CAND_0001", "reasoning": "Strong   ranking\n engineer."},
        {"candidate_id": "CAND_0002", "reasoning": "Solid retrieval engineer."},
    ]
    out = parse_reasoning_batch(payload, ["CAND_0001", "CAND_0002"])
    assert out["CAND_0001"] == "Strong ranking engineer."  # collapsed whitespace
    assert set(out) == {"CAND_0001", "CAND_0002"}


def test_parse_reasoning_batch_wrapped_object() -> None:
    payload = {"results": [{"candidate_id": "CAND_0001", "reasoning": "Strong fit."}]}
    assert parse_reasoning_batch(payload, ["CAND_0001"]) == {"CAND_0001": "Strong fit."}


def test_parse_reasoning_batch_single_binding_and_drops_unexpected() -> None:
    # A single expected id binds even if the model echoed a different id.
    one = parse_reasoning_batch(
        [{"candidate_id": "WRONG", "reasoning": "Strong fit."}], ["CAND_0001"]
    )
    assert one == {"CAND_0001": "Strong fit."}
    # In a multi-id batch, an unexpected id is dropped.
    many = parse_reasoning_batch(
        [{"candidate_id": "CAND_0009", "reasoning": "x"}], ["CAND_0001", "CAND_0002"]
    )
    assert many == {}


# --------------------------------------------------------------------------- #
# JSONL cache                                                                   #
# --------------------------------------------------------------------------- #
def test_cache_roundtrip_preserves_source(tmp_path: Path) -> None:
    path = tmp_path / "reasoning.jsonl"
    records: list[ReasoningRecord] = [
        {"candidate_id": "CAND_0001", "reasoning": "Strong fit.", "source": "llm"},
        {"candidate_id": "CAND_0002", "reasoning": "Solid fit.", "source": "deterministic"},
    ]
    append_reasoning(path, records)
    loaded = load_reasoning_cache(path)
    assert loaded["CAND_0001"]["source"] == "llm"
    assert loaded["CAND_0002"]["source"] == "deterministic"
    assert loaded["CAND_0002"]["reasoning"] == "Solid fit."


def test_cache_skips_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "reasoning.jsonl"
    path.write_text(
        '{"candidate_id": "CAND_0001", "reasoning": "ok", "source": "llm"}\n{ broken\n',
        encoding="utf-8",
    )
    loaded = load_reasoning_cache(path)
    assert set(loaded) == {"CAND_0001"}


def test_missing_cache_is_empty(tmp_path: Path) -> None:
    assert load_reasoning_cache(tmp_path / "nope.jsonl") == {}


# --------------------------------------------------------------------------- #
# variety                                                                       #
# --------------------------------------------------------------------------- #
def test_too_similar_detects_near_duplicates() -> None:
    base = "Strong ranking engineer with seven years building retrieval systems for search."
    assert too_similar(base, [base])  # exact repeat
    # Same skeleton with only a number swapped — fingerprints match (a templated row).
    assert too_similar(
        "Solid engineer who shipped retrieval ranking with 7 years in production.",
        ["Solid engineer who shipped retrieval ranking with 9 years in production."],
    )
    # A genuinely different reasoning is not flagged as templated.
    assert not too_similar("A measured data scientist focused on analytics dashboards.", [base])


def test_fingerprint_strips_numbers() -> None:
    a = reasoning_fingerprint("Engineer with 6 years.")
    b = reasoning_fingerprint("Engineer with 9 years.")
    assert a == b  # numbers normalized away
