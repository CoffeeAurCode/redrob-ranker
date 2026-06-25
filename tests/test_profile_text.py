"""Tests for profile_text — the skills-free candidate-text builder.

The headline assertion is the skill-leakage guard from ``plan/00_OVERVIEW.md``
risk #4: if a distinctive skill name ever appears in the built text, the
embedding/LLM input is polluted and every downstream sanity check is too.
"""

from __future__ import annotations

from src.io_utils import Candidate
from src.profile_text import build_embedding_text, build_llm_profile


def test_embedding_text_excludes_skills(sample_candidate: Candidate) -> None:
    text = build_embedding_text(sample_candidate)
    distinctive_skill = sample_candidate["skills"][0]["name"]  # the leakage canary
    assert distinctive_skill not in text
    assert "Python" not in text  # the second skill must not leak either


def test_embedding_text_includes_title_summary_and_career(sample_candidate: Candidate) -> None:
    text = build_embedding_text(sample_candidate)
    assert "ML Engineer" in text  # current title
    assert "Seven years building search" in text  # summary
    assert "hybrid retrieval and ranking system" in text  # career description
    assert "offline evaluation harness" in text  # second role's description


def test_embedding_text_is_deterministic(sample_candidate: Candidate) -> None:
    assert build_embedding_text(sample_candidate) == build_embedding_text(sample_candidate)


def test_embedding_text_normalizes_whitespace() -> None:
    candidate: Candidate = {
        "profile": {"current_title": "ML   Engineer", "summary": "line one\n\nline two\t end"},
        "career_history": [],
    }
    text = build_embedding_text(candidate)
    assert "  " not in text
    assert "\n" not in text
    assert "\t" not in text


def test_embedding_text_respects_char_cap() -> None:
    candidate: Candidate = {
        "profile": {"current_title": "Engineer", "summary": "x " * 5000},
        "career_history": [],
    }
    assert len(build_embedding_text(candidate)) <= 4000


def test_embedding_text_handles_missing_fields() -> None:
    assert build_embedding_text({}) == ""
    assert build_embedding_text({"profile": {}, "career_history": []}) == ""


def test_llm_profile_excludes_skills(sample_candidate: Candidate) -> None:
    profile = build_llm_profile(sample_candidate)
    distinctive_skill = sample_candidate["skills"][0]["name"]
    assert distinctive_skill not in profile
    assert "Python" not in profile


def test_llm_profile_includes_identity_career_and_signals(sample_candidate: Candidate) -> None:
    profile = build_llm_profile(sample_candidate)
    assert "ML Engineer" in profile
    assert "7 yrs experience" in profile
    assert "Bangalore, India" in profile
    assert "ProductCo" in profile  # career company
    assert "recruiter_response_rate=0.8" in profile  # availability signal
    assert "open_to_work=True" in profile


def test_llm_profile_is_deterministic(sample_candidate: Candidate) -> None:
    assert build_llm_profile(sample_candidate) == build_llm_profile(sample_candidate)
