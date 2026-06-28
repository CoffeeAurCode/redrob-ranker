"""Tests for deterministic honeypot detection (``src/honeypots.py``).

This is the only signal that *zeroes* a candidate, so a false positive is the
expensive failure. Each rule therefore gets both a **trigger** fixture (the
impossibility fires) and a **near-miss** fixture (the ordinary-but-messy case must
NOT fire), plus the boundary where the threshold flips. A realistic clean profile
must come back unflagged, and odd/missing fields must never crash.
"""

from __future__ import annotations

import copy

import pytest

from src.config import HONEYPOT
from src.honeypots import (
    REASON_EXPERT_ZERO_DURATION,
    REASON_IMPOSSIBLE_TIMELINE,
    REASON_ROLE_EXCEEDS_EXPERIENCE,
    detect_honeypot,
    has_expert_zero_duration,
    has_impossible_timeline,
    has_role_exceeding_experience,
    honeypot_reasons,
)
from src.io_utils import Candidate

MARGIN = HONEYPOT.role_excess_margin_months


# --------------------------------------------------------------------------- #
# Rule 1 — expert proficiency with zero months of use.                          #
# --------------------------------------------------------------------------- #
def test_expert_zero_duration_triggers(sample_candidate: Candidate) -> None:
    c = copy.deepcopy(sample_candidate)
    c["skills"] = [{"name": "RAG", "proficiency": "expert", "duration_months": 0}]
    assert has_expert_zero_duration(c)


def test_expert_with_real_usage_is_not_flagged(sample_candidate: Candidate) -> None:
    # Near-miss: genuine expertise that was actually used for years.
    c = copy.deepcopy(sample_candidate)
    c["skills"] = [{"name": "Python", "proficiency": "expert", "duration_months": 84}]
    assert not has_expert_zero_duration(c)


def test_lower_proficiency_at_zero_months_is_not_flagged(sample_candidate: Candidate) -> None:
    # Only "expert" is impossible-on-its-face; a beginner who hasn't logged time is
    # ordinary, not a honeypot (precision over recall).
    c = copy.deepcopy(sample_candidate)
    c["skills"] = [
        {"name": "Rust", "proficiency": "advanced", "duration_months": 0},
        {"name": "Go", "proficiency": "beginner", "duration_months": 0},
    ]
    assert not has_expert_zero_duration(c)


def test_expert_with_missing_duration_is_not_flagged(sample_candidate: Candidate) -> None:
    # A missing duration is not an explicit zero — do not infer impossibility.
    c = copy.deepcopy(sample_candidate)
    c["skills"] = [{"name": "RAG", "proficiency": "expert"}]
    assert not has_expert_zero_duration(c)


# --------------------------------------------------------------------------- #
# Rule 2 — a single role longer than the whole claimed career.                  #
# --------------------------------------------------------------------------- #
def _one_role(years_of_experience: float, duration_months: int) -> Candidate:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"years_of_experience": years_of_experience, "current_title": "Engineer"},
        "career_history": [
            {"company": "X", "title": "Engineer", "duration_months": duration_months}
        ],
        "skills": [],
    }


def test_role_exceeding_experience_triggers() -> None:
    # 9 years claimed, one role of 166 months (~13.8 years) — impossible.
    assert has_role_exceeding_experience(_one_role(9.0, 166))


def test_role_within_experience_is_not_flagged() -> None:
    # A long-but-plausible 7-year role for an 8-year career is ordinary.
    assert not has_role_exceeding_experience(_one_role(8.0, 84))


def test_role_excess_boundary_is_inclusive() -> None:
    # cutoff = yoe*12 + MARGIN. Exactly at the cutoff fires; one month under does not.
    months = 60  # yoe = 5 years
    cand = _one_role(5.0, months + MARGIN)
    assert has_role_exceeding_experience(cand)
    cand_under = _one_role(5.0, months + MARGIN - 1)
    assert not has_role_exceeding_experience(cand_under)


def test_role_rule_skips_missing_or_zero_experience() -> None:
    # Nothing to contradict when experience is absent or non-positive → no flag.
    no_yoe: Candidate = {
        "candidate_id": "CAND_0000002",
        "profile": {"current_title": "Engineer"},
        "career_history": [{"duration_months": 120}],
    }
    assert not has_role_exceeding_experience(no_yoe)
    assert not has_role_exceeding_experience(_one_role(0.0, 120))


# --------------------------------------------------------------------------- #
# Rule 3 — impossible timelines (insurance; fires on 0 rows of the real pool).  #
# --------------------------------------------------------------------------- #
def test_end_before_start_triggers() -> None:
    cand: Candidate = {
        "candidate_id": "CAND_0000003",
        "career_history": [{"start_date": "2020-01-01", "end_date": "2019-01-01"}],
    }
    assert has_impossible_timeline(cand)


def test_negative_duration_triggers() -> None:
    cand: Candidate = {
        "candidate_id": "CAND_0000004",
        "career_history": [{"duration_months": -5}],
    }
    assert has_impossible_timeline(cand)


def test_current_role_with_null_end_is_not_flagged() -> None:
    # A null end_date means "current" — not an impossible timeline.
    cand: Candidate = {
        "candidate_id": "CAND_0000005",
        "career_history": [{"start_date": "2021-01-01", "end_date": None, "duration_months": 40}],
    }
    assert not has_impossible_timeline(cand)


# --------------------------------------------------------------------------- #
# Aggregation, determinism, and robustness.                                    #
# --------------------------------------------------------------------------- #
def test_clean_realistic_profile_is_not_flagged(sample_candidate: Candidate) -> None:
    # The shared fixture is a strong, ordinary ML engineer — must come back clean.
    assert honeypot_reasons(sample_candidate) == []
    assert detect_honeypot(sample_candidate) is None


def test_reasons_are_sorted_and_deduplicated() -> None:
    # A profile that trips two rules reports both, sorted (deterministic artifact).
    cand: Candidate = {
        "candidate_id": "CAND_0000006",
        "profile": {"years_of_experience": 3.0},
        "career_history": [{"duration_months": 999}],
        "skills": [{"name": "RAG", "proficiency": "expert", "duration_months": 0}],
    }
    reasons = honeypot_reasons(cand)
    assert reasons == sorted([REASON_EXPERT_ZERO_DURATION, REASON_ROLE_EXCEEDS_EXPERIENCE])
    flag = detect_honeypot(cand)
    assert flag == {"honeypot": True, "reasons": reasons}


def test_missing_and_odd_fields_do_not_crash() -> None:
    # Empty record, wrong-typed containers, and an empty skills/career list.
    for cand in (
        {},
        {"candidate_id": "CAND_0000007"},
        {"candidate_id": "CAND_0000008", "skills": None, "career_history": None, "profile": None},
        {"candidate_id": "CAND_0000009", "skills": "oops", "career_history": 5},
    ):
        assert honeypot_reasons(cand) == []  # type: ignore[arg-type]
        assert detect_honeypot(cand) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", [REASON_IMPOSSIBLE_TIMELINE])
def test_single_reason_surfaces_through_aggregate(reason: str) -> None:
    cand: Candidate = {
        "candidate_id": "CAND_0000010",
        "career_history": [{"start_date": "2020-06-01", "end_date": "2020-01-01"}],
    }
    assert honeypot_reasons(cand) == [reason]
