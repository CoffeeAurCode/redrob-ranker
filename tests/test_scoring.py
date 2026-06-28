"""Tests for the transparent weighted score (``src/scoring.py``).

Scoring is where correctness is subtle, so the invariants the design rests on are
pinned down here: the breakdown must reconstruct the scalar, raising any positive
feature must never lower the score (monotonicity), a honeypot must zero everything,
penalties must subtract exactly and floor at 0, and availability/location must act
as the bounded multipliers they are. A separate block guards the config maps
against drift from the ``llm_signals`` controlled vocabulary.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.config import SCORING
from src.features import Features
from src.llm_signals import DISQUALIFIER_FLAGS, DOMAINS, ROLE_ARCHETYPES
from src.scoring import score

# The seven additive base terms, in the order they contribute to base_fit.
BASE_TERMS = (
    "career_sim",
    "role_match",
    "domain_match",
    "product_ratio",
    "seniority_fit",
    "built_ranking",
    "lexical_evidence",
)


def make_features(**overrides: Any) -> Features:
    """A middling, valid feature vector; override individual fields per test."""
    base = Features(
        candidate_id="CAND_0000001",
        career_sim=0.5,
        role_match=0.5,
        domain_match=0.5,
        product_ratio=0.5,
        seniority_fit=0.5,
        built_ranking=0.5,
        lexical_evidence=0.5,
        availability=0.8,
        location=1.0,
        disqualifier_flags=(),
        honeypot=False,
    )
    return dataclasses.replace(base, **overrides)


# --------------------------------------------------------------------------- #
# Breakdown structure reconstructs the scalar.                                  #
# --------------------------------------------------------------------------- #
def test_contributions_sum_to_base_fit() -> None:
    r = score(make_features())
    assert sum(r.contributions.values()) == pytest.approx(r.base_fit)
    assert set(r.contributions) == set(BASE_TERMS)


def test_final_reconstructs_from_the_breakdown() -> None:
    f = make_features(disqualifier_flags=("job_hopper",))
    r = score(f)
    expected = max(0.0, r.base_fit - r.penalties) * f.availability * f.location
    assert r.final == pytest.approx(expected)


def test_all_max_features_give_base_fit_one() -> None:
    # The seven weights sum to 1, so an all-1.0 feature vector yields base_fit == 1.
    r = score(make_features(**{term: 1.0 for term in BASE_TERMS}))
    assert r.base_fit == pytest.approx(1.0)


def test_as_dict_is_serializable_and_complete() -> None:
    d = score(make_features(disqualifier_flags=("cv_primary",))).as_dict()
    assert set(d) == {
        "final",
        "base_fit",
        "penalties",
        "availability",
        "location",
        "honeypot",
        "contributions",
        "penalty_detail",
    }
    assert d["penalty_detail"] == {"cv_primary": SCORING.penalty_per_flag["cv_primary"]}


# --------------------------------------------------------------------------- #
# Honeypot → final == 0 regardless of everything else.                          #
# --------------------------------------------------------------------------- #
def test_honeypot_zeroes_a_perfect_candidate() -> None:
    perfect = make_features(
        **{term: 1.0 for term in BASE_TERMS}, availability=1.0, location=1.0, honeypot=True
    )
    r = score(perfect)
    assert r.final == 0.0
    assert r.base_fit == pytest.approx(1.0)  # breakdown still computed for the deck


# --------------------------------------------------------------------------- #
# Monotonicity — raising any positive feature never lowers the final.           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term", BASE_TERMS)
def test_raising_a_base_feature_does_not_lower_final(term: str) -> None:
    base = make_features(**{term: 0.4})  # no flags → net_fit positive, so strictly up
    change: dict[str, Any] = {term: 0.7}
    raised = dataclasses.replace(base, **change)
    assert score(raised).final > score(base).final


@pytest.mark.parametrize("term", BASE_TERMS)
def test_each_weight_is_non_negative(term: str) -> None:
    assert getattr(SCORING, f"w_{term}") >= 0.0


def test_raising_availability_or_location_does_not_lower_final() -> None:
    base = make_features(availability=0.6, location=0.85)
    assert score(make_features(availability=0.9, location=0.85)).final > score(base).final
    assert score(make_features(availability=0.6, location=1.0)).final > score(base).final


# --------------------------------------------------------------------------- #
# Penalties — subtract exactly, stack, dedup, and floor the net fit at 0.        #
# --------------------------------------------------------------------------- #
def test_single_penalty_subtracts_the_configured_amount() -> None:
    flag = "consulting_only"
    r = score(make_features(disqualifier_flags=(flag,)))
    assert r.penalties == pytest.approx(SCORING.penalty_per_flag[flag])
    assert r.penalty_detail == {flag: SCORING.penalty_per_flag[flag]}


def test_multiple_penalties_stack() -> None:
    flags = ("cv_primary", "job_hopper")
    r = score(make_features(disqualifier_flags=flags))
    expected = sum(SCORING.penalty_per_flag[f] for f in flags)
    assert r.penalties == pytest.approx(expected)


def test_duplicate_flag_is_counted_once() -> None:
    r = score(make_features(disqualifier_flags=("job_hopper", "job_hopper")))
    assert r.penalties == pytest.approx(SCORING.penalty_per_flag["job_hopper"])


def test_unknown_flag_adds_no_penalty() -> None:
    r = score(make_features(disqualifier_flags=("not_a_real_flag",)))
    assert r.penalties == 0.0


def test_penalties_floor_net_fit_at_zero() -> None:
    # base_fit is small, penalties large → max(0, base-pen) clamps, final is 0 (not negative).
    weak = make_features(**{term: 0.1 for term in BASE_TERMS})
    r = score(dataclasses.replace(weak, disqualifier_flags=("consulting_only", "cv_primary")))
    assert r.base_fit < r.penalties
    assert r.final == 0.0


# --------------------------------------------------------------------------- #
# Availability / location are multipliers in the right ranges.                  #
# --------------------------------------------------------------------------- #
def test_availability_scales_the_final_linearly() -> None:
    a = score(make_features(availability=0.4))
    b = score(make_features(availability=0.8))
    assert b.final == pytest.approx(2.0 * a.final)


def test_location_scales_the_final_linearly() -> None:
    full = score(make_features(location=1.0))
    down = score(make_features(location=0.5))
    assert down.final == pytest.approx(0.5 * full.final)


# --------------------------------------------------------------------------- #
# Config maps cover the controlled vocabulary exactly (drift guard).            #
# --------------------------------------------------------------------------- #
def test_role_map_covers_the_archetype_vocabulary() -> None:
    assert set(SCORING.role_match_scores) == set(ROLE_ARCHETYPES)


def test_domain_map_covers_the_domain_vocabulary() -> None:
    assert set(SCORING.domain_match_scores) == set(DOMAINS)


def test_penalty_map_covers_the_flag_vocabulary() -> None:
    assert set(SCORING.penalty_per_flag) == set(DISQUALIFIER_FLAGS)


def test_match_scores_are_within_unit_interval() -> None:
    for value in (*SCORING.role_match_scores.values(), *SCORING.domain_match_scores.values()):
        assert 0.0 <= value <= 1.0


def test_term_weights_sum_to_one() -> None:
    total = sum(getattr(SCORING, f"w_{term}") for term in BASE_TERMS)
    assert total == pytest.approx(1.0)
