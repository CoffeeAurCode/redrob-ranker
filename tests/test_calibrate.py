"""Tests for the Session-08 calibration weight algebra (``eval/calibrate.py``).

The sweep / stability / ablation tables are only trustworthy if varying one weight
keeps the seven base weights summed to 1.0 (the scoring contract). These cover that
invariant directly; the data-loading and metric pieces are exercised by
``test_evaluate.py`` and ``test_metrics.py``.
"""

from __future__ import annotations

import pytest

from eval.calibrate import BASE_TERMS, base_weights, candidate_config, set_weight, with_weights
from src.config import SCORING


def _sum(cfg: object) -> float:
    return sum(float(getattr(cfg, f"w_{term}")) for term in BASE_TERMS)


def test_base_weights_match_config() -> None:
    assert base_weights(SCORING) == {term: getattr(SCORING, f"w_{term}") for term in BASE_TERMS}


def test_with_weights_renormalizes_to_one() -> None:
    doubled = {term: 2.0 * getattr(SCORING, f"w_{term}") for term in BASE_TERMS}
    cfg = with_weights(SCORING, doubled)
    assert _sum(cfg) == pytest.approx(1.0)
    # Doubling every weight is a pure rescale ⇒ the normalized config is unchanged.
    for term in BASE_TERMS:
        assert getattr(cfg, f"w_{term}") == pytest.approx(getattr(SCORING, f"w_{term}"))


def test_with_weights_rejects_nonpositive_total() -> None:
    with pytest.raises(ValueError):
        with_weights(SCORING, {term: 0.0 for term in BASE_TERMS})


def test_set_weight_sets_value_and_keeps_sum_one() -> None:
    cfg = set_weight(SCORING, "career_sim", 0.40)
    assert cfg.w_career_sim == pytest.approx(0.40)
    assert _sum(cfg) == pytest.approx(1.0)


def test_set_weight_scales_others_proportionally() -> None:
    # The other six keep their internal ratios — only career_sim's share changes.
    before = base_weights(SCORING)
    cfg = set_weight(SCORING, "career_sim", 0.40)
    ratio = cfg.w_role_match / cfg.w_domain_match
    assert ratio == pytest.approx(before["role_match"] / before["domain_match"])


def test_set_weight_zero_is_ablation() -> None:
    cfg = set_weight(SCORING, "lexical_evidence", 0.0)
    assert cfg.w_lexical_evidence == 0.0
    assert _sum(cfg) == pytest.approx(1.0)


def test_candidate_config_only_softens_availability_floor() -> None:
    cand = candidate_config(SCORING)
    assert cand.availability_floor == pytest.approx(0.70)
    # The locked config changes nothing else — the seven fit weights are untouched.
    assert base_weights(cand) == base_weights(SCORING)
