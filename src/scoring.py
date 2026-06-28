"""The transparent, linear candidate score — the Session-06 contract, exactly.

``rank.py`` (Session 10) calls :func:`score` once per candidate to turn a
:class:`features.Features` vector into a scalar plus a full term-by-term breakdown.
The model is deliberately a readable weighted sum — **no learned LTR model**:
there is no public leaderboard to climb, only a few days and a gold set, so
transparency and a defensible "explain this line" story (Stage 5) beat opaque
tuning.

The formula (``plan/00_OVERVIEW.md``)::

    base_fit  = w1·career_sim + w2·role_match + w3·domain_match
              + w4·product_ratio + w5·seniority_fit + w6·built_ranking
              + w7·lexical_evidence
    penalties = Σ penalty[flag] for flag in disqualifier_flags
    final     = max(0, base_fit - penalties) · availability · location
    final     = 0.0 if honeypot

Golden rule: like ``features.py`` this imports **no** network/LLM/embedding
library — only the config and the typed feature vector — so it is safe inside the
offline ``rank.py`` and trivial to unit-test. Every weight and penalty lives in
:class:`config.ScoringConfig`; nothing numeric is hard-coded here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.config import SCORING, ScoringConfig
from src.features import Features


@dataclass(frozen=True)
class ScoreResult:
    """A candidate's final score plus the breakdown every later stage needs.

    ``final`` is the scalar ``rank.py`` sorts on; the remaining fields are the
    breakdown that Session 09's reasoning, the deck, and the Stage-5 defense quote
    from. ``contributions`` holds the seven **weighted** base terms and sums to
    ``base_fit``; ``penalty_detail`` maps each fired disqualifier flag to the points
    it subtracted.
    """

    final: float
    base_fit: float
    penalties: float
    availability: float
    location: float
    honeypot: bool
    contributions: Mapping[str, float]
    penalty_detail: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view of the breakdown (for reasoning/deck artifacts)."""
        return {
            "final": self.final,
            "base_fit": self.base_fit,
            "penalties": self.penalties,
            "availability": self.availability,
            "location": self.location,
            "honeypot": self.honeypot,
            "contributions": dict(self.contributions),
            "penalty_detail": dict(self.penalty_detail),
        }


def score(features: Features, cfg: ScoringConfig = SCORING) -> ScoreResult:
    """Compute the transparent weighted score and its breakdown for one candidate.

    Pure and deterministic. A honeypot is forced to ``final = 0`` regardless of any
    other term; otherwise the disqualifier penalties subtract from ``base_fit``
    (floored at 0) before the availability and location multipliers apply.
    """
    contributions: dict[str, float] = {
        "career_sim": cfg.w_career_sim * features.career_sim,
        "role_match": cfg.w_role_match * features.role_match,
        "domain_match": cfg.w_domain_match * features.domain_match,
        "product_ratio": cfg.w_product_ratio * features.product_ratio,
        "seniority_fit": cfg.w_seniority_fit * features.seniority_fit,
        "built_ranking": cfg.w_built_ranking * features.built_ranking,
        "lexical_evidence": cfg.w_lexical_evidence * features.lexical_evidence,
    }
    base_fit = sum(contributions.values())

    # Keyed by flag so a (defensively) duplicated flag is counted once.
    penalty_detail: dict[str, float] = {
        flag: cfg.penalty_per_flag.get(flag, 0.0) for flag in features.disqualifier_flags
    }
    penalties = sum(penalty_detail.values())

    if features.honeypot:
        final = 0.0
    else:
        net_fit = max(0.0, base_fit - penalties)
        final = net_fit * features.availability * features.location

    return ScoreResult(
        final=final,
        base_fit=base_fit,
        penalties=penalties,
        availability=features.availability,
        location=features.location,
        honeypot=features.honeypot,
        contributions=contributions,
        penalty_detail=penalty_detail,
    )
