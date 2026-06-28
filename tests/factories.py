"""Builders for synthetic :class:`eval.ranking.RankedCandidate` objects in tests.

The eval tools operate on the ranked-candidate list; these factories let the eval
tests construct that list directly (with controlled features) instead of going
through the artifacts, so the selection/metric/report logic is tested in isolation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from eval.ranking import RankedCandidate
from src.features import Features
from src.io_utils import Candidate
from src.llm_signals import LLMSignals
from src.scoring import score


def make_features(cid: str = "CAND_TEST", **overrides: Any) -> Features:
    """An on-target, unflagged feature vector (base_fit == 1.0); override per test."""
    base = Features(
        candidate_id=cid,
        career_sim=1.0,
        role_match=1.0,
        domain_match=1.0,
        product_ratio=1.0,
        seniority_fit=1.0,
        built_ranking=1.0,
        lexical_evidence=1.0,
        availability=0.90,
        location=1.0,
        disqualifier_flags=(),
        honeypot=False,
    )
    return dataclasses.replace(base, **overrides)


def make_ranked(
    cid: str,
    rank: int,
    *,
    title: str = "ML Engineer",
    summary: str = "",
    country: str = "India",
    years: float = 6.0,
    signals: LLMSignals | None = None,
    **feature_overrides: Any,
) -> RankedCandidate:
    """A :class:`RankedCandidate` at ``rank`` whose features come from ``make_features``."""
    features = make_features(cid, **feature_overrides)
    result = score(features)
    candidate: Candidate = {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "summary": summary,
            "country": country,
            "years_of_experience": years,
        },
        "career_history": [],
    }
    return RankedCandidate(cid, rank, title, candidate, signals, features, result)
