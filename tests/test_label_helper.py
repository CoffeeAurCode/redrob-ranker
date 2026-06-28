"""Tests for the gold-set labeling helper (``eval/label_helper.py``).

Two pure pieces carry the weight: :func:`select_gold_candidates` (the stratified,
deterministic sample) and :func:`suggest_tier` (the draft label heuristic). Both must
be reproducible and must cover the categories that make the gold set test a ranker.
"""

from __future__ import annotations

import pytest

from eval.label_helper import (
    GoldPick,
    select_gold_candidates,
    select_pool_honeypots,
    suggest_tier,
)
from eval.ranking import RankedCandidate
from tests.factories import make_ranked


@pytest.fixture
def ranking() -> list[RankedCandidate]:
    """A 60-candidate synthetic ranking with planted category members past rank 16."""
    ranked: list[RankedCandidate] = []
    for i in range(1, 61):
        rank = i
        if rank <= 30:  # upper/mid band: solid fits, available
            ranked.append(
                make_ranked(
                    f"CAND_{i:04d}",
                    rank,
                    career_sim=0.8,
                    role_match=0.8,
                    domain_match=0.8,
                    product_ratio=0.8,
                    seniority_fit=0.8,
                    built_ranking=0.8,
                    lexical_evidence=0.8,
                    availability=0.90,
                )
            )
        else:  # low band: weaker, slightly less available
            ranked.append(
                make_ranked(
                    f"CAND_{i:04d}",
                    rank,
                    career_sim=0.45,
                    role_match=0.45,
                    domain_match=0.45,
                    product_ratio=0.45,
                    seniority_fit=0.45,
                    built_ranking=0.45,
                    lexical_evidence=0.45,
                    availability=0.78,
                )
            )
    # Plant the targeted categories at known ranks (> clear_fit's top-16 quota).
    ranked[24] = make_ranked(
        "CAND_INACT",
        25,
        title="Senior ML Engineer",
        career_sim=0.9,
        role_match=0.9,
        domain_match=0.9,
        product_ratio=0.9,
        seniority_fit=0.9,
        built_ranking=1.0,
        lexical_evidence=0.9,
        availability=0.66,
    )
    ranked[19] = make_ranked(
        "CAND_TITLE",
        20,
        title="Computer Vision Engineer",
        role_match=1.0,
        domain_match=1.0,
        availability=0.90,
    )
    ranked[34] = make_ranked(
        "CAND_CVBAIT",
        35,
        title="AI Specialist",
        role_match=1.0,
        domain_match=1.0,
        disqualifier_flags=("cv_primary",),
        availability=0.85,
    )
    ranked[39] = make_ranked(
        "CAND_CONSULT",
        40,
        role_match=1.0,
        disqualifier_flags=("consulting_only",),
        availability=0.80,
    )
    ranked[44] = make_ranked(
        "CAND_HOPPER", 45, disqualifier_flags=("job_hopper",), availability=0.80
    )
    ranked[49] = make_ranked(
        "CAND_STALE", 50, disqualifier_flags=("stale_coding",), availability=0.80
    )
    ranked[54] = make_ranked(
        "CAND_MISMATCH", 55, career_sim=0.9, role_match=0.2, domain_match=0.1, availability=0.80
    )
    ranked[57] = make_ranked("CAND_HONEY", 58, honeypot=True, availability=0.80)
    return ranked


# --------------------------------------------------------------------------- #
# Selection.                                                                    #
# --------------------------------------------------------------------------- #
def test_selection_is_deterministic(ranking: list[RankedCandidate]) -> None:
    first = select_gold_candidates(ranking)
    second = select_gold_candidates(ranking)
    assert first == second


def test_selection_has_no_duplicate_ids(ranking: list[RankedCandidate]) -> None:
    picks = select_gold_candidates(ranking)
    ids = [p.candidate_id for p in picks]
    assert len(ids) == len(set(ids))


def test_selection_respects_max_total(ranking: list[RankedCandidate]) -> None:
    picks = select_gold_candidates(ranking, max_total=20)
    assert len(picks) <= 20


def test_selection_covers_required_categories(ranking: list[RankedCandidate]) -> None:
    by_id = {p.candidate_id: p for p in select_gold_candidates(ranking)}
    # The planted members must be selected, in their intended category.
    assert by_id["CAND_HONEY"].category == "honeypot"
    assert by_id["CAND_CVBAIT"].category == "cv_bait"
    assert by_id["CAND_CONSULT"].category == "consulting_only"
    assert by_id["CAND_INACT"].category == "inactive_but_perfect"
    assert by_id["CAND_TITLE"].category == "title_surprise"
    assert by_id["CAND_MISMATCH"].category == "semantic_mismatch"
    # Clear fits come from the very top of the ranking.
    assert any(p.category == "clear_fit" and p.rank == 1 for p in by_id.values())


def test_selection_empty_ranking() -> None:
    assert select_gold_candidates([]) == []


def test_clear_fit_takes_the_top_of_the_ranking(ranking: list[RankedCandidate]) -> None:
    picks = select_gold_candidates(ranking)
    clear = [p for p in picks if p.category == "clear_fit"]
    assert clear[0].rank == 1
    assert all(isinstance(p, GoldPick) for p in clear)


def test_pool_honeypots_are_deterministic_and_outside_shortlist() -> None:
    flagged = frozenset({"CAND_0000005", "CAND_0000001", "CAND_0000009", "CAND_0000003"})
    shortlist = {"CAND_0000009"}  # already in the ranking
    picked = select_pool_honeypots(flagged, shortlist, count=2)
    assert picked == ["CAND_0000001", "CAND_0000003"]  # lowest ids, excluding shortlisted


# --------------------------------------------------------------------------- #
# Suggested tier (a draft for the human to overwrite).                          #
# --------------------------------------------------------------------------- #
def test_suggest_tier_honeypot_is_zero() -> None:
    tier, rationale = suggest_tier(make_ranked("CAND_H", 1, honeypot=True))
    assert tier == 0
    assert "honeypot" in rationale.lower()


def test_suggest_tier_ideal_is_five() -> None:
    tier, _ = suggest_tier(make_ranked("CAND_I", 1))  # all features 1.0, no flags
    assert tier == 5


def test_suggest_tier_off_target_is_low() -> None:
    tier, _ = suggest_tier(
        make_ranked(
            "CAND_O",
            1,
            role_match=0.2,
            domain_match=0.1,
            career_sim=0.9,
            product_ratio=0.2,
            seniority_fit=0.5,
            built_ranking=0.0,
            lexical_evidence=0.2,
        )
    )
    assert tier <= 1


def test_suggest_tier_hard_flag_caps_at_two() -> None:
    # Strong fit but a CV-primary disqualifier ⇒ capped at 2.
    tier, rationale = suggest_tier(make_ranked("CAND_C", 1, disqualifier_flags=("cv_primary",)))
    assert tier == 2
    assert "cv_primary" in rationale


def test_suggest_tier_soft_flag_nudges_down() -> None:
    clean, _ = suggest_tier(make_ranked("CAND_A", 1))
    hopped, rationale = suggest_tier(make_ranked("CAND_B", 1, disqualifier_flags=("job_hopper",)))
    assert hopped == clean - 1
    assert "job_hopper" in rationale
