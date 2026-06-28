"""Tests for the Session-03 cheap pre-filter (``src/features.py``).

The pre-filter is the one irreversible gate before the expensive LLM stage, so its
two branches are pinned down here:

* a strong fit survives — whether rescued by an archetype **title** or by a high
  **similarity** score on an unrelated-looking title; and
* obvious filler (an Accountant) is dropped **regardless of its skills list**,
  because the filter never reads skills (that is the whole point — see the JD's
  keyword trap).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from src.config import FILTER, SCORING
from src.features import (
    assemble_features,
    availability_score,
    career_sim_from_cosine,
    is_archetype_title,
    lexical_evidence_score,
    location_score,
    normalize_title,
    passes_prefilter,
)
from src.io_utils import Candidate
from src.llm_signals import LLMSignals
from src.precompute.build_shortlist import SHORTLIST_FILE

T = FILTER.similarity_threshold
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _signals(**overrides: object) -> LLMSignals:
    """A complete, valid LLMSignals record; override individual fields per test."""
    base: LLMSignals = {
        "candidate_id": "CAND_0000042",
        "role_archetype": "recsys_search",
        "domain": "nlp_ir",
        "product_vs_services": 1.0,
        "seniority_band_fit": 1.0,
        "built_ranking_or_search": True,
        "evidence_span": "built search ranking",
        "disqualifier_flags": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# --------------------------------------------------------------------------- #
# normalize_title                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ML Engineer", "ml engineer"),
        ("  Senior   Data   Scientist ", "senior data scientist"),
        ("AI ENGINEER", "ai engineer"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_title(raw: str | None, expected: str) -> None:
    assert normalize_title(raw) == expected


# --------------------------------------------------------------------------- #
# is_archetype_title — the curated set, plus the deliberate exclusions          #
# --------------------------------------------------------------------------- #
def _with_title(candidate: Candidate, title: str) -> Candidate:
    clone = copy.deepcopy(candidate)
    clone["profile"]["current_title"] = title
    return clone


def test_archetype_matches_core_ai_titles(sample_candidate: Candidate) -> None:
    for title in ("ML Engineer", "Data Scientist", "Recommendation Systems Engineer"):
        assert is_archetype_title(_with_title(sample_candidate, title)), title


def test_cv_bait_and_adjacent_titles_are_not_archetypes(sample_candidate: Candidate) -> None:
    # CV-primary is the JD's explicit bait; broad adjacent titles must earn their
    # way in via similarity, not via the title branch.
    for title in ("Computer Vision Engineer", "Backend Engineer", "Accountant", "HR Manager"):
        assert not is_archetype_title(_with_title(sample_candidate, title)), title


def test_archetype_handles_missing_profile() -> None:
    assert is_archetype_title({"candidate_id": "CAND_0000001"}) is False


# --------------------------------------------------------------------------- #
# passes_prefilter — the OR of the two branches                                 #
# --------------------------------------------------------------------------- #
def test_archetype_title_passes_even_with_zero_similarity(sample_candidate: Candidate) -> None:
    # sample_candidate is an "ML Engineer" → kept on title alone.
    assert passes_prefilter(sample_candidate, sim=0.0)


def test_high_similarity_rescues_a_non_archetype_title(sample_candidate: Candidate) -> None:
    backend = _with_title(sample_candidate, "Backend Engineer")
    assert not is_archetype_title(backend)
    assert passes_prefilter(backend, sim=T + 0.01)  # rescued by the similarity branch
    assert not passes_prefilter(backend, sim=T - 0.01)


def test_threshold_is_inclusive(sample_candidate: Candidate) -> None:
    backend = _with_title(sample_candidate, "Backend Engineer")
    assert passes_prefilter(backend, sim=T)  # sim >= T, boundary included


def _accountant_with_ai_skills() -> Candidate:
    """A filler profile whose *skills* are stuffed with AI keywords (the trap)."""
    return {
        "candidate_id": "CAND_0099999",
        "profile": {
            "current_title": "Accountant",
            "summary": "Chartered accountant focused on tax and audit.",
            "years_of_experience": 8.0,
            "location": "Mumbai",
            "country": "India",
        },
        "career_history": [
            {"title": "Accountant", "company": "LedgerCo", "description": "Managed ledgers."}
        ],
        "skills": [
            {"name": "Machine Learning", "proficiency": "expert"},
            {"name": "RAG", "proficiency": "expert"},
            {"name": "Pinecone", "proficiency": "advanced"},
        ],
    }


def test_filler_dropped_regardless_of_skills() -> None:
    accountant = _accountant_with_ai_skills()
    # Below threshold and not an archetype title → dropped, AI skills notwithstanding.
    assert not passes_prefilter(accountant, sim=T - 0.2)


def test_filler_only_survives_if_its_career_text_actually_matches() -> None:
    # The escape hatch is genuine semantic similarity, never the skills list.
    accountant = _accountant_with_ai_skills()
    assert passes_prefilter(accountant, sim=T + 0.05)


# --------------------------------------------------------------------------- #
# Real artifact (Definition of Done) — only after build_shortlist.py has run.   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (ARTIFACTS_DIR / SHORTLIST_FILE).exists(),
    reason="run src/precompute/build_shortlist.py first",
)
def test_real_shortlist_is_sane() -> None:
    ids = json.loads((ARTIFACTS_DIR / SHORTLIST_FILE).read_text(encoding="utf-8"))
    assert isinstance(ids, list)
    assert len(ids) == len(set(ids))  # unique
    assert ids == sorted(ids)  # deterministic on-disk order (candidate_id asc)
    assert all(re.fullmatch(r"CAND_\d{7}", cid) for cid in ids)
    assert 1000 <= len(ids) <= 3000  # the plan's ~1-3k shortlist target


# =========================================================================== #
# Session 06 — feature assembly.                                               #
# =========================================================================== #
# --------------------------------------------------------------------------- #
# career_sim — the cosine→[0,1] window rescale                                  #
# --------------------------------------------------------------------------- #
def test_career_sim_maps_the_window_to_unit_interval() -> None:
    floor, ceiling = SCORING.cosine_floor, SCORING.cosine_ceiling
    assert career_sim_from_cosine(floor) == pytest.approx(0.0)
    assert career_sim_from_cosine(ceiling) == pytest.approx(1.0)
    assert career_sim_from_cosine((floor + ceiling) / 2) == pytest.approx(0.5)


def test_career_sim_clamps_outside_the_window() -> None:
    assert career_sim_from_cosine(SCORING.cosine_floor - 0.2) == 0.0
    assert career_sim_from_cosine(SCORING.cosine_ceiling + 0.2) == 1.0


# --------------------------------------------------------------------------- #
# assemble_features — happy path and the missing-signals fallback              #
# --------------------------------------------------------------------------- #
def test_assemble_features_reads_every_signal(sample_candidate: Candidate) -> None:
    f = assemble_features(sample_candidate, cosine=0.78, signals=_signals(), honeypot=False)
    assert f.candidate_id == "CAND_0000042"
    assert f.role_match == SCORING.role_match_scores["recsys_search"]  # 1.0
    assert f.domain_match == SCORING.domain_match_scores["nlp_ir"]  # 1.0
    assert f.product_ratio == 1.0
    assert f.seniority_fit == 1.0
    assert f.built_ranking == 1.0
    assert f.career_sim == pytest.approx((0.78 - 0.60) / (0.80 - 0.60))  # 0.9
    assert f.location == 1.0  # India
    assert SCORING.availability_floor <= f.availability <= 1.0
    assert f.disqualifier_flags == ()
    assert f.honeypot is False


def test_assemble_features_carries_flags_and_honeypot(sample_candidate: Candidate) -> None:
    f = assemble_features(
        sample_candidate,
        cosine=0.70,
        signals=_signals(disqualifier_flags=["cv_primary", "job_hopper"]),
        honeypot=True,
    )
    assert f.disqualifier_flags == ("cv_primary", "job_hopper")
    assert f.honeypot is True


def test_missing_signals_fall_back_to_career_sim(sample_candidate: Candidate) -> None:
    # The gotcha: a candidate the LLM never covered must not crash — role/domain
    # fall back to career_sim and the LLM-only judgments to neutral.
    f = assemble_features(sample_candidate, cosine=0.70, signals=None, honeypot=False)
    assert f.career_sim == pytest.approx(0.5)
    assert f.role_match == f.career_sim
    assert f.domain_match == f.career_sim
    assert f.product_ratio == 0.5
    assert f.seniority_fit == 0.5
    assert f.built_ranking == 0.0
    assert f.disqualifier_flags == ()


def test_built_ranking_is_zero_when_false(sample_candidate: Candidate) -> None:
    f = assemble_features(
        sample_candidate,
        cosine=0.70,
        signals=_signals(built_ranking_or_search=False),
        honeypot=False,
    )
    assert f.built_ranking == 0.0


def test_unknown_archetype_uses_the_conservative_default(sample_candidate: Candidate) -> None:
    # llm_signals coerces unknowns to the generic bucket, but the map .get default
    # is the belt-and-braces guard if an off-vocab label ever slips through.
    f = assemble_features(
        sample_candidate,
        cosine=0.70,
        signals=_signals(role_archetype="quantum_alchemist"),
        honeypot=False,
    )
    assert f.role_match == SCORING.role_match_default


# --------------------------------------------------------------------------- #
# lexical_evidence — career text only, never the skills list                    #
# --------------------------------------------------------------------------- #
def test_lexical_evidence_counts_career_text_categories(sample_candidate: Candidate) -> None:
    # sample career text mentions retrieval + ranking/recommendation + NDCG/eval
    # (3 of the 4 categories), and never a vector-db product name.
    score = lexical_evidence_score(sample_candidate)
    assert score == pytest.approx(3 / 4)


def test_lexical_evidence_ignores_the_skills_list() -> None:
    # An Accountant whose *skills* are stuffed with retrieval/vectordb keywords but
    # whose career *text* says nothing of the sort scores zero — the trap is dodged.
    accountant: Candidate = {
        "candidate_id": "CAND_0099999",
        "profile": {"current_title": "Accountant", "summary": "Tax and audit.", "country": "India"},
        "career_history": [{"title": "Accountant", "description": "Managed ledgers and filings."}],
        "skills": [
            {"name": "Pinecone", "proficiency": "expert"},
            {"name": "RAG retrieval ranking NDCG", "proficiency": "expert"},
        ],
    }
    assert lexical_evidence_score(accountant) == 0.0


# --------------------------------------------------------------------------- #
# location — India or willing-to-relocate is full credit                        #
# --------------------------------------------------------------------------- #
def test_location_full_credit_for_india(sample_candidate: Candidate) -> None:
    assert location_score(sample_candidate) == 1.0  # country == India


def test_location_relocation_rescues_non_india(sample_candidate: Candidate) -> None:
    abroad = copy.deepcopy(sample_candidate)
    abroad["profile"]["country"] = "United States"
    abroad["redrob_signals"]["willing_to_relocate"] = True
    assert location_score(abroad) == 1.0


def test_location_down_weights_non_india_no_relocate(sample_candidate: Candidate) -> None:
    abroad = copy.deepcopy(sample_candidate)
    abroad["profile"]["country"] = "United States"
    abroad["redrob_signals"]["willing_to_relocate"] = False
    assert location_score(abroad) == SCORING.location_penalty


# --------------------------------------------------------------------------- #
# availability — bounded multiplier, neutral nulls, -1 sentinel not "worst"      #
# --------------------------------------------------------------------------- #
def test_availability_stays_within_floor_and_one(sample_candidate: Candidate) -> None:
    assert SCORING.availability_floor <= availability_score(sample_candidate) <= 1.0


def test_availability_neutral_when_all_signals_missing() -> None:
    bare: Candidate = {"candidate_id": "CAND_0000001", "redrob_signals": {}}
    # Every component falls back to neutral 0.5 → blend 0.5 → floor + 0.5·(1-floor).
    floor = SCORING.availability_floor
    assert availability_score(bare) == pytest.approx(floor + 0.5 * (1.0 - floor))


def test_perfectly_engaged_beats_fully_disengaged(sample_candidate: Candidate) -> None:
    engaged = copy.deepcopy(sample_candidate)
    engaged["redrob_signals"] = {
        "last_active_date": SCORING.snapshot_date,
        "open_to_work_flag": True,
        "recruiter_response_rate": 1.0,
        "interview_completion_rate": 1.0,
        "notice_period_days": 0,
        "offer_acceptance_rate": 1.0,
        "verified_email": True,
        "verified_phone": True,
        "willing_to_relocate": True,
    }
    disengaged = copy.deepcopy(sample_candidate)
    disengaged["redrob_signals"] = {
        "last_active_date": "2000-01-01",
        "open_to_work_flag": False,
        "recruiter_response_rate": 0.0,
        "interview_completion_rate": 0.0,
        "notice_period_days": 180,
        "offer_acceptance_rate": 0.0,
        "verified_email": False,
        "verified_phone": False,
    }
    assert availability_score(engaged) == pytest.approx(1.0)
    assert availability_score(disengaged) < availability_score(engaged)
    assert availability_score(disengaged) >= SCORING.availability_floor


def test_offer_acceptance_sentinel_is_neutral_not_worst(sample_candidate: Candidate) -> None:
    # -1 means "no prior offers", not a 0% acceptance rate — so it must score higher
    # than an actual 0.0 (the schema.md sentinel gotcha).
    no_history = copy.deepcopy(sample_candidate)
    no_history["redrob_signals"]["offer_acceptance_rate"] = -1
    zero_rate = copy.deepcopy(sample_candidate)
    zero_rate["redrob_signals"]["offer_acceptance_rate"] = 0.0
    assert availability_score(no_history) > availability_score(zero_rate)
