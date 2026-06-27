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

from src.config import FILTER
from src.features import is_archetype_title, normalize_title, passes_prefilter
from src.io_utils import Candidate
from src.precompute.build_shortlist import SHORTLIST_FILE

T = FILTER.similarity_threshold
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


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
