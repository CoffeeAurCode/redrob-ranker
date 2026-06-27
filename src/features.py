"""Feature assembly for scoring — begins here with the cheap Session-03 pre-filter.

This module grows into the full feature vector in Session 06 (``career_sim``,
``role_match``, ``domain_match``, …). For now it holds the one piece Session 03
needs and Session 10 reuses inside ``rank.py``: the deterministic pre-filter that
drops the ~68k filler profiles to a ~1-3k shortlist before the expensive LLM
extraction stage.

Everything here is **pure and deterministic** — no I/O, no model, no network — so
it is safe to import anywhere, including the offline ``rank.py``. The similarity
score is computed by the caller (one vectorized dot product against the JD
reference embedding) and passed in, which keeps the decision logic itself trivial
to unit-test.
"""

from __future__ import annotations

from src.config import FILTER, FilterConfig
from src.io_utils import Candidate, Profile


def normalize_title(title: str | None) -> str:
    """Lower-case and collapse whitespace so a title compares against the archetype set.

    The pool's titles come from a fixed vocabulary (Session-01 EDA), so this light
    normalization is enough to make archetype membership an exact-set test. A
    missing/empty title normalizes to ``""`` (never an archetype).
    """
    if not title:
        return ""
    return " ".join(title.split()).lower()


def is_archetype_title(candidate: Candidate, cfg: FilterConfig = FILTER) -> bool:
    """True if the candidate's current title is a known AI/ML/IR/DS archetype.

    See :data:`config._ARCHETYPE_TITLES` for the curated set and why CV-primary and
    broad adjacent titles are deliberately excluded.
    """
    profile: Profile = candidate.get("profile") or Profile()
    return normalize_title(profile.get("current_title")) in cfg.archetype_titles


def passes_prefilter(candidate: Candidate, sim: float, cfg: FilterConfig = FILTER) -> bool:
    """Keep a candidate for the shortlist if title OR similarity clears the bar.

    ``sim`` is the candidate's career-text cosine similarity to the JD reference
    embedding (precomputed by the caller). The OR is deliberately generous:

    * a real fit whose *title* looks unrelated (e.g. a "Backend Engineer" who
      actually built a recommender) is rescued by the similarity branch; while
    * an archetype title with weak career text still passes on title, leaving the
      precision call to the LLM stage.

    Losing a true fit here is unrecoverable, so when in doubt we keep — Session 04
    refines precision downstream.
    """
    return is_archetype_title(candidate, cfg) or sim >= cfg.similarity_threshold
