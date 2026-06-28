"""Build the system ranking from the precomputed artifacts (Session 07 shared core).

Both eval tools — :mod:`eval.label_helper` (which candidates to hand-label) and
:mod:`eval.evaluate` (how the ranking scores against those labels) — need *the same*
ranked list the scorer produces. This module is that single builder, so the two
tools can never drift from each other or from the rank-time contract.

It deliberately mirrors the path ``src/precompute/score_shortlist.py`` (and, later,
Session 10's ``rank.py``) walks — load the committed artifacts, assemble a
:class:`~src.features.Features` per shortlisted candidate, apply
:func:`~src.scoring.score`, and sort by the golden key (score descending, then
``candidate_id`` ascending). It reuses only the **pure** ``src`` helpers, so like
them it makes no network/LLM call and is safe to run offline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import SCORING, ScoringConfig
from src.embedding import load_embeddings
from src.features import Features, assemble_features
from src.honeypots import load_flagged_ids
from src.io_utils import Candidate, load_candidates
from src.jd_reference import load_jd_reference
from src.llm_signals import LLM_SIGNALS_FILE, LLMSignals, load_signal_cache
from src.scoring import ScoreResult, score

SHORTLIST_FILE = "shortlist_ids.json"


@dataclass(frozen=True)
class RankedCandidate:
    """One scored candidate at its position in the final ranking.

    ``rank`` is 1-based after the deterministic sort. ``candidate`` is the raw record
    (the labeling helper renders its summary/career history); ``signals`` is the LLM
    record or ``None``; ``features`` and ``result`` are the exact rank-time score
    inputs/outputs (the deck and ``evaluate`` quote from the breakdown).
    """

    candidate_id: str
    rank: int
    title: str
    candidate: Candidate
    signals: LLMSignals | None
    features: Features
    result: ScoreResult


def reference_similarities(artifacts_dir: Path) -> dict[str, float]:
    """Cosine of every pool candidate against the JD reference, keyed by id.

    Vectors are L2-normalized, so cosine is a plain dot product of the float16
    candidate matrix (upcast to float32) with the float32 reference vector — one
    vectorized pass over the pool, identical to the scorer's similarity step.
    """
    reference = load_jd_reference(artifacts_dir)
    ref_vec = np.asarray(reference["reference_embedding"], dtype=np.float32)
    ids, embeddings = load_embeddings(artifacts_dir, mmap=True)
    sims = np.asarray(embeddings, dtype=np.float32) @ ref_vec
    return dict(zip(ids.tolist(), sims.tolist(), strict=True))


def build_ranking(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    cfg: ScoringConfig = SCORING,
) -> list[RankedCandidate]:
    """Score every shortlisted candidate and return them in final ranked order.

    Streams the pool (never slurps it), scores only the shortlist (the set that has
    LLM signals), and sorts by score descending then ``candidate_id`` ascending — the
    exact deterministic order ``rank.py`` emits, with ``rank`` assigned 1-based.
    """
    shortlist = _load_shortlist(artifacts_dir / SHORTLIST_FILE)
    sim_by_id = reference_similarities(artifacts_dir)
    signals = load_signal_cache(artifacts_dir / LLM_SIGNALS_FILE)
    flagged = load_flagged_ids(artifacts_dir)

    shortlisted = (
        candidate
        for candidate in load_candidates(candidates_path)
        if isinstance(candidate.get("candidate_id"), str)
        and candidate.get("candidate_id") in shortlist
    )
    return rank_candidates(
        shortlisted, sim_by_id=sim_by_id, signals=signals, flagged=flagged, cfg=cfg
    )


def rank_candidates(
    candidates: Iterable[Candidate],
    *,
    sim_by_id: Mapping[str, float],
    signals: Mapping[str, LLMSignals],
    flagged: frozenset[str] | set[str],
    cfg: ScoringConfig = SCORING,
) -> list[RankedCandidate]:
    """Score an already-selected set of candidates and return them in ranked order.

    The pure scoring core shared by :func:`build_ranking` (which streams the pool to
    select the shortlist) and the Session-08 calibration harness (which holds the
    shortlist records in memory and re-scores them under many configs). Both feed the
    same assemble→score→sort path, so neither can drift from the rank-time contract:
    sort by score descending then ``candidate_id`` ascending, ``rank`` 1-based.
    """
    scored: list[RankedCandidate] = []
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str):
            continue
        signal = signals.get(cid)
        features = assemble_features(
            candidate,
            cosine=sim_by_id.get(cid, -1.0),
            signals=signal,
            honeypot=cid in flagged,
            cfg=cfg,
        )
        result = score(features, cfg)
        scored.append(
            RankedCandidate(
                candidate_id=cid,
                rank=0,  # assigned after the sort below
                title=_title(candidate),
                candidate=candidate,
                signals=signal,
                features=features,
                result=result,
            )
        )

    scored.sort(key=lambda c: (-c.result.final, c.candidate_id))
    return [_with_rank(c, rank) for rank, c in enumerate(scored, start=1)]


def _with_rank(candidate: RankedCandidate, rank: int) -> RankedCandidate:
    """Return a copy of ``candidate`` with its 1-based ``rank`` set."""
    return RankedCandidate(
        candidate_id=candidate.candidate_id,
        rank=rank,
        title=candidate.title,
        candidate=candidate.candidate,
        signals=candidate.signals,
        features=candidate.features,
        result=candidate.result,
    )


def _title(candidate: Candidate) -> str:
    profile = candidate.get("profile") or {}
    return profile.get("current_title") or "(no title)"


def _load_shortlist(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run build_shortlist.py first.")
    return set(json.loads(path.read_text(encoding="utf-8")))
