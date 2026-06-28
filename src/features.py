"""Feature assembly for scoring — the Session-03 pre-filter and the Session-06 vector.

Two responsibilities, both **pure and deterministic** (no I/O, no model, no
network), so the whole module is safe to import inside the offline ``rank.py``:

* **Pre-filter** (Session 03): :func:`passes_prefilter` drops the ~68k filler
  profiles to a ~1-3k shortlist before the expensive LLM extraction stage.
* **Feature vector** (Session 06): :func:`assemble_features` turns a candidate plus
  its precomputed artifacts (embedding cosine, ``llm_signals`` record, honeypot
  flag) into a typed :class:`Features` — every term traceable to a JD requirement
  and explainable in one sentence (see each helper's docstring). ``scoring.py``
  consumes the result.

The embedding cosine is computed by the caller (one vectorized dot product against
the JD reference embedding) and passed in, so the decision logic here stays trivial
to unit-test. Every weight, map and normalization parameter lives in
:class:`config.ScoringConfig` — nothing is hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from src.config import FILTER, SCORING, FilterConfig, ScoringConfig
from src.io_utils import Candidate, Profile
from src.llm_signals import LLMSignals
from src.profile_text import build_embedding_text

# A judgment we have no evidence for is neutral, never penalizing — a parse gap or
# a missing behavioral field must not silently bury a candidate.
_NEUTRAL = 0.5


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


# --------------------------------------------------------------------------- #
# Session 06 — the per-candidate feature vector.                                #
# Every field is in [0, 1] (except the carried-through flags/honeypot), and each #
# traces to a JD requirement via the helper below it.                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Features:
    """One candidate's scoring features — the typed input to ``scoring.score``.

    The seven additive fit terms (``career_sim`` … ``lexical_evidence``) are each in
    ``[0, 1]``; ``availability`` and ``location`` are bounded multipliers;
    ``disqualifier_flags`` and ``honeypot`` are carried through for the penalty and
    the hard-zero rule. See each ``*_score`` helper for the one-sentence meaning.
    """

    candidate_id: str
    career_sim: float  # semantic fit: candidate career text ↔ JD ideal (cosine→[0,1])
    role_match: float  # how well the LLM archetype matches the target role family
    domain_match: float  # how well the LLM domain matches NLP/IR/recsys
    product_ratio: float  # product- vs services/consulting-company career (LLM)
    seniority_fit: float  # how squarely experience sits in the JD band (LLM)
    built_ranking: float  # 1.0 if they shipped a ranking/search/recsys system, else 0
    lexical_evidence: float  # fraction of retrieval/vectordb/ranking/eval evidence in career text
    availability: float  # behavioral-signal blend, squashed to [floor, 1.0] (a modifier)
    location: float  # 1.0 if India/willing-to-relocate, else a mild down-weight
    disqualifier_flags: tuple[str, ...]  # JD "do-NOT-want" flags → per-flag penalty
    honeypot: bool  # impossible profile → final score forced to 0


def assemble_features(
    candidate: Candidate,
    *,
    cosine: float,
    signals: LLMSignals | None,
    honeypot: bool,
    cfg: ScoringConfig = SCORING,
) -> Features:
    """Assemble a candidate's :class:`Features` from its precomputed artifacts.

    ``cosine`` is the raw embedding similarity to the JD reference (computed by the
    caller); ``signals`` is the candidate's ``llm_signals`` record, or ``None`` when
    the LLM stage never covered it (only the shortlist is extracted). When signals
    are absent we fall back to the semantic signal rather than crash — ``role_match``
    and ``domain_match`` default to ``career_sim``, the LLM-only judgments to neutral,
    and no disqualifier flags are carried — which floors the (by-construction
    low-cosine, non-shortlisted) candidate sensibly. Pure and deterministic.
    """
    career_sim = career_sim_from_cosine(cosine, cfg)

    if signals is None:
        role_match = career_sim
        domain_match = career_sim
        product_ratio = _NEUTRAL
        seniority_fit = _NEUTRAL
        built_ranking = 0.0
        flags: tuple[str, ...] = ()
    else:
        role_match = role_match_score(signals, cfg)
        domain_match = domain_match_score(signals, cfg)
        product_ratio = _clip01(signals["product_vs_services"])
        seniority_fit = _clip01(signals["seniority_band_fit"])
        built_ranking = 1.0 if signals["built_ranking_or_search"] else 0.0
        flags = tuple(signals["disqualifier_flags"])

    return Features(
        candidate_id=candidate.get("candidate_id") or "",
        career_sim=career_sim,
        role_match=role_match,
        domain_match=domain_match,
        product_ratio=product_ratio,
        seniority_fit=seniority_fit,
        built_ranking=built_ranking,
        lexical_evidence=lexical_evidence_score(candidate, cfg),
        availability=availability_score(candidate, cfg),
        location=location_score(candidate, cfg),
        disqualifier_flags=flags,
        honeypot=honeypot,
    )


def career_sim_from_cosine(cosine: float, cfg: ScoringConfig = SCORING) -> float:
    """Map the embedding cosine to ``[0, 1]`` by stretching this pool's discriminating band.

    A flat ``(cos+1)/2`` would squash this pool's compressed-but-high cosines into a
    near-constant ~0.8; instead we linearly rescale the ``[floor, ceiling]`` window
    where the signal actually lives (see ``ScoringConfig.cosine_floor/_ceiling``) and
    clamp, so the weight on this term stays meaningful.
    """
    span = cfg.cosine_ceiling - cfg.cosine_floor
    if span <= 0:  # degenerate config — fall back to a plain clamp.
        return _clip01(cosine)
    return _clip01((cosine - cfg.cosine_floor) / span)


def role_match_score(signals: LLMSignals, cfg: ScoringConfig = SCORING) -> float:
    """Score how well the LLM's role archetype matches the JD's target role family."""
    return cfg.role_match_scores.get(signals["role_archetype"], cfg.role_match_default)


def domain_match_score(signals: LLMSignals, cfg: ScoringConfig = SCORING) -> float:
    """Score how well the LLM's problem domain matches NLP/IR/recsys (the JD's space)."""
    return cfg.domain_match_scores.get(signals["domain"], cfg.domain_match_default)


def lexical_evidence_score(candidate: Candidate, cfg: ScoringConfig = SCORING) -> float:
    """Fraction of evidence categories whose keywords appear in the candidate's career text.

    Matched against the career FREE-TEXT (``build_embedding_text`` — title, summary,
    role descriptions), **never** the structured ``skills`` list the dataset turned
    into uniform noise: a description of building learning-to-rank evaluated with
    NDCG is genuine evidence; a "RAG" skill tag is not. The score saturates at the
    number of categories, so breadth of real evidence is rewarded without overweight.
    """
    categories = cfg.lexical_keyword_categories
    if not categories:
        return 0.0
    text = build_embedding_text(candidate).lower()
    hits = sum(1 for terms in categories.values() if any(term in text for term in terms))
    return hits / len(categories)


def location_score(candidate: Candidate, cfg: ScoringConfig = SCORING) -> float:
    """Full credit for a candidate in India or willing to relocate, else a mild down-weight.

    The role is based in India (Noida/Pune); a candidate elsewhere who will not
    relocate is a weaker logistical fit but never excluded — hence a bounded factor,
    not a zero.
    """
    profile: Profile = candidate.get("profile") or Profile()
    country = (profile.get("country") or "").strip().lower()
    if country == cfg.home_country:
        return 1.0
    signals = candidate.get("redrob_signals") or {}
    if signals.get("willing_to_relocate") is True:
        return 1.0
    return cfg.location_penalty


def availability_score(candidate: Candidate, cfg: ScoringConfig = SCORING) -> float:
    """Blend the behavioral signals into a ``[floor, 1.0]`` availability multiplier.

    A weighted average of recency, recruiter engagement, notice period and identity
    checks (see :func:`_availability_components`), squashed into ``[floor, 1.0]`` so
    even a fully-disengaged profile keeps half its fit — availability *modifies* a
    score, it never zeroes a strong candidate on its own (that is the honeypot rule).
    """
    components = _availability_components(candidate, cfg)
    weights = cfg.availability_weights
    total_weight = sum(weights.values())
    blend = (
        sum(weights.get(name, 0.0) * value for name, value in components.items()) / total_weight
        if total_weight > 0
        else 0.0
    )
    floor = cfg.availability_floor
    return floor + (1.0 - floor) * blend


# --------------------------------------------------------------------------- #
# Behavioral-signal normalizers (each → [0, 1]; nulls and -1 sentinels neutral). #
# --------------------------------------------------------------------------- #
def _availability_components(candidate: Candidate, cfg: ScoringConfig) -> dict[str, float]:
    """Normalize each behavioral field to ``[0, 1]`` (keys match ``availability_weights``)."""
    signals = candidate.get("redrob_signals") or {}
    return {
        "recency": _recency(signals.get("last_active_date"), cfg),
        "recruiter_response_rate": _rate01(signals.get("recruiter_response_rate")),
        "open_to_work": _flag_score(
            signals.get("open_to_work_flag"), true=1.0, false=cfg.open_to_work_false
        ),
        "interview_completion_rate": _rate01(signals.get("interview_completion_rate")),
        "notice_period": _notice_period(signals.get("notice_period_days"), cfg),
        "offer_acceptance": _sentinel_rate(signals.get("offer_acceptance_rate")),
        "verified_email": _flag_score(signals.get("verified_email"), true=1.0, false=0.0),
        "verified_phone": _flag_score(signals.get("verified_phone"), true=1.0, false=0.0),
    }


def _rate01(value: Any) -> float:
    """A 0.0-1.0 rate used as-is; missing/non-numeric → neutral."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _NEUTRAL
    return _clip01(float(value))


def _sentinel_rate(value: Any) -> float:
    """Like :func:`_rate01`, but the ``-1`` "no history" sentinel reads as neutral, not worst."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _NEUTRAL
    if value < 0:  # -1 = no prior offers; unknown, not a 0% acceptance rate.
        return _NEUTRAL
    return _clip01(float(value))


def _flag_score(value: Any, *, true: float, false: float) -> float:
    """A boolean signal scored ``true``/``false``; a missing flag is neutral."""
    if value is True:
        return true
    if value is False:
        return false
    return _NEUTRAL


def _notice_period(value: Any, cfg: ScoringConfig) -> float:
    """Shorter notice → more available: ``1 - days/max`` clamped; missing → neutral."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _NEUTRAL
    return _clip01(1.0 - float(value) / cfg.notice_period_max_days)


def _recency(value: Any, cfg: ScoringConfig) -> float:
    """Linear decay of last-active recency against the FIXED snapshot date; missing → neutral.

    Anchoring to a config snapshot (never ``datetime.now()``) keeps rank.py
    byte-identical on re-run; activity on/after the snapshot is fully recent (1.0)
    and older activity decays to 0 across ``recency_horizon_days``.
    """
    active = _parse_date(value)
    snapshot = _parse_date(cfg.snapshot_date)
    if active is None or snapshot is None:
        return _NEUTRAL
    days = (snapshot - active).days
    if days <= 0:
        return 1.0
    return _clip01(1.0 - days / cfg.recency_horizon_days)


def _parse_date(value: Any) -> date | None:
    """Parse a ``YYYY-MM-DD`` string to a date, or ``None`` (null/garbled)."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _clip01(value: float) -> float:
    """Clamp a float to ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))
