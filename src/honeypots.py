"""Deterministic honeypot detection — the testable core of Session 05 (precompute 04).

The challenge injected ~80 "subtly impossible" profiles (honeypots), forces them to
relevance tier 0 in the ground truth, and **disqualifies any submission that ranks
>10% of them into the top 100**. This module flags the ones that are impossible from
the *structured* fields alone, so ``scoring.py`` can zero them out as insurance.

Design stance (from ``docs/schema.md`` and the bundle's ``submission_spec.md``):

* **Precision over recall.** A false positive zeroes a real candidate — that is
  unrecoverable. Every rule below is a *logical* impossibility, not "looks odd."
  When in doubt we do not flag.
* **Only what the data can prove.** Two of the bundle's example honeypots are
  detectable here ("expert in N skills, 0 years used"; "8 years at a company,
  fewer years of total experience"). The other archetype it names — *tenure
  exceeding a company's age* — needs company founding dates the dataset does not
  carry, so it is left to the LLM/embedding signals (the bundle explicitly expects
  a good ranker to "naturally avoid them; you don't need to special-case them").
* **Rules that are normal noise are excluded.** ``skill.duration_months > yoe*12``
  fires on 51% of the pool and is *not* a rule; the date-impossibility checks fire
  on 0 rows of the (clean) pool and are kept only as cheap, correct insurance.

Pure and stdlib-only: no network, no model, no numpy. Safe to import anywhere —
including ``rank.py``, which actually consumes the precomputed
``honeypot_flags.json`` rather than re-deriving it. The thin CLI
``src/precompute/04_detect_honeypots.py`` streams the pool through these predicates
and writes that artifact.
"""

from __future__ import annotations

from datetime import date
from typing import TypedDict

from src.config import HONEYPOT, HoneypotConfig
from src.io_utils import Candidate, CareerEntry, Profile, Skill

# The artifact this stage produces; rank-time scoring (Session 06/10) reads it by name.
HONEYPOT_FLAGS_FILE = "honeypot_flags.json"

# Reason codes recorded per flagged candidate (sorted into the artifact for a stable,
# debuggable record — they also feed the deck's trap-avoidance example).
REASON_EXPERT_ZERO_DURATION = "expert_zero_duration"
REASON_ROLE_EXCEEDS_EXPERIENCE = "role_exceeds_experience"
REASON_IMPOSSIBLE_TIMELINE = "impossible_timeline"


class HoneypotFlag(TypedDict):
    """One flagged candidate's verdict: impossible, plus the reason codes that fired."""

    honeypot: bool
    reasons: list[str]


# --------------------------------------------------------------------------- #
# Individual rules — each a small, pure, independently-testable predicate.      #
# --------------------------------------------------------------------------- #
def has_expert_zero_duration(candidate: Candidate, cfg: HoneypotConfig = HONEYPOT) -> bool:
    """True if any skill claims ``expert`` proficiency with ``duration_months == 0``.

    "Mastery of a skill never used" is a clean impossibility. Matched on an *exact*
    zero (a missing/garbled duration does not fire), and only at the top proficiency
    level — see :class:`config.HoneypotConfig` for why "expert" alone.
    """
    for skill in _skills(candidate):
        proficiency = (skill.get("proficiency") or "").strip().lower()
        if proficiency == cfg.expert_proficiency and skill.get("duration_months") == 0:
            return True
    return False


def has_role_exceeding_experience(candidate: Candidate, cfg: HoneypotConfig = HONEYPOT) -> bool:
    """True if one role lasted far longer than the candidate's whole claimed career.

    A single ``duration_months`` exceeding ``years_of_experience * 12`` by at least
    :attr:`HoneypotConfig.role_excess_margin_months` is impossible — you cannot hold
    one job longer than your total experience. The margin keeps ordinary rounding
    and part-time spells from firing. Skipped when experience is missing or
    non-positive (nothing to contradict — conservative).
    """
    months_of_experience = _years_of_experience_months(candidate)
    if months_of_experience is None or months_of_experience <= 0:
        return False
    cutoff = months_of_experience + cfg.role_excess_margin_months
    for role in _career_history(candidate):
        duration = role.get("duration_months")
        if isinstance(duration, int) and duration >= cutoff:
            return True
    return False


def has_impossible_timeline(candidate: Candidate) -> bool:
    """True if any role has an end before its start, or a negative duration.

    Reference-free logical impossibilities. They fire on 0 rows of the provided
    (clean) pool, but are kept as cheap, correct insurance should the data change.
    """
    for role in _career_history(candidate):
        duration = role.get("duration_months")
        if isinstance(duration, int) and duration < 0:
            return True
        start = _parse_iso_date(role.get("start_date"))
        end = _parse_iso_date(role.get("end_date"))
        if start is not None and end is not None and end < start:
            return True
    return False


# --------------------------------------------------------------------------- #
# Aggregation.                                                                  #
# --------------------------------------------------------------------------- #
def honeypot_reasons(candidate: Candidate, cfg: HoneypotConfig = HONEYPOT) -> list[str]:
    """Return the sorted reason codes that fired for a candidate ('' if none)."""
    reasons: list[str] = []
    if has_expert_zero_duration(candidate, cfg):
        reasons.append(REASON_EXPERT_ZERO_DURATION)
    if has_role_exceeding_experience(candidate, cfg):
        reasons.append(REASON_ROLE_EXCEEDS_EXPERIENCE)
    if has_impossible_timeline(candidate):
        reasons.append(REASON_IMPOSSIBLE_TIMELINE)
    return sorted(reasons)


def detect_honeypot(candidate: Candidate, cfg: HoneypotConfig = HONEYPOT) -> HoneypotFlag | None:
    """Return a :class:`HoneypotFlag` if any rule fires, else ``None`` (clean profile)."""
    reasons = honeypot_reasons(candidate, cfg)
    if not reasons:
        return None
    return {"honeypot": True, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Small defensive field accessors (the data is clean, but never crash on odd    #
# shapes — the same loud-precompute / safe-rank-time contract as io_utils).     #
# --------------------------------------------------------------------------- #
def _skills(candidate: Candidate) -> list[Skill]:
    value = candidate.get("skills")
    return value if isinstance(value, list) else []


def _career_history(candidate: Candidate) -> list[CareerEntry]:
    value = candidate.get("career_history")
    return value if isinstance(value, list) else []


def _years_of_experience_months(candidate: Candidate) -> float | None:
    """Claimed total experience in months, or ``None`` if absent/non-numeric."""
    profile: Profile = candidate.get("profile") or Profile()
    years = profile.get("years_of_experience")
    if isinstance(years, bool) or not isinstance(years, (int, float)):
        return None
    return float(years) * 12.0


def _parse_iso_date(value: object) -> date | None:
    """Parse a ``YYYY-MM-DD`` string to a date, or ``None`` (null end_date, junk)."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
