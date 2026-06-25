"""The single source of truth for *what text represents a candidate*.

Every later stage that turns a candidate into text — embeddings (Session 02),
LLM extraction (Session 04), reasoning (Session 09) — goes through here, so the
"career history is signal, skills are noise" decision is made in exactly one
place and can be defended line-by-line at Stage 5.

Why skills are excluded (see ``docs/challenge/job_description.md``): the dataset
deliberately makes the ``skills`` list near-uniform noise — every AI keyword
appears on ~12% of profiles regardless of real fit. The JD's "right answer"
reasons about *career history*, not keyword presence. So the embedding/LLM text
is built from **current title + summary + career-history descriptions** and the
structured ``skills`` list is never concatenated in. (Skills still feed honeypot
detection in Session 05 — that is handled separately from this text.)

Both builders are pure and deterministic: same candidate in, same string out.
"""

from __future__ import annotations

from src.io_utils import Candidate, CareerEntry, Profile

# Whitespace/length knobs. Kept here (not scattered as literals) per CONVENTIONS.
# Embedding models (BGE/E5/GTE) truncate at ~512 tokens anyway; the char cap is a
# memory/time guard, set well above a typical built text (title + summary + 3
# career descriptions ≈ 1.7k chars).
_MAX_EMBEDDING_CHARS = 4000
# How many most-recent roles to include in the compact LLM profile. The pool
# caps career_history at 10; the recent few carry the relevant signal.
_MAX_LLM_CAREER_ENTRIES = 6

# Behavioral fields surfaced to the LLM as availability context (the JD treats
# these as a hireability signal). Each pair is (prompt label, real schema key in
# redrob_signals). Ordered for a stable, readable prompt.
_LLM_SIGNAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("last_active", "last_active_date"),
    ("open_to_work", "open_to_work_flag"),
    ("recruiter_response_rate", "recruiter_response_rate"),
    ("interview_completion_rate", "interview_completion_rate"),
    ("notice_period_days", "notice_period_days"),
    ("willing_to_relocate", "willing_to_relocate"),
)


def build_embedding_text(candidate: Candidate) -> str:
    """Build the text embedded to represent a candidate (skills excluded).

    Concatenates, in a stable order: current title, professional summary, then
    each career-history role's description. Whitespace is normalized and the
    result is capped at ``_MAX_EMBEDDING_CHARS``. Deterministic and pure.
    """
    profile = _profile(candidate)
    parts: list[str] = []

    title = _clean(profile.get("current_title"))
    if title:
        parts.append(title)

    summary = _clean(profile.get("summary"))
    if summary:
        parts.append(summary)

    for entry in _career(candidate):
        description = _clean(entry.get("description"))
        if description:
            parts.append(description)

    text = " ".join(parts)
    return text[:_MAX_EMBEDDING_CHARS].strip()


def build_llm_profile(candidate: Candidate) -> str:
    """Render a compact, labeled profile for LLM prompts (skills excluded).

    Includes identity (title, years, location), the professional summary, the
    most-recent roles with company/title/duration/description, and the
    availability-relevant behavioral signals. The ``skills`` list is omitted so
    the fit judgment cannot be gamed by keyword stuffing; honeypot/skills checks
    live in Session 05. Deterministic and pure.
    """
    profile = _profile(candidate)
    lines: list[str] = []

    title = _clean(profile.get("current_title"))
    years = profile.get("years_of_experience")
    location = _clean(profile.get("location"))
    country = _clean(profile.get("country"))

    header = title or "(no title)"
    if isinstance(years, (int, float)):
        header += f" · {years:g} yrs experience"
    where = ", ".join(part for part in (location, country) if part)
    if where:
        header += f" · {where}"
    lines.append(f"Current role: {header}")

    summary = _clean(profile.get("summary"))
    if summary:
        lines.append(f"Summary: {summary}")

    history = _career(candidate)[:_MAX_LLM_CAREER_ENTRIES]
    if history:
        lines.append("Career history:")
        for entry in history:
            lines.append(_format_career_entry(entry))

    signal_line = _format_signals(candidate)
    if signal_line:
        lines.append(f"Availability signals: {signal_line}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internal helpers.                                                            #
# --------------------------------------------------------------------------- #
def _format_career_entry(entry: CareerEntry) -> str:
    """One bullet: '- <Title> @ <Company> (<n> mo): <description>'."""
    role = _clean(entry.get("title")) or "(role)"
    company = _clean(entry.get("company")) or "(company)"
    months = entry.get("duration_months")
    tenure = f" ({months} mo)" if isinstance(months, int) else ""
    description = _clean(entry.get("description"))
    body = f": {description}" if description else ""
    return f"- {role} @ {company}{tenure}{body}"


def _format_signals(candidate: Candidate) -> str:
    """'key=value' list of availability signals present on the candidate."""
    signals = candidate.get("redrob_signals") or {}
    rendered: list[str] = []
    for label, key in _LLM_SIGNAL_FIELDS:
        if key in signals and signals[key] is not None:
            rendered.append(f"{label}={signals[key]}")
    return ", ".join(rendered)


def _profile(candidate: Candidate) -> Profile:
    """The ``profile`` block, or an empty mapping if absent."""
    return candidate.get("profile") or Profile()


def _career(candidate: Candidate) -> list[CareerEntry]:
    """The ``career_history`` list, or an empty list if absent."""
    return candidate.get("career_history") or []


def _clean(value: object) -> str:
    """Normalize a value to a single-spaced, stripped string ('' if None)."""
    if value is None:
        return ""
    return " ".join(str(value).split())
