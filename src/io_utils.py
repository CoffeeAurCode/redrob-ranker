"""Shared I/O helpers: streaming candidate loading and defensive JSON parsing.

These are the single source of truth for *reading* data across the project. The
precompute scripts (Sessions 02-05) and the rank-time runtime (Session 10) all
import from here rather than re-implementing JSONL/JSON handling.

Two design rules from ``plan/CONVENTIONS.md`` apply:

* **Loud in precompute, safe at rank time.** ``load_candidates`` raises on a
  malformed row by default (precompute wants to know); pass ``skip_errors=True``
  for the defensive rank-time path that must still produce a valid CSV.
* **Stream, don't slurp.** The pool is ~100k rows / ~465 MB. ``load_candidates``
  yields one record at a time so callers never need the whole file in RAM.

The TypedDicts mirror the real ``docs/challenge/candidate_schema.json`` (verified
against the data in Session 01 — note identity fields live under ``profile`` and
behavioral fields under ``redrob_signals``, *not* at the top level).
"""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict, cast

logger = logging.getLogger(__name__)


def use_utf8_stdout() -> None:
    """Force UTF-8 stdout so precompute CLIs can print non-ASCII safely.

    Windows consoles default to cp1252, which raises ``UnicodeEncodeError`` on the
    em-dashes/arrows in the JD text and the ``✓`` marks in the sanity reports. This
    is a no-op where ``stdout`` isn't a regular text stream (e.g. captured under
    pytest). rank.py never calls this — it is a precompute/CLI convenience only.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Typed candidate view (mirrors candidate_schema.json).                        #
# total=False throughout: every field is treated as possibly-absent so that    #
# downstream code accesses defensively (the data is clean, but rank.py must     #
# never crash on a surprise null).                                              #
# --------------------------------------------------------------------------- #
class Skill(TypedDict, total=False):
    """One entry of the (noise) ``skills`` list. Excluded from profile text."""

    name: str
    proficiency: str  # beginner | intermediate | advanced | expert
    endorsements: int
    duration_months: int


class CareerEntry(TypedDict, total=False):
    """One role in ``career_history`` — the primary free-text signal."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


class Profile(TypedDict, total=False):
    """Top-level identity block (nested under ``profile`` in each record)."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


class Candidate(TypedDict, total=False):
    """A full candidate record as loaded from ``candidates.jsonl``."""

    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry]
    education: list[dict[str, Any]]
    skills: list[Skill]
    certifications: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    redrob_signals: dict[str, Any]


class CandidateLoadError(ValueError):
    """A line in the JSONL pool could not be parsed as a JSON object."""

    def __init__(self, path: Path, lineno: int, reason: str) -> None:
        super().__init__(f"{path}:{lineno}: invalid JSONL row ({reason})")
        self.path = path
        self.lineno = lineno


class JSONParseError(ValueError):
    """No valid JSON value could be recovered from a (possibly noisy) string."""


# --------------------------------------------------------------------------- #
# Streaming JSONL loader.                                                       #
# --------------------------------------------------------------------------- #
def load_candidates(
    path: str | Path,
    *,
    skip_errors: bool = False,
) -> Iterator[Candidate]:
    """Stream candidate records from a JSONL file, one dict per line.

    Yields each record lazily so the 100k pool never has to be held in memory.
    Blank lines are ignored. By default a malformed line raises
    ``CandidateLoadError`` (precompute should fail loudly); set
    ``skip_errors=True`` to log-and-skip instead (the defensive rank-time path).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_errors:
                    logger.warning("%s:%d: skipping malformed row (%s)", path, lineno, exc)
                    continue
                raise CandidateLoadError(path, lineno, str(exc)) from exc
            yield cast(Candidate, record)


def count_candidates(path: str | Path) -> int:
    """Count non-blank rows in a JSONL pool without parsing each line."""
    path = Path(path)
    total = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                total += 1
    return total


# --------------------------------------------------------------------------- #
# Defensive JSON parsing for LLM output.                                        #
# Reused by the LLM sessions (03 extraction, 05 reasoning): models often wrap   #
# JSON in ```code fences``` or surround it with prose. This recovers the value. #
# --------------------------------------------------------------------------- #
def parse_json_safe(text: str) -> Any:
    """Parse JSON from a possibly noisy LLM response.

    Strips Markdown code fences and any surrounding prose, then parses. If a
    direct parse fails, recovers the first balanced ``{...}`` or ``[...]`` block.
    Raises ``JSONParseError`` if nothing parseable is found.
    """
    if not isinstance(text, str):  # defensive: callers sometimes pass None
        raise JSONParseError(f"expected str, got {type(text).__name__}")

    cleaned = _strip_code_fences(text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    snippet = _extract_first_json(cleaned)
    if snippet is not None:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    preview = text.strip()[:200]
    raise JSONParseError(f"no parseable JSON found in: {preview!r}")


def _strip_code_fences(text: str) -> str:
    """Return the contents of the first ```...``` fence, or the text unchanged.

    Handles ```json fences and bare ``` fences. If no closing fence is present,
    the opening fence line is dropped so a direct parse can still succeed.
    """
    lines = text.splitlines()
    fence_idxs = [i for i, line in enumerate(lines) if line.lstrip().startswith("```")]
    if not fence_idxs:
        return text
    start = fence_idxs[0]
    end = fence_idxs[1] if len(fence_idxs) >= 2 else len(lines)
    return "\n".join(lines[start + 1 : end])


def _extract_first_json(text: str) -> str | None:
    """Slice out the first complete JSON object/array, ignoring trailing prose.

    Scans for the first ``{`` or ``[`` and returns the substring up to its
    matching close brace, respecting string literals and escapes so braces
    inside strings don't throw off the depth count.
    """
    start = _first_index(text, "{", "[")
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _first_index(text: str, *chars: str) -> int | None:
    """Index of the earliest occurrence of any of ``chars``, or None."""
    found = [text.find(char) for char in chars]
    candidates = [i for i in found if i != -1]
    return min(candidates) if candidates else None
