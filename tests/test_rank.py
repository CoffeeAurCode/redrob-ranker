"""Tests for the offline ranking entrypoint (``src/rank.py``).

This is the most disqualification-sensitive file in the project, so the tests are a
gate, not a formality:

* **CSV invariants** — exactly N rows, ranks 1..N unique, score non-increasing,
  ties broken by ``candidate_id`` ascending, reasoning never empty.
* **Determinism** — the same inputs produce identical rows and byte-identical CSV.
* **Offline import guard** — a subprocess proves ``src.rank``'s import closure pulls
  in no network/LLM library (the golden rule, mechanically enforced).
* **Defensive behavior** — honeypots are zeroed, an unseen id degrades without
  crashing, an uncached top candidate still gets a grounded reasoning, and too few
  survivors fail loudly rather than emit an invalid CSV.

The core (:func:`src.rank.select_ranked`) takes already-loaded artifact maps, so it
is tested directly without touching the 100k pool or the embedding model.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from src.io_utils import Candidate
from src.llm_signals import LLMSignals
from src.rank import (
    SUBMISSION_HEADER,
    SubmissionRow,
    format_score,
    select_ranked,
    write_submission,
)
from src.reasoning import ReasoningRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Builders.                                                                      #
# --------------------------------------------------------------------------- #
def make_candidate(cid: str, *, title: str = "ML Engineer", years: float = 6.0) -> Candidate:
    """A minimal archetype-titled candidate (passes the pre-filter on title alone)."""
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "summary": "Built retrieval and ranking systems in production.",
            "country": "India",
            "years_of_experience": years,
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": title,
                "duration_months": int(years * 12),
                "description": "Shipped a learning-to-rank search system evaluated with NDCG.",
            }
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "interview_completion_rate": 0.9,
            "notice_period_days": 30,
        },
    }


def make_signals(cid: str, *, archetype: str = "ml_engineer") -> LLMSignals:
    """A strong, unflagged LLM signal record for ``cid``."""
    return {
        "candidate_id": cid,
        "role_archetype": archetype,
        "domain": "nlp_ir",
        "product_vs_services": 1.0,
        "seniority_band_fit": 1.0,
        "built_ranking_or_search": True,
        "evidence_span": "Shipped a learning-to-rank search system evaluated with NDCG.",
        "disqualifier_flags": [],
    }


def assert_submission_valid(rows: list[SubmissionRow], expected_n: int) -> None:
    """Assert the rows satisfy every invariant the official validator enforces."""
    assert len(rows) == expected_n
    assert [r.rank for r in rows] == list(range(1, expected_n + 1))
    assert len({r.candidate_id for r in rows}) == expected_n  # ranks + ids unique

    values = [float(r.score) for r in rows]
    for upper, lower in pairwise(values):
        assert upper >= lower, "score must be non-increasing by rank"

    for upper_row, lower_row in pairwise(rows):
        if float(upper_row.score) == float(lower_row.score):
            assert (
                upper_row.candidate_id < lower_row.candidate_id
            ), "ties must break by id ascending"

    for row in rows:
        assert row.reasoning.strip(), "every row must carry a non-empty reasoning"


# --------------------------------------------------------------------------- #
# CSV invariants.                                                               #
# --------------------------------------------------------------------------- #
def test_select_ranked_satisfies_csv_invariants() -> None:
    candidates = [make_candidate(f"CAND_{i:07d}", years=4.0 + i * 0.5) for i in range(1, 13)]
    signals = {c["candidate_id"]: make_signals(c["candidate_id"]) for c in candidates[::2]}
    rows = select_ranked(
        candidates,
        sim_by_id={c["candidate_id"]: 0.70 + 0.01 * i for i, c in enumerate(candidates)},
        signals=signals,
        flagged=frozenset(),
        reasoning_cache={},
        top_n=10,
    )
    assert_submission_valid(rows, 10)


def test_ties_break_by_candidate_id_ascending() -> None:
    # Two candidates with identical inputs ⇒ identical score ⇒ a genuine tie.
    a = make_candidate("CAND_0000200")
    b = make_candidate("CAND_0000100")
    sig = {cid: make_signals(cid) for cid in ("CAND_0000200", "CAND_0000100")}
    rows = select_ranked(
        [a, b],
        sim_by_id={"CAND_0000200": 0.75, "CAND_0000100": 0.75},
        signals=sig,
        flagged=frozenset(),
        reasoning_cache={},
        top_n=2,
    )
    assert float(rows[0].score) == float(rows[1].score)
    assert [r.candidate_id for r in rows] == ["CAND_0000100", "CAND_0000200"]  # lower id first


# --------------------------------------------------------------------------- #
# Determinism.                                                                  #
# --------------------------------------------------------------------------- #
def test_select_ranked_is_deterministic() -> None:
    def build() -> list[SubmissionRow]:
        candidates = [make_candidate(f"CAND_{i:07d}", years=4.0 + i * 0.3) for i in range(1, 11)]
        return select_ranked(
            candidates,
            sim_by_id={c["candidate_id"]: 0.72 + 0.005 * i for i, c in enumerate(candidates)},
            signals={c["candidate_id"]: make_signals(c["candidate_id"]) for c in candidates},
            flagged=frozenset(),
            reasoning_cache={},
            top_n=5,
        )

    assert build() == build()


def test_written_csv_is_byte_identical_and_official_header(tmp_path: Path) -> None:
    rows = [
        SubmissionRow("CAND_0000002", 1, "0.900000", "Top-tier match, reason one."),
        SubmissionRow("CAND_0000001", 2, "0.900000", "Strong match, reason, two."),  # comma quoted
    ]
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    write_submission(rows, first)
    write_submission(rows, second)

    assert first.read_bytes() == second.read_bytes()
    lines = first.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "candidate_id,rank,score,reasoning"
    assert tuple(SUBMISSION_HEADER) == ("candidate_id", "rank", "score", "reasoning")
    assert first.read_bytes().endswith(b"\n")  # trailing newline, LF only
    assert b"\r" not in first.read_bytes()  # no CRLF even on Windows


def test_format_score_is_fixed_precision() -> None:
    assert format_score(0.5) == "0.500000"
    assert format_score(1.0) == "1.000000"
    assert format_score(0.0) == "0.000000"
    # Sorting on the parsed-back value matches the written string (validator-safe).
    assert float(format_score(0.123456789)) == float("0.123457")


# --------------------------------------------------------------------------- #
# Defensive behavior.                                                           #
# --------------------------------------------------------------------------- #
def test_honeypot_is_zeroed_and_not_in_top() -> None:
    good = [make_candidate(f"CAND_{i:07d}") for i in range(1, 4)]
    trap = make_candidate("CAND_0009999")
    pool = [*good, trap]
    rows = select_ranked(
        pool,
        sim_by_id={c["candidate_id"]: 0.78 for c in pool},
        signals={c["candidate_id"]: make_signals(c["candidate_id"]) for c in pool},
        flagged=frozenset({"CAND_0009999"}),
        reasoning_cache={},
        top_n=3,
    )
    assert "CAND_0009999" not in {r.candidate_id for r in rows}


def test_unseen_id_degrades_without_crashing() -> None:
    seen = make_candidate("CAND_0000001")
    # Not in sim_by_id and NOT an archetype title ⇒ dropped, no crash.
    unseen = make_candidate("CAND_0000002", title="Mechanical Engineer")
    rows = select_ranked(
        [seen, unseen] + [make_candidate(f"CAND_{i:07d}") for i in range(3, 6)],
        sim_by_id={
            "CAND_0000001": 0.78,
            "CAND_0000003": 0.78,
            "CAND_0000004": 0.78,
            "CAND_0000005": 0.78,
        },
        signals={},
        flagged=frozenset(),
        reasoning_cache={},
        top_n=4,
    )
    assert "CAND_0000002" not in {r.candidate_id for r in rows}
    assert_submission_valid(rows, 4)


def test_duplicate_input_ids_are_deduplicated() -> None:
    dup = make_candidate("CAND_0000001")
    candidates = [dup, dup] + [make_candidate(f"CAND_{i:07d}") for i in range(2, 5)]
    rows = select_ranked(
        candidates,
        sim_by_id={c["candidate_id"]: 0.78 for c in candidates},
        signals={},
        flagged=frozenset(),
        reasoning_cache={},
        top_n=4,
    )
    assert len({r.candidate_id for r in rows}) == 4


def test_too_few_survivors_fails_loudly() -> None:
    candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 4)]
    with pytest.raises(SystemExit):
        select_ranked(
            candidates,
            sim_by_id={c["candidate_id"]: 0.78 for c in candidates},
            signals={},
            flagged=frozenset(),
            reasoning_cache={},
            top_n=100,
        )


# --------------------------------------------------------------------------- #
# Reasoning join.                                                               #
# --------------------------------------------------------------------------- #
def test_cached_reasoning_is_used_and_fallback_fills_gaps() -> None:
    cached = make_candidate("CAND_0000001")
    uncached = make_candidate("CAND_0000002")
    reasoning_cache: dict[str, ReasoningRecord] = {
        "CAND_0000001": {
            "candidate_id": "CAND_0000001",
            "reasoning": "Cached grounded reasoning for this candidate.",
            "source": "llm",
        }
    }
    rows = select_ranked(
        [cached, uncached],
        sim_by_id={"CAND_0000001": 0.80, "CAND_0000002": 0.78},
        signals={c["candidate_id"]: make_signals(c["candidate_id"]) for c in (cached, uncached)},
        flagged=frozenset(),
        reasoning_cache=reasoning_cache,
        top_n=2,
    )
    by_id = {r.candidate_id: r.reasoning for r in rows}
    assert by_id["CAND_0000001"] == "Cached grounded reasoning for this candidate."
    assert by_id["CAND_0000002"].strip()  # deterministic fallback, never empty
    assert by_id["CAND_0000002"] != by_id["CAND_0000001"]


# --------------------------------------------------------------------------- #
# The golden rule, mechanically enforced.                                       #
# --------------------------------------------------------------------------- #
def test_rank_import_graph_is_offline() -> None:
    """Importing ``src.rank`` must not pull in any network/LLM library."""
    forbidden = [
        "requests",
        "httpx",
        "httpcore",
        "openai",
        "google.generativeai",
        "google.genai",
        "ollama",
        "sentence_transformers",
        "torch",
        "grpc",
    ]
    code = (
        "import sys, importlib\n"
        "importlib.import_module('src.rank')\n"
        f"bad = [m for m in {forbidden!r} if m in sys.modules]\n"
        "print(';'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"forbidden modules imported by src.rank: {result.stdout.strip()!r} "
        f"(stderr: {result.stderr.strip()!r})"
    )
