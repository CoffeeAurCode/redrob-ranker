"""Validate a redrob-ranker submission CSV against the format contract.

This is a faithful, *to-spec* validator. The official challenge validator was not
available at scaffolding time (Session 00), so this enforces the documented CSV
invariants directly; swap in the official one if/when obtained (see PROGRESS.md).

Contract enforced:
  * exactly 100 data rows, header ``rank,candidate_id,score,reasoning``
  * ranks are 1..100, unique, ascending and contiguous
  * score is non-increasing as rank increases
  * ties (equal score) are broken by candidate_id ascending
  * reasoning is non-empty for every row

Usage:
    python validate_submission.py submission.csv

Exit code 0 = valid; 1 = invalid (errors printed to stderr).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

EXPECTED_HEADER: list[str] = ["rank", "candidate_id", "score", "reasoning"]
EXPECTED_ROWS: int = 100


@dataclass(frozen=True)
class Row:
    """One parsed submission row (line number kept for error messages)."""

    line: int
    rank: int
    candidate_id: str
    score: float
    reasoning: str


def _id_sort_key(candidate_id: str) -> tuple[int, object]:
    """Order ids numerically when they are all integers, else lexicographically.

    Returns a 2-tuple whose first element groups ints (0) before strings (1) so
    the key is internally consistent regardless of which branch a value takes.
    """
    try:
        return (0, int(candidate_id))
    except ValueError:
        return (1, candidate_id)


def parse_rows(path: Path) -> tuple[list[Row], list[str]]:
    """Read and type-parse the CSV. Returns (rows, structural_errors)."""
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], ["file is empty (no header row)"]

        if header != EXPECTED_HEADER:
            errors.append(f"header must be {EXPECTED_HEADER}, got {header}")
            return [], errors

        rows: list[Row] = []
        for line_no, record in enumerate(reader, start=2):
            if len(record) != len(EXPECTED_HEADER):
                errors.append(f"line {line_no}: expected 4 fields, got {len(record)}")
                continue
            rank_s, cid, score_s, reasoning = record
            try:
                rank = int(rank_s)
            except ValueError:
                errors.append(f"line {line_no}: rank {rank_s!r} is not an integer")
                continue
            try:
                score = float(score_s)
            except ValueError:
                errors.append(f"line {line_no}: score {score_s!r} is not a number")
                continue
            rows.append(Row(line_no, rank, cid, score, reasoning))
    return rows, errors


def check_invariants(rows: list[Row]) -> list[str]:
    """Validate the semantic contract over already-parsed rows."""
    errors: list[str] = []

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} data rows, got {len(rows)}")

    ranks = [r.rank for r in rows]
    if ranks != list(range(1, len(rows) + 1)):
        errors.append("ranks must be 1..N, unique, ascending and contiguous")

    for prev, curr in pairwise(rows):
        if curr.score > prev.score:
            errors.append(
                f"line {curr.line}: score {curr.score} > previous {prev.score} "
                "(scores must be non-increasing)"
            )
        elif curr.score == prev.score and _id_sort_key(curr.candidate_id) < _id_sort_key(
            prev.candidate_id
        ):
            errors.append(
                f"line {curr.line}: tie at score {curr.score} but candidate_id "
                f"{curr.candidate_id!r} < previous {prev.candidate_id!r} "
                "(ties must break by candidate_id ascending)"
            )

    for r in rows:
        if not r.reasoning.strip():
            errors.append(f"line {r.line}: reasoning is empty")

    return errors


def validate(path: Path) -> list[str]:
    """Return a list of human-readable errors; empty list means valid."""
    if not path.is_file():
        return [f"file not found: {path}"]
    rows, structural = parse_rows(path)
    return structural + check_invariants(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="path to the submission CSV")
    args = parser.parse_args(argv)

    errors = validate(args.csv_path)
    if errors:
        print(f"INVALID: {args.csv_path} ({len(errors)} error(s))", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"VALID: {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
