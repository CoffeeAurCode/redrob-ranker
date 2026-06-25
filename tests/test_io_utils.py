"""Tests for io_utils — streaming loader and defensive JSON parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from src.io_utils import (
    CandidateLoadError,
    JSONParseError,
    count_candidates,
    load_candidates,
    parse_json_safe,
)


# --------------------------------------------------------------------------- #
# load_candidates / count_candidates                                          #
# --------------------------------------------------------------------------- #
def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, object]], *, blank_lines: bool = False
) -> None:
    lines = [json.dumps(row) for row in rows]
    if blank_lines:
        lines = ["", *lines, "   ", ""]  # leading/trailing/interior blanks
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_candidates_yields_right_count(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [{"candidate_id": f"CAND_000000{i}"} for i in range(5)]
    _write_jsonl(pool, rows, blank_lines=True)

    loaded = list(load_candidates(pool))
    assert len(loaded) == 5
    assert [c["candidate_id"] for c in loaded] == [r["candidate_id"] for r in rows]


def test_load_candidates_streams_lazily(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    _write_jsonl(pool, [{"candidate_id": "CAND_0000001"}, {"candidate_id": "CAND_0000002"}])
    iterator = load_candidates(pool)
    first = next(iterator)
    assert first["candidate_id"] == "CAND_0000001"


def test_load_candidates_raises_on_malformed_row(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"candidate_id": "CAND_0000001"}\n{not valid json}\n', encoding="utf-8")
    with pytest.raises(CandidateLoadError) as exc_info:
        list(load_candidates(pool))
    assert exc_info.value.lineno == 2


def test_load_candidates_skip_errors(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        '{"candidate_id": "CAND_0000001"}\n{bad}\n{"candidate_id": "CAND_0000002"}\n',
        encoding="utf-8",
    )
    loaded = list(load_candidates(pool, skip_errors=True))
    assert [c["candidate_id"] for c in loaded] == ["CAND_0000001", "CAND_0000002"]


def test_count_candidates_ignores_blank_lines(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    _write_jsonl(pool, [{"a": 1}, {"a": 2}, {"a": 3}], blank_lines=True)
    assert count_candidates(pool) == 3


# --------------------------------------------------------------------------- #
# parse_json_safe                                                             #
# --------------------------------------------------------------------------- #
def test_parse_plain_object() -> None:
    assert parse_json_safe('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_parse_fenced_json() -> None:
    text = '```json\n{"role": "ml_engineer", "score": 0.9}\n```'
    assert parse_json_safe(text) == {"role": "ml_engineer", "score": 0.9}


def test_parse_bare_fenced_json() -> None:
    text = "```\n[1, 2, 3]\n```"
    assert parse_json_safe(text) == [1, 2, 3]


def test_parse_json_array() -> None:
    assert parse_json_safe('[{"id": 1}, {"id": 2}]') == [{"id": 1}, {"id": 2}]


def test_parse_with_leading_and_trailing_prose() -> None:
    text = (
        'Sure! Here is the JSON you asked for:\n{"verdict": "fit"}\nLet me know if you need more.'
    )
    assert parse_json_safe(text) == {"verdict": "fit"}


def test_parse_ignores_braces_inside_strings() -> None:
    text = 'prefix {"note": "contains } and { braces", "ok": true} suffix'
    assert parse_json_safe(text) == {"note": "contains } and { braces", "ok": True}


def test_parse_raises_on_garbage() -> None:
    with pytest.raises(JSONParseError):
        parse_json_safe("there is no json here at all")


def test_parse_raises_on_non_string() -> None:
    with pytest.raises(JSONParseError):
        parse_json_safe(None)  # type: ignore[arg-type]
