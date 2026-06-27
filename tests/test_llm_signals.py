"""Tests for the LLM-extraction logic (``src/llm_signals.py``).

Two layers, mirroring ``test_jd_reference.py``:

* **Pure logic** — defensive parsing (fenced/array/prose, clamping, enum/flag
  coercion, dropped-id handling), the crash-safe JSONL cache, and the batched
  ``extract_signals`` loop driven by a *fake* client (so the cache/fallback
  behaviour is proven without a network).
* The digit-prefixed CLI owns only the real Gemini call, so it is not imported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from pathlib import Path

import pytest

from src.io_utils import parse_json_safe
from src.llm_signals import (
    DISQUALIFIER_FLAGS,
    LLMSignals,
    append_signals,
    build_batch_prompt,
    build_rubric_summary,
    chunk,
    coerce_signal,
    extract_signals,
    load_signal_cache,
    parse_signals_batch,
    pending_ids,
)


def _record(cid: str, **overrides: object) -> dict[str, object]:
    """A well-formed raw LLM object for one candidate, with optional field overrides."""
    base: dict[str, object] = {
        "candidate_id": cid,
        "role_archetype": "ml_engineer",
        "domain": "nlp_ir",
        "product_vs_services": 0.9,
        "seniority_band_fit": 0.8,
        "built_ranking_or_search": True,
        "evidence_span": "Built a hybrid retrieval and ranking system in production.",
        "disqualifier_flags": [],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Prompt                                                                        #
# --------------------------------------------------------------------------- #
def test_rubric_summary_is_compact_and_deterministic() -> None:
    rubric = {
        "role_archetypes": ["Applied ML Engineer", "Search Engineer"],
        "must_haves": ["embeddings retrieval in production"],
        "hard_disqualifiers": ["pure research, no production"],
        "domains": ["Information Retrieval", "Ranking"],
        "seniority_band": {"min_years": 5, "max_years": 9, "notes": "ideal 6-8"},
    }
    summary = build_rubric_summary(rubric)  # type: ignore[arg-type]
    assert "Applied ML Engineer" in summary
    assert "embeddings retrieval in production" in summary
    assert "pure research" in summary
    assert "5-9 years" in summary
    assert summary == build_rubric_summary(rubric)  # type: ignore[arg-type]  # pure


def test_batch_prompt_embeds_ids_enums_and_profiles() -> None:
    prompt = build_batch_prompt(
        "RUBRIC", [("CAND_0001", "profile one"), ("CAND_0002", "profile two")]
    )
    assert "candidate_id: CAND_0001" in prompt and "candidate_id: CAND_0002" in prompt
    assert "profile one" in prompt and "profile two" in prompt
    # Controlled vocabularies are spelled out so the model stays in-vocabulary.
    assert "ml_engineer" in prompt and "nlp_ir" in prompt and "consulting_only" in prompt
    assert "[1]" in prompt and "[2]" in prompt


# --------------------------------------------------------------------------- #
# parse_signals_batch / coerce_signal — happy path + defensiveness              #
# --------------------------------------------------------------------------- #
def test_parses_clean_array() -> None:
    payload = [_record("CAND_0001"), _record("CAND_0002", role_archetype="data_scientist")]
    out = parse_signals_batch(payload, ["CAND_0001", "CAND_0002"])
    assert set(out) == {"CAND_0001", "CAND_0002"}
    assert out["CAND_0002"]["role_archetype"] == "data_scientist"


def test_parses_fenced_and_prose_wrapped_response() -> None:
    raw = (
        "Sure, here are the results:\n```json\n"
        + json.dumps([_record("CAND_0001")])
        + "\n```\nLet me know if you need more."
    )
    out = parse_signals_batch(parse_json_safe(raw), ["CAND_0001"])
    assert out["CAND_0001"]["candidate_id"] == "CAND_0001"


def test_unwraps_single_key_object() -> None:
    payload = {"candidates": [_record("CAND_0001")]}
    out = parse_signals_batch(payload, ["CAND_0001"])
    assert "CAND_0001" in out


def test_clamps_out_of_range_numbers() -> None:
    out = parse_signals_batch(
        [_record("CAND_0001", product_vs_services=1.7, seniority_band_fit=-0.4)], ["CAND_0001"]
    )
    assert out["CAND_0001"]["product_vs_services"] == 1.0
    assert out["CAND_0001"]["seniority_band_fit"] == 0.0


def test_accepts_string_numbers_and_defaults_garbage_to_neutral() -> None:
    out = parse_signals_batch(
        [_record("CAND_0001", product_vs_services="0.7", seniority_band_fit="n/a")], ["CAND_0001"]
    )
    assert out["CAND_0001"]["product_vs_services"] == pytest.approx(0.7)
    assert out["CAND_0001"]["seniority_band_fit"] == 0.5  # neutral fallback


def test_coerces_unknown_enums_to_generic_bucket() -> None:
    out = parse_signals_batch(
        [_record("CAND_0001", role_archetype="wizard", domain="alchemy")], ["CAND_0001"]
    )
    assert out["CAND_0001"]["role_archetype"] == "swe_generic"
    assert out["CAND_0001"]["domain"] == "generic_swe"


def test_normalizes_enum_spelling_variants() -> None:
    out = parse_signals_batch(
        [_record("CAND_0001", role_archetype="ML-Engineer", domain="NLP/IR")], ["CAND_0001"]
    )
    assert out["CAND_0001"]["role_archetype"] == "ml_engineer"
    assert out["CAND_0001"]["domain"] == "nlp_ir"


def test_drops_unknown_flags_and_sorts_dedupes_known_ones() -> None:
    out = parse_signals_batch(
        [
            _record(
                "CAND_0001",
                disqualifier_flags=["job_hopper", "made_up", "cv_primary", "job_hopper"],
            )
        ],
        ["CAND_0001"],
    )
    assert out["CAND_0001"]["disqualifier_flags"] == ["cv_primary", "job_hopper"]
    assert all(f in DISQUALIFIER_FLAGS for f in out["CAND_0001"]["disqualifier_flags"])


@pytest.mark.parametrize(
    "raw,expected",
    [(True, True), ("true", True), (1, True), ("no", False), (0, False), (False, False)],
)
def test_coerces_boolean(raw: object, expected: bool) -> None:
    out = parse_signals_batch([_record("CAND_0001", built_ranking_or_search=raw)], ["CAND_0001"])
    assert out["CAND_0001"]["built_ranking_or_search"] is expected


def test_truncates_long_evidence() -> None:
    out = parse_signals_batch([_record("CAND_0001", evidence_span="x " * 500)], ["CAND_0001"])
    assert len(out["CAND_0001"]["evidence_span"]) <= 400


def test_drops_record_without_candidate_id() -> None:
    assert coerce_signal(_record("CAND_0001", candidate_id="")) is None
    assert coerce_signal({"role_archetype": "ml_engineer"}) is None
    assert coerce_signal("not a dict") is None


def test_ignores_ids_outside_expected_set() -> None:
    payload = [_record("CAND_0001"), _record("CAND_HALLUCINATED")]
    out = parse_signals_batch(payload, ["CAND_0001"])
    assert set(out) == {"CAND_0001"}


def test_single_candidate_binds_to_our_id() -> None:
    # On a one-candidate call we trust the id we asked about over the model's echo.
    out = parse_signals_batch([_record("CAND_WRONG_ECHO")], ["CAND_0001"])
    assert set(out) == {"CAND_0001"}
    assert out["CAND_0001"]["candidate_id"] == "CAND_0001"


# --------------------------------------------------------------------------- #
# Cache: append / load / pending / chunk                                        #
# --------------------------------------------------------------------------- #
def test_append_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "llm_signals.jsonl"
    records: list[LLMSignals] = [
        coerce_signal(_record("CAND_0002")),  # type: ignore[list-item]
        coerce_signal(_record("CAND_0001")),  # type: ignore[list-item]
    ]
    append_signals(path, records)
    append_signals(path, [coerce_signal(_record("CAND_0003"))])  # type: ignore[list-item]
    cache = load_signal_cache(path)
    assert set(cache) == {"CAND_0001", "CAND_0002", "CAND_0003"}


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_signal_cache(tmp_path / "nope.jsonl") == {}


def test_load_skips_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "llm_signals.jsonl"
    path.write_text(
        json.dumps(_record("CAND_0001"))
        + "\n{ this is not json\n"
        + json.dumps(_record("CAND_0002"))
        + "\n",
        encoding="utf-8",
    )
    cache = load_signal_cache(path)
    assert set(cache) == {"CAND_0001", "CAND_0002"}  # the garbage line is skipped


def test_pending_ids_excludes_cached_and_sorts() -> None:
    cache = {"CAND_0002": _record("CAND_0002")}
    assert pending_ids(["CAND_0003", "CAND_0001", "CAND_0002"], cache) == ["CAND_0001", "CAND_0003"]


def test_chunk_splits_and_rejects_bad_size() -> None:
    assert list(chunk(["a", "b", "c", "d", "e"], 2)) == [["a", "b"], ["c", "d"], ["e"]]
    with pytest.raises(ValueError):
        list(chunk(["a"], 0))


# --------------------------------------------------------------------------- #
# extract_signals — the batched loop with a fake client                         #
# --------------------------------------------------------------------------- #
def _fake_model(
    seen: list[str],
    *,
    omit_in_batch: Collection[str] = (),
    never: Collection[str] = (),
) -> Callable[[str], str]:
    """A deterministic fake ``call_fn`` that echoes valid records for the ids it sees.

    ``omit_in_batch`` ids are dropped from multi-candidate calls (to force the
    per-candidate fallback) but returned on a single-candidate call. ``never`` ids
    are never returned (to exercise the failure path).
    """

    def call(prompt: str) -> str:
        ids = re.findall(r"candidate_id: (CAND_\w+)", prompt)
        seen.extend(ids)
        out = [
            _record(cid)
            for cid in ids
            if cid not in never and not (len(ids) > 1 and cid in omit_in_batch)
        ]
        return json.dumps(out)

    return call


def test_extract_only_requests_pending_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "llm_signals.jsonl"
    append_signals(path, [coerce_signal(_record("CAND_0001"))])  # type: ignore[list-item]
    shortlist = ["CAND_0001", "CAND_0002", "CAND_0003"]
    pending = pending_ids(shortlist, load_signal_cache(path))
    assert pending == ["CAND_0002", "CAND_0003"]

    seen: list[str] = []
    new_count, failed = extract_signals(
        profiles={cid: f"profile {cid}" for cid in pending},
        rubric_summary="RUBRIC",
        call_fn=_fake_model(seen),
        on_results=lambda recs: append_signals(path, recs),
        batch_size=8,
    )
    assert new_count == 2 and failed == []
    assert "CAND_0001" not in seen  # the cached candidate is never re-requested
    assert load_signal_cache(path).keys() == {"CAND_0001", "CAND_0002", "CAND_0003"}


def test_extract_falls_back_to_per_candidate(tmp_path: Path) -> None:
    path = tmp_path / "llm_signals.jsonl"
    pending = ["CAND_0001", "CAND_0002", "CAND_0003"]
    seen: list[str] = []
    new_count, failed = extract_signals(
        profiles={cid: f"profile {cid}" for cid in pending},
        rubric_summary="RUBRIC",
        # CAND_0002 is dropped from the batch but returns on its own → fallback fills it.
        call_fn=_fake_model(seen, omit_in_batch={"CAND_0002"}),
        on_results=lambda recs: append_signals(path, recs),
        batch_size=8,
    )
    assert new_count == 3 and failed == []
    assert load_signal_cache(path).keys() == {"CAND_0001", "CAND_0002", "CAND_0003"}


def test_extract_reports_unrecoverable_failures(tmp_path: Path) -> None:
    path = tmp_path / "llm_signals.jsonl"
    pending = ["CAND_0001", "CAND_0002"]
    new_count, failed = extract_signals(
        profiles={cid: f"profile {cid}" for cid in pending},
        rubric_summary="RUBRIC",
        call_fn=_fake_model([], never={"CAND_0002"}),  # never returns CAND_0002
        on_results=lambda recs: append_signals(path, recs),
        batch_size=8,
    )
    assert new_count == 1 and failed == ["CAND_0002"]


def test_extract_retries_a_garbled_batch() -> None:
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"  # first attempt unparseable → retry
        return json.dumps([_record(cid) for cid in re.findall(r"candidate_id: (CAND_\w+)", prompt)])

    collected: list[LLMSignals] = []
    new_count, failed = extract_signals(
        profiles={"CAND_0001": "p"},
        rubric_summary="RUBRIC",
        call_fn=flaky,
        on_results=collected.extend,
        batch_size=8,
    )
    assert new_count == 1 and failed == [] and calls["n"] == 2
    assert collected[0]["candidate_id"] == "CAND_0001"
