"""Tests for the JD-reference logic (``src/jd_reference.py``) and its artifact.

Two layers, mirroring ``test_embedding.py``:

* **Pure logic** — the LLM response is validated into a typed rubric (loudly
  rejecting malformed shapes), and the reference round-trips through disk with its
  dimension guarded against the candidate embedding space.
* **Real artifact** — once ``02_build_jd_reference.py`` has run, assert the
  on-disk ``jd_reference.json`` satisfies the session's Definition of Done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import EMBEDDING
from src.embedding import META_FILE
from src.jd_reference import (
    _REQUIRED_LIST_FIELDS,
    JD_REFERENCE_FILE,
    JDReferenceError,
    assert_reference_dim,
    build_jd_reference,
    build_rubric_prompt,
    load_jd_reference,
    parse_jd_response,
    save_jd_reference,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _valid_payload() -> dict[str, Any]:
    """A well-formed LLM response, with stray whitespace to prove it's cleaned."""
    return {
        "rubric": {
            "role_archetypes": ["ML engineer", "  search/retrieval engineer "],
            "must_haves": ["embeddings retrieval in production", "vector DB operations"],
            "nice_to_haves": ["LoRA fine-tuning"],
            "hard_disqualifiers": ["pure research, no production", "consulting-only career"],
            "domains": ["retrieval", "ranking", "recommendation"],
            "seniority_band": {"min_years": 5, "max_years": 9, "notes": "ideal 6-8; a guide"},
        },
        "ideal_candidate_description": "  Six to eight years   building retrieval systems.  ",
    }


# --------------------------------------------------------------------------- #
# Prompt                                                                       #
# --------------------------------------------------------------------------- #
def test_prompt_embeds_jd_and_is_deterministic() -> None:
    jd = "Senior AI Engineer — build retrieval and ranking."
    prompt = build_rubric_prompt(jd)
    assert jd in prompt
    assert '"rubric"' in prompt and "ideal_candidate_description" in prompt
    assert prompt == build_rubric_prompt(jd)  # pure


# --------------------------------------------------------------------------- #
# parse_jd_response — happy path + cleaning                                     #
# --------------------------------------------------------------------------- #
def test_parse_returns_clean_rubric_and_ideal_text() -> None:
    rubric, ideal_text = parse_jd_response(_valid_payload())

    for field in _REQUIRED_LIST_FIELDS:
        assert rubric[field] and all(isinstance(item, str) for item in rubric[field])  # type: ignore[literal-required]
    assert "search/retrieval engineer" in rubric["role_archetypes"]  # whitespace collapsed
    band = rubric["seniority_band"]
    assert band["min_years"] == 5.0 and isinstance(band["min_years"], float)
    assert band["max_years"] == 9.0
    assert ideal_text == "Six to eight years building retrieval systems."  # collapsed


def test_parse_accepts_alternate_ideal_text_key() -> None:
    payload = _valid_payload()
    payload["ideal_text"] = payload.pop("ideal_candidate_description")
    _, ideal_text = parse_jd_response(payload)
    assert ideal_text.startswith("Six to eight years")


# --------------------------------------------------------------------------- #
# parse_jd_response — loud on malformed input                                   #
# --------------------------------------------------------------------------- #
def test_parse_rejects_non_object() -> None:
    with pytest.raises(JDReferenceError):
        parse_jd_response(["not", "an", "object"])


def test_parse_rejects_missing_rubric() -> None:
    with pytest.raises(JDReferenceError, match="rubric"):
        parse_jd_response({"ideal_candidate_description": "x"})


def test_parse_rejects_empty_list_field() -> None:
    payload = _valid_payload()
    payload["rubric"]["must_haves"] = []
    with pytest.raises(JDReferenceError, match="must_haves"):
        parse_jd_response(payload)


def test_parse_rejects_non_list_field() -> None:
    payload = _valid_payload()
    payload["rubric"]["domains"] = "retrieval"
    with pytest.raises(JDReferenceError, match="domains"):
        parse_jd_response(payload)


def test_parse_rejects_non_numeric_seniority() -> None:
    payload = _valid_payload()
    payload["rubric"]["seniority_band"]["min_years"] = "five"
    with pytest.raises(JDReferenceError, match="min_years"):
        parse_jd_response(payload)


def test_parse_rejects_bool_seniority() -> None:
    # bool is an int subclass; the validator must still reject it as non-numeric.
    payload = _valid_payload()
    payload["rubric"]["seniority_band"]["max_years"] = True
    with pytest.raises(JDReferenceError, match="max_years"):
        parse_jd_response(payload)


def test_parse_rejects_missing_ideal_text() -> None:
    payload = _valid_payload()
    del payload["ideal_candidate_description"]
    with pytest.raises(JDReferenceError):
        parse_jd_response(payload)


# --------------------------------------------------------------------------- #
# build / save / load / dim guard                                              #
# --------------------------------------------------------------------------- #
def _build_small_reference(dim: int = 4) -> Any:
    rubric, ideal_text = parse_jd_response(_valid_payload())
    return build_jd_reference(
        rubric=rubric,
        ideal_text=ideal_text,
        reference_embedding=[0.1 * i for i in range(dim)],
        model_id=EMBEDDING.model_id,
        query_prefix=EMBEDDING.query_prefix,
        llm_model="test-llm",
        jd_source="docs/challenge/job_description.md",
        created="2026-06-27",
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    reference = _build_small_reference()
    save_jd_reference(tmp_path, reference)
    loaded = load_jd_reference(tmp_path)

    assert (tmp_path / JD_REFERENCE_FILE).exists()
    assert loaded["embedding_dim"] == 4
    assert loaded["reference_embedding"] == reference["reference_embedding"]
    assert loaded["rubric"]["role_archetypes"] == reference["rubric"]["role_archetypes"]
    assert loaded["model_id"] == EMBEDDING.model_id


def test_load_rejects_file_without_embedding(tmp_path: Path) -> None:
    (tmp_path / JD_REFERENCE_FILE).write_text(json.dumps({"rubric": {}}), encoding="utf-8")
    with pytest.raises(JDReferenceError, match="reference_embedding"):
        load_jd_reference(tmp_path)


def _write_meta(tmp_path: Path, dim: int) -> None:
    (tmp_path / META_FILE).write_text(json.dumps({"dim": dim}), encoding="utf-8")


def test_assert_reference_dim_passes_on_match(tmp_path: Path) -> None:
    _write_meta(tmp_path, 4)
    assert_reference_dim(_build_small_reference(dim=4), tmp_path)  # no raise


def test_assert_reference_dim_raises_on_mismatch(tmp_path: Path) -> None:
    _write_meta(tmp_path, 768)
    with pytest.raises(JDReferenceError, match="dim"):
        assert_reference_dim(_build_small_reference(dim=4), tmp_path)


# --------------------------------------------------------------------------- #
# Real artifact (Definition of Done) — only after precompute 02 has run.        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (ARTIFACTS_DIR / JD_REFERENCE_FILE).exists(),
    reason="run src/precompute/02_build_jd_reference.py first",
)
def test_real_jd_reference_matches_embedding_space() -> None:
    reference = load_jd_reference(ARTIFACTS_DIR)
    # Dimension matches the candidate embeddings (the Session-03 gotcha guard).
    assert_reference_dim(reference, ARTIFACTS_DIR)
    assert reference["model_id"] == EMBEDDING.model_id
    assert reference["ideal_text"].strip()
    for field in _REQUIRED_LIST_FIELDS:
        assert reference["rubric"][field]  # type: ignore[literal-required]
