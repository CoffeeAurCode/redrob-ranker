"""Tests for the embedding helpers and the precompute-01 artifacts.

Two layers:

* **Pure helpers** (no model load) — the exact encoder input excludes skills, the
  float16 cast keeps vectors normalized, and the save/load round-trip preserves
  the id↔row alignment the rest of the pipeline depends on.
* **Real artifacts** — if ``artifacts/candidate_embeddings.npy`` exists (precompute
  has been run), assert the alignment/dtype invariants from the session's
  Definition of Done. Skipped on a fresh clone before precompute runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.embedding as emb_mod
from src.config import EMBEDDING
from src.embedding import (
    EMBEDDINGS_PREFIX,
    IDS_FILE,
    META_FILE,
    build_meta,
    embeddings_exist,
    load_embeddings,
    passage_text,
    query_text,
    save_artifacts,
    to_float16,
)
from src.io_utils import Candidate
from src.profile_text import build_embedding_text

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


# --------------------------------------------------------------------------- #
# Exact encoder input — the skill-leakage guard, one layer up from profile_text.
# --------------------------------------------------------------------------- #
def test_passage_text_is_build_embedding_text_with_prefix(sample_candidate: Candidate) -> None:
    assert passage_text(sample_candidate) == EMBEDDING.passage_prefix + build_embedding_text(
        sample_candidate
    )


def test_passage_text_excludes_skills(sample_candidate: Candidate) -> None:
    text = passage_text(sample_candidate)
    assert sample_candidate["skills"][0]["name"] not in text  # the leakage canary
    assert "Python" not in text


def test_query_text_applies_query_prefix() -> None:
    assert query_text("hello").startswith(EMBEDDING.query_prefix)
    assert query_text("hello").endswith("hello")


# --------------------------------------------------------------------------- #
# float16 cast keeps vectors (almost) unit-norm — so dot product ≈ cosine.
# --------------------------------------------------------------------------- #
def test_to_float16_preserves_shape_and_normalization() -> None:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((16, EMBEDDING.dim)).astype(np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    cast = to_float16(unit)

    assert cast.dtype == np.float16
    assert cast.shape == unit.shape
    norms = np.linalg.norm(cast.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)


# --------------------------------------------------------------------------- #
# Save/load round-trip preserves the id↔row alignment downstream joins rely on.
# --------------------------------------------------------------------------- #
def test_save_load_round_trip_is_aligned(tmp_path: Path) -> None:
    ids = np.array(["CAND_0000001", "CAND_0000002", "CAND_0000003"])
    embeddings = to_float16(np.eye(3, EMBEDDING.dim, dtype=np.float32))
    meta = build_meta(
        n=3, source="test.jsonl", sentence_transformers_version="x", created="2026-06-26"
    )

    save_artifacts(tmp_path, ids, embeddings, meta)
    loaded_ids, loaded_emb = load_embeddings(tmp_path, mmap=False)

    assert embeddings_exist(tmp_path)
    assert (tmp_path / IDS_FILE).exists()
    assert (tmp_path / META_FILE).exists()
    assert loaded_ids.tolist() == ids.tolist()
    assert loaded_emb.shape == (3, EMBEDDING.dim)
    assert loaded_emb.dtype == np.float16


def test_save_artifacts_rejects_misaligned_inputs(tmp_path: Path) -> None:
    ids = np.array(["CAND_0000001", "CAND_0000002"])
    embeddings = to_float16(np.zeros((3, EMBEDDING.dim), dtype=np.float32))  # 3 ≠ 2
    meta = build_meta(n=2, source="t", sentence_transformers_version="x", created="2026-06-26")
    with pytest.raises(ValueError, match="mismatch"):
        save_artifacts(tmp_path, ids, embeddings, meta)


def test_sharding_splits_and_stitches_losslessly(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Force a tiny shard budget so a small array still splits into many files.
    monkeypatch.setattr(emb_mod, "_MAX_SHARD_BYTES", 4096)
    rng = np.random.default_rng(0)
    ids = np.array([f"CAND_{i:07d}" for i in range(50)])
    embeddings = to_float16(rng.standard_normal((50, EMBEDDING.dim)).astype(np.float32))
    meta = build_meta(n=50, source="t", sentence_transformers_version="x", created="2026-06-26")

    save_artifacts(tmp_path, ids, embeddings, meta)

    shard_files = sorted(tmp_path.glob(f"{EMBEDDINGS_PREFIX}_*.npy"))
    assert len(shard_files) > 1  # actually split across files
    loaded_ids, loaded_emb = load_embeddings(tmp_path, mmap=False)
    assert loaded_ids.tolist() == ids.tolist()
    assert np.array_equal(loaded_emb, embeddings)  # stitched back in row order, lossless
    import json

    assert json.loads((tmp_path / META_FILE).read_text())["shards"] == len(shard_files)


def test_build_meta_records_provenance() -> None:
    meta = build_meta(
        n=100,
        source="candidates.jsonl",
        sentence_transformers_version="5.6.0",
        created="2026-06-26",
    )
    assert meta["model_id"] == EMBEDDING.model_id
    assert meta["dim"] == EMBEDDING.dim
    assert meta["n"] == 100
    assert meta["normalize"] is True
    assert meta["embeddings_dtype"] == "float16"
    assert meta["passage_prefix"] == EMBEDDING.passage_prefix
    assert meta["query_prefix"] == EMBEDDING.query_prefix


# --------------------------------------------------------------------------- #
# Real artifacts (Definition of Done) — only when precompute 01 has been run.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not embeddings_exist(ARTIFACTS_DIR),
    reason="run src/precompute/01_embed_candidates.py first",
)
def test_real_artifacts_are_aligned_and_float16() -> None:
    ids, embeddings = load_embeddings(ARTIFACTS_DIR, mmap=True)
    assert len(ids) == embeddings.shape[0]
    assert embeddings.shape[1] == EMBEDDING.dim
    assert embeddings.dtype == np.float16
    assert len(set(ids.tolist())) == len(ids)  # ids unique
