"""Embedding helpers shared by the precompute encoders (Sessions 02 and 03).

This is the single place that knows *how* text becomes a vector and *how* the
resulting matrix is stored. The candidate encoder (``precompute/01_…``) and the
JD-reference builder (Session 03) both import from here so the two sides land in
the same vector space with the same prefix/normalization rules.

Design notes:

* **Lazy heavy import.** ``sentence_transformers`` (and its torch dependency) is
  imported inside :func:`load_model` only, so the pure text/array helpers — and
  their unit tests — run without loading the model stack.
* **Normalize in float32, store in float16.** Encoding returns unit-norm float32
  vectors (cosine == dot product); we cast to float16 for compact, mmap-friendly
  artifacts. The norm stays ≈1 after the cast, so dot products still behave.
* **The id array is the canonical alignment.** ``candidate_ids.npy`` is row-aligned
  to ``candidate_embeddings.npy``; every later artifact joins on ``candidate_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.config import EMBEDDING, EmbeddingConfig
from src.io_utils import Candidate
from src.profile_text import build_embedding_text

if TYPE_CHECKING:  # avoid importing torch at module load (and for type checkers).
    from sentence_transformers import SentenceTransformer

# Artifact file names (defined once; rank.py and Session 03 reuse these).
EMBEDDINGS_FILE = "candidate_embeddings.npy"
IDS_FILE = "candidate_ids.npy"
META_FILE = "embeddings_meta.json"


# --------------------------------------------------------------------------- #
# Text → exact encoder input.                                                  #
# --------------------------------------------------------------------------- #
def passage_text(candidate: Candidate, cfg: EmbeddingConfig = EMBEDDING) -> str:
    """The exact string encoded for a candidate (corpus side).

    The career-only profile text from :func:`build_embedding_text` (skills
    excluded — see ``profile_text``) prefixed with the corpus instruction, which
    is empty for BGE v1.5. Keeping the prefix as a seam means a model that *does*
    want a passage instruction needs no change here.
    """
    return cfg.passage_prefix + build_embedding_text(candidate)


def query_text(text: str, cfg: EmbeddingConfig = EMBEDDING) -> str:
    """The exact string encoded for a query (the JD reference or a sanity probe).

    Applies the BGE retrieval instruction so a query embeds into the same space
    as the (instruction-free) candidate passages.
    """
    return cfg.query_prefix + text


# --------------------------------------------------------------------------- #
# Encoding (requires the model — the only network/heavyweight step).           #
# --------------------------------------------------------------------------- #
def load_model(cfg: EmbeddingConfig = EMBEDDING) -> SentenceTransformer:
    """Load the sentence-transformer once. Offline-only; never called by rank.py."""
    from sentence_transformers import SentenceTransformer

    return cast("SentenceTransformer", SentenceTransformer(cfg.model_id))


def encode_normalized(
    model: SentenceTransformer,
    texts: list[str],
    cfg: EmbeddingConfig = EMBEDDING,
) -> np.ndarray:
    """Encode texts to L2-normalized float32 vectors (cosine == dot product)."""
    vectors = model.encode(
        texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=cfg.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def to_float16(vectors: np.ndarray) -> np.ndarray:
    """Cast normalized float32 vectors to compact float16 (norm stays ≈1)."""
    return np.asarray(vectors, dtype=np.float32).astype(np.float16)


# --------------------------------------------------------------------------- #
# Persistence + provenance.                                                    #
# --------------------------------------------------------------------------- #
def save_artifacts(
    out_dir: Path,
    ids: np.ndarray,
    embeddings: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Write the embeddings matrix, the aligned id array, and the manifest."""
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"id/row mismatch: {len(ids)} ids vs {embeddings.shape[0]} rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / EMBEDDINGS_FILE, embeddings)
    np.save(out_dir / IDS_FILE, ids)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_embeddings(
    artifacts_dir: Path,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(ids, embeddings)``; embeddings are memory-mapped by default.

    ``mmap=True`` is the rank-time path: the float16 matrix stays on disk and is
    paged in on demand, keeping RAM low (Session 10).
    """
    embeddings = np.load(
        artifacts_dir / EMBEDDINGS_FILE,
        mmap_mode="r" if mmap else None,
    )
    ids = np.load(artifacts_dir / IDS_FILE)
    return ids, embeddings


def build_meta(
    n: int,
    *,
    cfg: EmbeddingConfig = EMBEDDING,
    source: str,
    sentence_transformers_version: str,
    created: str,
) -> dict[str, Any]:
    """Provenance manifest for the embedding artifacts (defensibility at Stage 5)."""
    return {
        "model_id": cfg.model_id,
        "dim": cfg.dim,
        "n": n,
        "normalize": cfg.normalize,
        "similarity": "cosine == dot product (vectors L2-normalized)",
        "embeddings_dtype": "float16",
        # Candidates (corpus) use passage_prefix; the JD/sanity query uses
        # query_prefix. Recorded so Session 03 applies the identical convention.
        "passage_prefix": cfg.passage_prefix,
        "query_prefix": cfg.query_prefix,
        "text_source": "profile_text.build_embedding_text (career text; skills excluded)",
        "source": source,
        "sentence_transformers_version": sentence_transformers_version,
        "created": created,
    }
