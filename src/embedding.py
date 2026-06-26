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
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from src.config import EMBEDDING, EmbeddingConfig
from src.io_utils import Candidate
from src.profile_text import build_embedding_text

if TYPE_CHECKING:  # avoid importing torch at module load (and for type checkers).
    from sentence_transformers import SentenceTransformer

# Artifact file names (defined once; rank.py and Session 03 reuse these).
# The embeddings matrix is stored as one or more row-shards
# (``candidate_embeddings_000.npy`` …) so no single file exceeds GitHub's 100 MiB
# push limit and the repo stays self-contained (no Git LFS). That matters because
# Stage 3 re-runs rank.py from a plain clone in a network-less container, where an
# un-pulled LFS pointer would be a silent failure.
EMBEDDINGS_PREFIX = "candidate_embeddings"
EMBEDDINGS_FILE = f"{EMBEDDINGS_PREFIX}.npy"  # legacy single-file (still loadable)
IDS_FILE = "candidate_ids.npy"
META_FILE = "embeddings_meta.json"

# Keep each shard under GitHub's 50 MiB warning (and well under the 100 MiB limit).
_MAX_SHARD_BYTES = 45 * 1024 * 1024


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
def _shard_name(index: int) -> str:
    return f"{EMBEDDINGS_PREFIX}_{index:03d}.npy"


def _shard_paths(artifacts_dir: Path) -> list[Path]:
    """Embedding shard files in row order (empty if only a legacy file exists)."""
    return sorted(artifacts_dir.glob(f"{EMBEDDINGS_PREFIX}_*.npy"))


def _plan_shard_count(embeddings: np.ndarray) -> int:
    """How many row-shards keep each file under ``_MAX_SHARD_BYTES`` (>= 1)."""
    return max(1, -(-embeddings.nbytes // _MAX_SHARD_BYTES))  # ceil division


def embeddings_exist(artifacts_dir: Path) -> bool:
    """True if sharded (or legacy single-file) embeddings are present."""
    return bool(_shard_paths(artifacts_dir)) or (artifacts_dir / EMBEDDINGS_FILE).exists()


def save_artifacts(
    out_dir: Path,
    ids: np.ndarray,
    embeddings: np.ndarray,
    meta: dict[str, Any],
) -> None:
    """Write the row-sharded embeddings, the aligned id array, and the manifest.

    The matrix is split by rows into ``candidate_embeddings_<NNN>.npy`` shards,
    each under ``_MAX_SHARD_BYTES`` so every file is committable to GitHub. The
    caller's ``meta`` is preserved and augmented with the shard count.
    """
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"id/row mismatch: {len(ids)} ids vs {embeddings.shape[0]} rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_embeddings(out_dir)

    shards = np.array_split(embeddings, _plan_shard_count(embeddings), axis=0)
    for index, shard in enumerate(shards):
        np.save(out_dir / _shard_name(index), shard)

    np.save(out_dir / IDS_FILE, ids)
    (out_dir / META_FILE).write_text(
        json.dumps({**meta, "shards": len(shards)}, indent=2) + "\n", encoding="utf-8"
    )


def _clear_embeddings(out_dir: Path) -> None:
    """Remove any existing shards / legacy file so a re-save leaves no stale rows."""
    for path in out_dir.glob(f"{EMBEDDINGS_PREFIX}_*.npy"):
        path.unlink()
    legacy = out_dir / EMBEDDINGS_FILE
    if legacy.exists():
        legacy.unlink()


def load_embeddings(
    artifacts_dir: Path,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(ids, embeddings)`` from row-shards (or a legacy single file).

    A single shard is returned memory-mapped when ``mmap=True`` (lazy, low RAM);
    multiple shards are stitched with ``np.concatenate``, which materializes the
    full float16 matrix (~150 MiB for the 100k pool — trivial under the 16 GB
    budget). ``mmap`` still avoids a float32 upcast on load.
    """
    mode: Literal["r"] | None = "r" if mmap else None
    shards = _shard_paths(artifacts_dir)
    if shards:
        parts = [np.load(path, mmap_mode=mode) for path in shards]
        embeddings = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    else:
        embeddings = np.load(artifacts_dir / EMBEDDINGS_FILE, mmap_mode=mode)
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
