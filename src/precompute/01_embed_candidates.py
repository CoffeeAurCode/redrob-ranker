"""Precompute 01 — embed every candidate from their career text.

OFFLINE step (the golden rule: this never runs inside ``rank.py``). Streams the
~100k pool, builds each candidate's skills-free career text via
``profile_text.build_embedding_text``, encodes it with the configured BGE model,
and writes three row-aligned artifacts under ``artifacts/``:

    candidate_embeddings.npy   (N, dim) float16, L2-normalized
    candidate_ids.npy          (N,)     str, index-aligned to the rows above
    embeddings_meta.json       provenance (model id, dim, prefixes, date, …)

Run from the repo root::

    python src/precompute/01_embed_candidates.py                  # full pool
    python src/precompute/01_embed_candidates.py --limit 200      # smoke test
    python src/precompute/01_embed_candidates.py --force          # rebuild

The encode is the slow part on CPU, so it is **idempotent and resumable**: vectors
are checkpointed to ``artifacts/.embed_ckpt/`` one chunk at a time; a crash
re-resumes from the last completed chunk. On success the checkpoint dir is removed
and the final artifacts are assembled in frozen stream order.

This module's filename starts with a digit, so it cannot be imported — all of the
reusable logic lives in ``src.embedding`` (which Session 03 also uses). This file
is the thin CLI/orchestration layer only.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np

# Runnable scripts under src/ aren't on the import path; add the repo root so the
# `src` package (io_utils, profile_text, embedding, config) resolves.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBEDDING  # noqa: E402  (after sys.path bootstrap)
from src.embedding import (  # noqa: E402
    build_meta,
    encode_normalized,
    load_model,
    passage_text,
    save_artifacts,
    to_float16,
)
from src.io_utils import Candidate, load_candidates  # noqa: E402

logger = logging.getLogger("embed_candidates")

DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_OUT = REPO_ROOT / "artifacts"
CKPT_DIRNAME = ".embed_ckpt"
DEFAULT_CHUNK_SIZE = 1000

# Zero-padded chunk index width: 100k / 1000 ≈ 100 chunks, 5 digits is ample.
_CHUNK_GLOB = "chunk_*.npy"
_IDS_GLOB = "ids_*.npy"


def _chunk_path(ckpt: Path, idx: int) -> Path:
    return ckpt / f"chunk_{idx:05d}.npy"


def _ids_path(ckpt: Path, idx: int) -> Path:
    return ckpt / f"ids_{idx:05d}.npy"


def _completed_chunks(ckpt: Path) -> int:
    """Number of fully-written chunks (an id file is written last, so it gates)."""
    return len(sorted(ckpt.glob(_IDS_GLOB)))


def stream_chunks(
    data_path: Path,
    chunk_size: int,
    *,
    skip: int = 0,
    limit: int | None = None,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield ``(ids, texts)`` chunks of up to ``chunk_size`` candidates in file order.

    ``skip`` fast-forwards past already-checkpointed candidates on resume; ``limit``
    caps the total number processed (smoke tests). Each candidate must carry a
    ``candidate_id`` — precompute fails loudly if one is missing.
    """
    ids: list[str] = []
    texts: list[str] = []
    seen = 0
    for candidate in load_candidates(data_path):
        if seen < skip:
            seen += 1
            continue
        if limit is not None and seen >= skip + limit:
            break
        ids.append(_require_id(candidate))
        texts.append(passage_text(candidate))
        seen += 1
        if len(ids) == chunk_size:
            yield ids, texts
            ids, texts = [], []
    if ids:
        yield ids, texts


def _require_id(candidate: Candidate) -> str:
    candidate_id = candidate.get("candidate_id")
    if not candidate_id:
        raise ValueError("candidate is missing 'candidate_id'")
    return candidate_id


def _assemble(ckpt: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all checkpoint chunks back into the final id + embedding arrays."""
    id_files = sorted(ckpt.glob(_IDS_GLOB))
    emb_files = sorted(ckpt.glob(_CHUNK_GLOB))
    if len(id_files) != len(emb_files):
        raise RuntimeError(
            f"checkpoint corrupt: {len(emb_files)} chunks vs {len(id_files)} id files"
        )
    ids = np.concatenate([np.load(f) for f in id_files])
    embeddings = np.concatenate([np.load(f) for f in emb_files], axis=0)
    return ids, embeddings


def run_embedding(
    *,
    data_path: Path,
    out_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    force: bool = False,
) -> None:
    """Embed the pool into ``out_dir`` with resumable checkpointing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "candidate_embeddings.npy"
    if final.exists() and not force:
        logger.info("artifacts already present at %s — use --force to rebuild", final)
        return

    ckpt = out_dir / CKPT_DIRNAME
    ckpt.mkdir(exist_ok=True)
    start_idx = _completed_chunks(ckpt)
    skip = start_idx * chunk_size
    if start_idx:
        logger.info("resuming from chunk %d (%d candidates already embedded)", start_idx, skip)

    logger.info("loading model %s …", EMBEDDING.model_id)
    model = load_model()

    idx = start_idx
    started = time.perf_counter()
    for chunk_ids, chunk_texts in stream_chunks(data_path, chunk_size, skip=skip, limit=limit):
        vectors = to_float16(encode_normalized(model, chunk_texts))
        # Embeddings first, id file last: the id file's presence marks "done".
        np.save(_chunk_path(ckpt, idx), vectors)
        np.save(_ids_path(ckpt, idx), np.asarray(chunk_ids))
        idx += 1
        done = idx * chunk_size if limit is None else min(idx * chunk_size, skip + limit)
        rate = (idx - start_idx) * chunk_size / max(time.perf_counter() - started, 1e-6)
        logger.info("chunk %d done · ~%d candidates · %.0f/s", idx, done, rate)

    ids, embeddings = _assemble(ckpt)
    meta = build_meta(
        n=int(embeddings.shape[0]),
        source=str(data_path.name),
        sentence_transformers_version=_st_version(),
        created=date.today().isoformat(),
    )
    save_artifacts(out_dir, ids, embeddings, meta)
    shutil.rmtree(ckpt)
    logger.info(
        "wrote %d embeddings (dim %d, %s) to %s",
        embeddings.shape[0],
        embeddings.shape[1],
        embeddings.dtype,
        out_dir,
    )


def _st_version() -> str:
    import sentence_transformers

    return str(sentence_transformers.__version__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES, help="path to candidates.jsonl"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifacts output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="candidates per checkpoint"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap candidates (smoke test)")
    parser.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run_embedding(
        data_path=args.candidates,
        out_dir=args.out,
        chunk_size=args.chunk_size,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
