"""Sanity check for the candidate embeddings (Session 02 risk #4: skill leakage).

Embeds a hand-written "ideal Senior AI Engineer" sentence with the *same* model
and query prefix, cosine-ranks the whole pool against it, and prints the top-K
with their real titles. What we want to see: ML / retrieval / ranking / recsys /
search engineers surface near the top. What would be a red flag: an obvious
keyword-stuffer (e.g. a "Marketing Manager" with AI skills) dominating — that
would mean the embedding text is leaking skills and ``profile_text`` needs a fix
before any later session trusts these vectors.

Run from the repo root, after precompute 01 has produced the artifacts::

    python src/precompute/sanity_rank.py
    python src/precompute/sanity_rank.py --top 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embedding import (  # noqa: E402
    encode_normalized,
    load_embeddings,
    load_model,
    query_text,
)
from src.io_utils import load_candidates  # noqa: E402

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"

# Built from the JD's "how to read between the lines" ideal — career signal only,
# no bare skill keywords, so it probes for the *meaning* of the role.
IDEAL_QUERY = (
    "Senior AI engineer with six to eight years building embeddings-based "
    "retrieval, hybrid search and ranking systems in production at product "
    "companies. Shipped end-to-end recommendation and search systems to real "
    "users at scale, owned the offline evaluation framework (NDCG, MRR, MAP) and "
    "online A/B testing, and used vector databases and LLM re-ranking. Strong "
    "Python; understood retrieval and ranking before it was fashionable."
)


def rank_pool(query: str, artifacts_dir: Path, top: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``top`` (ids, scores) most similar candidates to ``query``."""
    ids, embeddings = load_embeddings(artifacts_dir, mmap=True)
    model = load_model()
    query_vec = encode_normalized(model, [query_text(query)])[0]
    # float16 matrix · float32 unit vector → float32 cosine scores (vectors are
    # L2-normalized, so the dot product is the cosine similarity).
    scores = np.asarray(embeddings, dtype=np.float32) @ query_vec
    order = np.argsort(-scores)[:top]
    return ids[order], scores[order]


def _titles_for(ids: np.ndarray, candidates_path: Path) -> dict[str, str]:
    """Look up ``current_title`` for the given ids by streaming the pool once."""
    wanted = set(ids.tolist())
    titles: dict[str, str] = {}
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if cid in wanted:
            profile = candidate.get("profile") or {}
            titles[cid] = profile.get("current_title") or "(no title)"
            if len(titles) == len(wanted):
                break
    return titles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

    top_ids, top_scores = rank_pool(IDEAL_QUERY, args.artifacts, args.top)
    titles = _titles_for(top_ids, args.candidates)

    print(f'\nTop {args.top} by cosine similarity to the "ideal Senior AI Engineer" query:\n')
    print(f"{'#':>3}  {'score':>6}  {'candidate_id':<14}  current_title")
    print("-" * 70)
    for rank, (cid, score) in enumerate(
        zip(top_ids.tolist(), top_scores.tolist(), strict=True), start=1
    ):
        print(f"{rank:>3}  {score:>6.3f}  {cid:<14}  {titles.get(cid, '?')}")
    print()


if __name__ == "__main__":
    main()
