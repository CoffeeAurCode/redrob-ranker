"""Cheap pre-filter → shortlist export + the Phase-1 sanity check (Session 03).

OFFLINE step. Scores the whole pool against the JD reference embedding with one
vectorized dot product, applies the deterministic :func:`features.passes_prefilter`
(archetype title OR similarity ≥ threshold), and writes the surviving ids to
``artifacts/shortlist_ids.json`` — the input set for Sessions 04/05.

Three modes:

    python src/precompute/build_shortlist.py            # export shortlist + sanity report
    python src/precompute/build_shortlist.py --scan     # tune T: survivors per threshold, no write
    python src/precompute/build_shortlist.py --threshold 0.72   # override config T for this run

The export path streams the pool and calls ``passes_prefilter`` per candidate —
exactly the decision ``rank.py`` will make in Session 10 — so there is one source
of truth for "who is in the shortlist". ``--scan`` is an analysis helper that
vectorizes the same OR across a grid of thresholds to pick T.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FILTER, FilterConfig  # noqa: E402
from src.embedding import load_embeddings  # noqa: E402
from src.features import is_archetype_title, normalize_title, passes_prefilter  # noqa: E402
from src.io_utils import Candidate, load_candidates, use_utf8_stdout  # noqa: E402
from src.jd_reference import load_jd_reference  # noqa: E402

logger = logging.getLogger("build_shortlist")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
SHORTLIST_FILE = "shortlist_ids.json"

DEFAULT_SANITY_TOP = 50
# Threshold grid for --scan (BGE cosine on this pool lives roughly in 0.4-0.85).
_SCAN_GRID = [round(0.50 + 0.025 * i, 3) for i in range(15)]  # 0.500 … 0.850


def reference_similarities(artifacts_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of every candidate against the JD reference: returns ``(ids, sims)``.

    Vectors are L2-normalized, so the cosine is a plain dot product of the float16
    candidate matrix (upcast to float32) with the float32 reference vector.
    """
    reference = load_jd_reference(artifacts_dir)
    ref_vec = np.asarray(reference["reference_embedding"], dtype=np.float32)
    ids, embeddings = load_embeddings(artifacts_dir, mmap=True)
    sims = np.asarray(embeddings, dtype=np.float32) @ ref_vec
    return ids, sims


def export_shortlist(
    *,
    candidates_path: Path,
    ids: np.ndarray,
    sims: np.ndarray,
    cfg: FilterConfig,
) -> tuple[list[str], Counter[str]]:
    """Stream the pool, apply ``passes_prefilter``, return ``(sorted_ids, stats)``.

    ``stats`` counts how survivors were rescued: ``archetype`` (title only),
    ``similarity`` (score only) and ``both`` — useful for the sanity report.
    """
    sim_by_id = dict(zip(ids.tolist(), sims.tolist(), strict=True))
    survivors: list[str] = []
    stats: Counter[str] = Counter()
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if cid is None:
            continue
        sim = sim_by_id.get(cid, -1.0)
        if not passes_prefilter(candidate, sim, cfg):
            continue
        survivors.append(cid)
        stats[_branch(candidate, sim, cfg)] += 1
    # Deterministic on-disk order (candidate_id ascending) — the golden rule.
    return sorted(survivors), stats


def _branch(candidate: Candidate, sim: float, cfg: FilterConfig) -> str:
    """Which branch kept this survivor: 'archetype', 'similarity', or 'both'."""
    by_title = is_archetype_title(candidate, cfg)
    by_sim = sim >= cfg.similarity_threshold
    if by_title and by_sim:
        return "both"
    return "archetype" if by_title else "similarity"


def scan_thresholds(
    *,
    candidates_path: Path,
    ids: np.ndarray,
    sims: np.ndarray,
    cfg: FilterConfig,
    grid: list[float],
) -> None:
    """Print survivor counts across a threshold grid to help pick T (no write)."""
    archetype_ids = {
        cid
        for candidate in load_candidates(candidates_path)
        if (cid := candidate.get("candidate_id")) is not None and is_archetype_title(candidate, cfg)
    }
    archetype_mask = np.array([cid in archetype_ids for cid in ids.tolist()], dtype=bool)
    n_archetype = int(archetype_mask.sum())

    print(f"\nArchetype-title survivors (threshold-independent): {n_archetype}\n")
    print(f"{'T':>7}  {'sim>=T':>8}  {'union':>8}  {'+over archetype':>16}")
    print("-" * 46)
    for threshold in grid:
        by_sim = sims >= threshold
        union = int((archetype_mask | by_sim).sum())
        print(f"{threshold:>7.3f}  {int(by_sim.sum()):>8}  {union:>8}  {union - n_archetype:>16}")
    print()


def sanity_report(
    *,
    candidates_path: Path,
    ids: np.ndarray,
    sims: np.ndarray,
    cfg: FilterConfig,
    top: int,
) -> None:
    """Print the Phase-1 checkpoint: the top-K by similarity with their real titles."""
    order = np.argsort(-sims)[:top]
    wanted = {ids[i] for i in order}
    title_by_id: dict[str, str] = {}
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if cid is not None and cid in wanted:
            profile = candidate.get("profile") or {}
            title_by_id[cid] = profile.get("current_title") or "(no title)"
            if len(title_by_id) == len(wanted):
                break

    print(f'\nTop {top} by cosine similarity to the JD "ideal candidate":\n')
    print(f"{'#':>3}  {'score':>6}  {'arch':>4}  {'candidate_id':<14}  current_title")
    print("-" * 72)
    for rank, i in enumerate(order.tolist(), start=1):
        cid = ids[i]
        title = title_by_id.get(cid, "?")
        is_arch = "✓" if normalize_title(title) in cfg.archetype_titles else ""
        print(f"{rank:>3}  {float(sims[i]):>6.3f}  {is_arch:>4}  {cid:<14}  {title}")
    print()


def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    cfg: FilterConfig,
    top: int,
    scan: bool,
) -> None:
    """Compute similarities, then either scan thresholds or export + report."""
    ids, sims = reference_similarities(artifacts_dir)
    logger.info("scored %d candidates against the JD reference", len(ids))

    if scan:
        scan_thresholds(
            candidates_path=candidates_path, ids=ids, sims=sims, cfg=cfg, grid=_SCAN_GRID
        )
        return

    survivors, stats = export_shortlist(
        candidates_path=candidates_path, ids=ids, sims=sims, cfg=cfg
    )
    out_path = artifacts_dir / SHORTLIST_FILE
    out_path.write_text(json.dumps(survivors, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "shortlist: %d survivors at T=%.3f (archetype-only %d, similarity-only %d, both %d)",
        len(survivors),
        cfg.similarity_threshold,
        stats["archetype"],
        stats["similarity"],
        stats["both"],
    )
    logger.info("wrote %s", out_path)
    sanity_report(candidates_path=candidates_path, ids=ids, sims=sims, cfg=cfg, top=top)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--threshold", type=float, default=None, help="override the config similarity threshold T"
    )
    parser.add_argument("--top", type=int, default=DEFAULT_SANITY_TOP, help="sanity top-K size")
    parser.add_argument(
        "--scan", action="store_true", help="print survivors per threshold and exit (no write)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    cfg = (
        FILTER
        if args.threshold is None
        else dataclasses.replace(FILTER, similarity_threshold=args.threshold)
    )
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        cfg=cfg,
        top=args.top,
        scan=args.scan,
    )


if __name__ == "__main__":
    main()
