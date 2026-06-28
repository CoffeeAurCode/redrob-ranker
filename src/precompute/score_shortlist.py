"""Score the shortlist end-to-end and print a ranked sanity report (Session 06).

OFFLINE analysis tool — **not** the submission entrypoint (that is Session 10's
``rank.py``), but it exercises the exact same path: load the precomputed artifacts,
assemble :class:`features.Features`, apply :func:`scoring.score`, and sort by the
golden-rule key (score descending, then ``candidate_id`` ascending). It is the
harness Session 08 uses to watch the ranking move as the weights are calibrated.

It reads only committed artifacts (embeddings, ``jd_reference.json``,
``llm_signals.jsonl``, ``honeypot_flags.json``) and the candidate pool; it makes no
network/LLM call. Run from the repo root::

    python src/precompute/score_shortlist.py              # top-40 sanity table
    python src/precompute/score_shortlist.py --top 100    # show the prospective top 100
    python src/precompute/score_shortlist.py --explain 5  # full breakdown for the top 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import SCORING, ScoringConfig  # noqa: E402
from src.embedding import load_embeddings  # noqa: E402
from src.features import Features, assemble_features  # noqa: E402
from src.honeypots import load_flagged_ids  # noqa: E402
from src.io_utils import Candidate, load_candidates, use_utf8_stdout  # noqa: E402
from src.jd_reference import load_jd_reference  # noqa: E402
from src.llm_signals import LLM_SIGNALS_FILE, load_signal_cache  # noqa: E402
from src.scoring import ScoreResult, score  # noqa: E402

logger = logging.getLogger("score_shortlist")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
SHORTLIST_FILE = "shortlist_ids.json"
DEFAULT_TOP = 40


@dataclass(frozen=True)
class ScoredCandidate:
    """A shortlisted candidate with its assembled features, score, and display title."""

    candidate_id: str
    title: str
    features: Features
    result: ScoreResult


def similarity_by_id(artifacts_dir: Path) -> dict[str, float]:
    """Cosine of every candidate against the JD reference, keyed by ``candidate_id``.

    Vectors are L2-normalized, so cosine is a plain dot product of the float16
    candidate matrix (upcast to float32) with the float32 reference vector.
    """
    reference = load_jd_reference(artifacts_dir)
    ref_vec = np.asarray(reference["reference_embedding"], dtype=np.float32)
    ids, embeddings = load_embeddings(artifacts_dir, mmap=True)
    sims = np.asarray(embeddings, dtype=np.float32) @ ref_vec
    return dict(zip(ids.tolist(), sims.tolist(), strict=True))


def score_shortlist(
    *,
    candidates_path: Path,
    shortlist: set[str],
    sim_by_id: dict[str, float],
    signals: dict[str, object],
    flagged: frozenset[str],
    cfg: ScoringConfig,
) -> list[ScoredCandidate]:
    """Assemble + score every shortlisted candidate, sorted by the rank-time key.

    Streams the pool (never slurps it), scores only the shortlist (the set with LLM
    signals), and returns them sorted by score descending then ``candidate_id``
    ascending — the exact deterministic order ``rank.py`` will emit.
    """
    scored: list[ScoredCandidate] = []
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or cid not in shortlist:
            continue
        features = assemble_features(
            candidate,
            cosine=sim_by_id.get(cid, -1.0),
            signals=signals.get(cid),  # type: ignore[arg-type]  # LLMSignals | None
            honeypot=cid in flagged,
            cfg=cfg,
        )
        result = score(features, cfg)
        scored.append(ScoredCandidate(cid, _title(candidate), features, result))

    scored.sort(key=lambda s: (-s.result.final, s.candidate_id))
    return scored


def _title(candidate: Candidate) -> str:
    profile = candidate.get("profile") or {}
    return profile.get("current_title") or "(no title)"


# --------------------------------------------------------------------------- #
# Reporting.                                                                    #
# --------------------------------------------------------------------------- #
def print_table(scored: list[ScoredCandidate], top: int) -> None:
    """Print the ranked top-K with the headline score and its multipliers/penalties."""
    print(f"\nTop {min(top, len(scored))} of {len(scored)} scored shortlist candidates:\n")
    header = f"{'#':>3}  {'final':>6}  {'base':>5}  {'pen':>4}  {'avail':>5}  {'loc':>4}  "
    header += f"{'candidate_id':<14}  current_title  [flags]"
    print(header)
    print("-" * len(header))
    for rank, sc in enumerate(scored[:top], start=1):
        r = sc.result
        flags = ",".join(sc.features.disqualifier_flags)
        flag_str = f"  [{flags}]" if flags else ""
        hp = "  HONEYPOT" if r.honeypot else ""
        print(
            f"{rank:>3}  {r.final:>6.3f}  {r.base_fit:>5.3f}  {r.penalties:>4.2f}  "
            f"{r.availability:>5.3f}  {r.location:>4.2f}  {sc.candidate_id:<14}  "
            f"{sc.title}{flag_str}{hp}"
        )
    print()


def print_explanations(scored: list[ScoredCandidate], n: int) -> None:
    """Print the full term-by-term breakdown for the top-N (the Stage-5 'explain' view)."""
    print(f"=== Full breakdown, top {min(n, len(scored))} ===\n")
    for rank, sc in enumerate(scored[:n], start=1):
        f, r = sc.features, sc.result
        print(f"#{rank}  {sc.candidate_id}  {sc.title}  ->  final={r.final:.4f}")
        for term, contribution in r.contributions.items():
            raw = getattr(f, term)
            print(f"     {term:<17} raw={raw:>5.3f}  contribution={contribution:>6.4f}")
        print(
            f"     base_fit={r.base_fit:.4f}  penalties={r.penalties:.4f} {dict(r.penalty_detail)}"
        )
        print(f"     availability={r.availability:.3f}  location={r.location:.2f}", end="")
        print(f"  honeypot={r.honeypot}\n")


def print_summary(scored: list[ScoredCandidate], top_n: int = 100) -> None:
    """Print the checkpoints that matter: honeypots in the top 100 and the availability spread."""
    top = scored[:top_n]
    honeypots = [sc.candidate_id for sc in top if sc.result.honeypot]
    flag_count = sum(1 for sc in top if sc.features.disqualifier_flags)
    print(f"=== Summary (top {len(top)}) ===")
    print(f"  honeypots in top {len(top)}: {len(honeypots)}  (must be 0)")
    print(f"  candidates carrying a disqualifier flag: {flag_count}")
    if top:
        avails = [sc.features.availability for sc in top]
        print(f"  availability multiplier range: {min(avails):.3f} - {max(avails):.3f}")
        print("  (a strong-but-inactive candidate should still rank high — Session 08 verifies)")
    print()


def run(
    *, artifacts_dir: Path, candidates_path: Path, cfg: ScoringConfig, top: int, explain: int
) -> None:
    """Load artifacts, score the shortlist, and print the ranked report."""
    shortlist_path = artifacts_dir / SHORTLIST_FILE
    shortlist = set(_load_json_list(shortlist_path))
    sim_by_id = similarity_by_id(artifacts_dir)
    signals = load_signal_cache(artifacts_dir / LLM_SIGNALS_FILE)
    flagged = load_flagged_ids(artifacts_dir)
    logger.info(
        "shortlist=%d  signals=%d  honeypot-flags=%d", len(shortlist), len(signals), len(flagged)
    )

    scored = score_shortlist(
        candidates_path=candidates_path,
        shortlist=shortlist,
        sim_by_id=sim_by_id,
        signals=dict(signals),
        flagged=flagged,
        cfg=cfg,
    )
    print_table(scored, top)
    if explain > 0:
        print_explanations(scored, explain)
    print_summary(scored)


def _load_json_list(path: Path) -> list[str]:
    import json

    if not path.exists():
        raise SystemExit(f"{path} not found — run build_shortlist.py first.")
    return list(json.loads(path.read_text(encoding="utf-8")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="rows in the ranked table")
    parser.add_argument(
        "--explain", type=int, default=0, help="print the full breakdown for the top N"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        cfg=SCORING,
        top=args.top,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
