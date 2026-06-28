"""Evaluate the system ranking against the hand-labeled gold set (Session 07, Part C).

Loads ``eval/gold_labels.csv`` (human-reviewed tiers), rebuilds the exact system
ranking (:mod:`eval.ranking`), joins the two, and reports the challenge's four
metrics — **NDCG@10, NDCG@50, MAP, P@10** — plus the targeted checks that catch the
failure modes this dataset is built around:

* honeypots in the top 100 (must be **0** — >10% is an instant disqualification),
* off-target / disqualified candidates leaking into the top, and
* whether the **inactive-but-perfect** fits and the labeled tier-5/4 candidates are
  actually surfaced near the top.

Metrics are computed over the system ranking **restricted to judged candidates**
(the standard pooled-evaluation approach for a sampled gold set): the binary cutoff
for MAP/P@10 is ``tier >= 3`` (:data:`eval.metrics.RELEVANCE_CUTOFF`). The report is
printed and written to ``eval/metrics.json``; it is deterministic (no wall-clock) so
re-runs diff cleanly as Session 08 tunes the weights. Run from the repo root::

    python eval/evaluate.py
    python eval/evaluate.py --gold eval/gold_labels.csv --out eval/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (  # noqa: E402
    RELEVANCE_CUTOFF,
    average_precision,
    binary_relevances,
    ndcg,
    precision_at_k,
)
from eval.ranking import RankedCandidate, build_ranking  # noqa: E402
from src.config import SCORING, ScoringConfig  # noqa: E402
from src.io_utils import use_utf8_stdout  # noqa: E402

logger = logging.getLogger("evaluate")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold_labels.csv"
DEFAULT_OUT = REPO_ROOT / "eval" / "metrics.json"

# A judged candidate counts as "off-target near the top" if its role family is a
# clean mismatch (CV/speech, non-tech) — used only for the surfaced warning, not the
# metrics themselves. The penalty/honeypot machinery already down-weights these.
_OFF_TARGET_ROLE_MATCH = 0.30

# "Inactive-but-perfect" surfacing: a strong-hire label (tier >= 4) whose availability
# multiplier sits in roughly the pool's bottom quintile. Tracks the Session-08 calibrated
# floor (availability now runs ~[0.70, 0.95] since availability_floor=0.70); 0.85 is near
# its low end, so the check still names the least-reachable strong hires it lifted but did
# not bury. Diagnostic only — it does not affect any metric.
_INACTIVE_AVAILABILITY = 0.85


@dataclass(frozen=True)
class GoldLabel:
    """One human label: a relevance tier (0-5) and an optional reviewer note."""

    candidate_id: str
    tier: int
    note: str


@dataclass(frozen=True)
class JudgedRow:
    """A gold-labeled candidate located in the system ranking."""

    rank: int
    candidate_id: str
    tier: int


def parse_gold_labels(path: Path) -> dict[str, GoldLabel]:
    """Read ``candidate_id,tier[,note]`` labels, skipping rows with a blank tier.

    Accepts the simple gold file or the fuller ``gold_review.csv`` (it only needs the
    ``candidate_id`` and ``tier`` columns). Blank/missing tiers are treated as
    "not yet labeled" and ignored, so a partially-filled sheet still evaluates. A
    present-but-non-integer or out-of-range (not 0-5) tier is a hard error — a typo in
    the one feedback signal must fail loudly, never silently mislabel.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build the review sheet with `python eval/label_helper.py`, "
            "hand-label it per eval/LABELING_GUIDE.md, and save eval/gold_labels.csv."
        )
    labels: dict[str, GoldLabel] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "candidate_id" not in reader.fieldnames:
            raise SystemExit(f"{path}: missing required 'candidate_id' column.")
        if "tier" not in reader.fieldnames:
            raise SystemExit(f"{path}: missing required 'tier' column.")
        for lineno, row in enumerate(reader, start=2):
            cid = (row.get("candidate_id") or "").strip()
            raw_tier = (row.get("tier") or "").strip()
            if not cid or not raw_tier:
                continue
            try:
                tier = int(raw_tier)
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: tier {raw_tier!r} is not an integer.") from exc
            if not 0 <= tier <= 5:
                raise SystemExit(f"{path}:{lineno}: tier {tier} out of range (expected 0-5).")
            labels[cid] = GoldLabel(cid, tier, (row.get("note") or "").strip())
    return labels


def judged_rows(ranking: list[RankedCandidate], gold: dict[str, GoldLabel]) -> list[JudgedRow]:
    """Gold-labeled candidates in system-ranked order (the input to every metric)."""
    return [
        JudgedRow(c.rank, c.candidate_id, gold[c.candidate_id].tier)
        for c in ranking
        if c.candidate_id in gold
    ]


def build_report(
    ranking: list[RankedCandidate],
    gold: dict[str, GoldLabel],
    *,
    cutoff: int = RELEVANCE_CUTOFF,
    cfg: ScoringConfig = SCORING,
) -> dict[str, Any]:
    """Compute the four metrics and the targeted checks. Pure (no I/O)."""
    judged = judged_rows(ranking, gold)
    tiers = [row.tier for row in judged]
    relevances = binary_relevances(tiers, cutoff)

    ranked_ids = {c.candidate_id for c in ranking}
    missing = sorted(set(gold) - ranked_ids)
    feature_by_id = {c.candidate_id: c for c in ranking}
    missing_relevant = [cid for cid in missing if gold[cid].tier >= cutoff]

    metrics = {
        "ndcg_at_10": round(ndcg(tiers, 10), 6),
        "ndcg_at_50": round(ndcg(tiers, 50), 6),
        "map": round(average_precision(relevances), 6),  # one JD ⇒ MAP == AP
        "precision_at_10": round(precision_at_k(relevances, 10), 6),
    }

    honeypots_top_100 = sorted(c.candidate_id for c in ranking[:100] if c.features.honeypot)
    off_target_top_50 = sorted(
        c.candidate_id
        for c in ranking[:50]
        if c.features.honeypot
        or c.features.role_match <= _OFF_TARGET_ROLE_MATCH
        or c.features.disqualifier_flags
    )

    tier_ranks = {tier: sorted(row.rank for row in judged if row.tier == tier) for tier in range(6)}
    # "Inactive-but-perfect": labeled a strong hire (tier >= 4) yet low availability.
    inactive_items: list[dict[str, Any]] = [
        {
            "candidate_id": row.candidate_id,
            "tier": row.tier,
            "rank": row.rank,
            "availability": round(feature_by_id[row.candidate_id].features.availability, 3),
        }
        for row in judged
        if row.tier >= 4
        and feature_by_id[row.candidate_id].features.availability <= _INACTIVE_AVAILABILITY
    ]
    inactive_but_perfect = sorted(inactive_items, key=lambda item: item["rank"])
    # Traps that leaked up: low-relevance (tier <= 1) judged candidates inside top 50.
    traps_in_top_50 = sorted(row.candidate_id for row in judged if row.tier <= 1 and row.rank <= 50)

    return {
        "metrics": metrics,
        "challenge_weighted_score": round(
            0.50 * metrics["ndcg_at_10"]
            + 0.30 * metrics["ndcg_at_50"]
            + 0.15 * metrics["map"]
            + 0.05 * metrics["precision_at_10"],
            6,
        ),
        "relevance_cutoff": cutoff,
        "gold": {
            "labeled": len(gold),
            "judged_in_ranking": len(judged),
            "missing_from_ranking": missing,
            "missing_but_relevant": missing_relevant,
            "tier_histogram": {str(t): tiers.count(t) for t in range(6)},
        },
        "ranking": {"size": len(ranking)},
        "checks": {
            "honeypots_in_top_100": len(honeypots_top_100),
            "honeypot_ids_in_top_100": honeypots_top_100,
            "disqualifier_flagged_in_top_10": _flagged_in_top(ranking, 10),
            "disqualifier_flagged_in_top_50": _flagged_in_top(ranking, 50),
            "disqualifier_flagged_in_top_100": _flagged_in_top(ranking, 100),
            "off_target_or_flagged_in_top_50": off_target_top_50,
            "tier_ranks": {str(t): tier_ranks[t] for t in range(6)},
            "inactive_but_perfect": inactive_but_perfect,
            "traps_in_top_50": traps_in_top_50,
        },
        "config": {
            "snapshot_date": cfg.snapshot_date,
            "weights": {
                "career_sim": cfg.w_career_sim,
                "role_match": cfg.w_role_match,
                "domain_match": cfg.w_domain_match,
                "product_ratio": cfg.w_product_ratio,
                "seniority_fit": cfg.w_seniority_fit,
                "built_ranking": cfg.w_built_ranking,
                "lexical_evidence": cfg.w_lexical_evidence,
            },
        },
    }


def _flagged_in_top(ranking: list[RankedCandidate], k: int) -> int:
    return sum(1 for c in ranking[:k] if c.features.disqualifier_flags)


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def print_report(report: dict[str, Any]) -> None:
    """Print the metrics and the checks that matter, with PASS/FAIL where it counts."""
    metrics = report["metrics"]
    gold = report["gold"]
    checks = report["checks"]

    print("\n=== Gold-set evaluation ===")
    print(f"  NDCG@10 = {metrics['ndcg_at_10']:.4f}")
    print(f"  NDCG@50 = {metrics['ndcg_at_50']:.4f}")
    print(
        f"  MAP     = {metrics['map']:.4f}   (binary cutoff: tier >= {report['relevance_cutoff']})"
    )
    print(f"  P@10    = {metrics['precision_at_10']:.4f}")
    print(f"  challenge-weighted score = {report['challenge_weighted_score']:.4f}")

    print(
        f"\n  gold: {gold['labeled']} labeled, {gold['judged_in_ranking']} judged in ranking, "
        f"{len(gold['missing_from_ranking'])} not in ranking"
    )
    print(f"  tier histogram (judged): {gold['tier_histogram']}")
    if gold["missing_but_relevant"]:
        print(
            f"  ⚠ relevant gold not in ranking (pre-filter dropped a real fit): "
            f"{gold['missing_but_relevant']}"
        )

    hp = checks["honeypots_in_top_100"]
    status = "PASS" if hp == 0 else "FAIL"
    print(f"\n  [{status}] honeypots in top 100: {hp}  (must be 0)")
    if checks["honeypot_ids_in_top_100"]:
        print(f"          {checks['honeypot_ids_in_top_100']}")
    print(
        f"  disqualifier-flagged in top 10/50/100: "
        f"{checks['disqualifier_flagged_in_top_10']}/"
        f"{checks['disqualifier_flagged_in_top_50']}/"
        f"{checks['disqualifier_flagged_in_top_100']}"
    )
    tier_ranks = checks["tier_ranks"]
    print(f"  ranks of tier-5 (ideal) labels: {tier_ranks['5']}")
    print(f"  ranks of tier-4 labels:         {tier_ranks['4']}")
    if checks["inactive_but_perfect"]:
        print("  inactive-but-perfect (tier >=4, low availability) — where they ranked:")
        for item in checks["inactive_but_perfect"]:
            print(
                f"    {item['candidate_id']}  tier {item['tier']}  rank #{item['rank']}  "
                f"availability {item['availability']}"
            )
    if checks["traps_in_top_50"]:
        print(
            f"  ⚠ low-relevance (tier <=1) judged candidates inside top 50: "
            f"{checks['traps_in_top_50']}"
        )
    print()


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the report as deterministic JSON (sorted keys, no wall-clock)."""
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    gold_path: Path,
    out_path: Path,
    cfg: ScoringConfig = SCORING,
) -> dict[str, Any]:
    """Load labels, rebuild the ranking, compute the report, print + write it."""
    gold = parse_gold_labels(gold_path)
    logger.info("loaded %d gold labels from %s", len(gold), gold_path)
    ranking = build_ranking(artifacts_dir=artifacts_dir, candidates_path=candidates_path, cfg=cfg)
    report = build_report(ranking, gold, cfg=cfg)
    print_report(report)
    write_report(report, out_path)
    print(f"Wrote {out_path}\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        gold_path=args.gold,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
