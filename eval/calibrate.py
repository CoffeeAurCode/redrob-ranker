"""Session-08 weight calibration harness — sweep, stability, and ablation.

Calibration means choosing the scoring weights/thresholds deliberately against the
gold set (``eval/gold_labels.csv``) and proving the choice is **stable**, not a
fragile peak on ~66 labels. This tool exists so that work is reproducible and cheap:

* it loads the gold set **and** the precomputed artifacts **once** (the shortlist
  records, their JD-reference cosines, the LLM signals, the honeypot flags), then
* re-scores the whole shortlist **in memory** under any number of
  :class:`~src.config.ScoringConfig` variants — no re-streaming the 100k pool, no
  network, no LLM — reusing the exact rank-time core (:func:`eval.ranking.rank_candidates`)
  and the exact eval report (:func:`eval.evaluate.build_report`), so a swept config is
  scored byte-for-byte the way ``rank.py`` and ``evaluate.py`` would score it.

Three views, each a Stage-5 talking point:

* ``--sweep``     coordinate sweep over the availability floor and each base-term
                  weight (others renormalized so the seven still sum to 1.0).
* ``--stability`` perturb each weight / the floor by ±10% and ±20%; a configuration
                  is only trustworthy if the four metrics barely move.
* ``--ablation``  drop each base term (weight→0, the rest renormalized) and show the
                  metric delta — proof every term earns its place.

Run from the repo root::

    python eval/calibrate.py                 # baseline + all three views
    python eval/calibrate.py --sweep         # just the coordinate sweep
    python eval/calibrate.py --candidate     # score the proposed locked config only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluate import GoldLabel, build_report, parse_gold_labels  # noqa: E402
from eval.ranking import (  # noqa: E402
    SHORTLIST_FILE,
    RankedCandidate,
    rank_candidates,
    reference_similarities,
)
from src.config import SCORING, ScoringConfig  # noqa: E402
from src.honeypots import load_flagged_ids  # noqa: E402
from src.io_utils import Candidate, load_candidates, use_utf8_stdout  # noqa: E402
from src.llm_signals import LLM_SIGNALS_FILE, LLMSignals, load_signal_cache  # noqa: E402

logger = logging.getLogger("calibrate")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold_labels.csv"

# The seven additive base-fit term names (the ``w_*`` weights without the prefix).
BASE_TERMS: tuple[str, ...] = (
    "career_sim",
    "role_match",
    "domain_match",
    "product_ratio",
    "seniority_fit",
    "built_ranking",
    "lexical_evidence",
)

# A gold strong-hire (tier >= 4) whose baseline full rank is past this is "buried" —
# the inactive-but-perfect set we track across configs (availability over-penalizes
# genuine ideal hires; Session 07 surfaced six of them at ranks ~220-284).
_BURIED_RANK = 100
_STRONG_HIRE_TIER = 4


@dataclass(frozen=True)
class CalibrationData:
    """Everything needed to re-score the shortlist under any config, loaded once."""

    records: list[Candidate]
    sim_by_id: Mapping[str, float]
    signals: Mapping[str, LLMSignals]
    flagged: frozenset[str]
    gold: dict[str, GoldLabel]


@dataclass(frozen=True)
class Headline:
    """The four challenge metrics plus the checks calibration must not regress."""

    ndcg10: float
    ndcg50: float
    map: float
    p10: float
    weighted: float
    honeypots_top100: int
    traps_top50: int
    # Full ranks of the tracked inactive-but-perfect ids, in the same id order.
    inactive_ranks: tuple[int, ...]


def load_calibration_data(
    *, artifacts_dir: Path, candidates_path: Path, gold_path: Path
) -> CalibrationData:
    """Load the gold labels and the shortlist's scoring inputs into memory once.

    Streams the pool a single time and keeps only the shortlisted records (~1.2k),
    so every subsequent config is scored without touching disk again.
    """
    shortlist = set(json.loads((artifacts_dir / SHORTLIST_FILE).read_text(encoding="utf-8")))
    gold = parse_gold_labels(gold_path)
    sim_by_id = reference_similarities(artifacts_dir)
    signals = load_signal_cache(artifacts_dir / LLM_SIGNALS_FILE)
    flagged = load_flagged_ids(artifacts_dir)

    records = [
        candidate
        for candidate in load_candidates(candidates_path)
        if isinstance(candidate.get("candidate_id"), str)
        and candidate.get("candidate_id") in shortlist
    ]
    logger.info(
        "loaded %d shortlist records, %d gold labels, %d honeypot flags",
        len(records),
        len(gold),
        len(flagged),
    )
    return CalibrationData(records, sim_by_id, signals, flagged, gold)


def rank(data: CalibrationData, cfg: ScoringConfig) -> list[RankedCandidate]:
    """Re-score the in-memory shortlist under ``cfg`` (the rank-time core, no I/O)."""
    return rank_candidates(
        data.records,
        sim_by_id=data.sim_by_id,
        signals=data.signals,
        flagged=data.flagged,
        cfg=cfg,
    )


def evaluate(data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]) -> Headline:
    """Score ``cfg`` against the gold set and pull out the headline numbers."""
    ranking = rank(data, cfg)
    report = build_report(ranking, data.gold, cfg=cfg)
    rank_by_id = {c.candidate_id: c.rank for c in ranking}
    metrics = report["metrics"]
    checks = report["checks"]
    return Headline(
        ndcg10=metrics["ndcg_at_10"],
        ndcg50=metrics["ndcg_at_50"],
        map=metrics["map"],
        p10=metrics["precision_at_10"],
        weighted=report["challenge_weighted_score"],
        honeypots_top100=checks["honeypots_in_top_100"],
        traps_top50=len(checks["traps_in_top_50"]),
        inactive_ranks=tuple(rank_by_id.get(cid, -1) for cid in inactive_ids),
    )


# --------------------------------------------------------------------------- #
# Config algebra — vary weights while keeping the seven summed to 1.0.          #
# --------------------------------------------------------------------------- #
def base_weights(cfg: ScoringConfig) -> dict[str, float]:
    """The seven current base-term weights as a plain dict."""
    return {term: getattr(cfg, f"w_{term}") for term in BASE_TERMS}


def with_weights(cfg: ScoringConfig, weights: Mapping[str, float]) -> ScoringConfig:
    """Return ``cfg`` with the seven base weights replaced and renormalized to sum 1.0."""
    total = sum(weights[term] for term in BASE_TERMS)
    if total <= 0:
        raise ValueError("base-term weights must sum to a positive number")
    # ``Any`` keeps mypy --strict happy: ``replace``'s per-field kwargs have different
    # declared types, so a homogeneous ``dict[str, float]`` splat does not type-check.
    changes: dict[str, Any] = {f"w_{term}": weights[term] / total for term in BASE_TERMS}
    return dataclasses.replace(cfg, **changes)


def set_weight(cfg: ScoringConfig, term: str, value: float) -> ScoringConfig:
    """Set one base term to ``value`` and scale the *other* six to keep the sum at 1.0.

    Scaling the others proportionally (rather than renormalizing everything) keeps a
    coordinate sweep honest: only the swept term's *share* changes, the remaining mass
    keeps its internal ratios. ``value=0.0`` is the ablation of ``term``.
    """
    others = {t: w for t, w in base_weights(cfg).items() if t != term}
    remaining = max(0.0, 1.0 - value)
    others_total = sum(others.values())
    scaled = (
        {t: w / others_total * remaining for t, w in others.items()}
        if others_total > 0
        else {t: remaining / len(others) for t in others}
    )
    scaled[term] = value
    return with_weights(cfg, scaled)


# --------------------------------------------------------------------------- #
# Reporting.                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(h: Headline) -> str:
    inactive = ",".join(str(r) for r in h.inactive_ranks)
    worst = max(h.inactive_ranks) if h.inactive_ranks else 0
    return (
        f"ndcg@10={h.ndcg10:.4f}  ndcg@50={h.ndcg50:.4f}  map={h.map:.4f}  "
        f"p@10={h.p10:.4f}  weighted={h.weighted:.4f}  "
        f"hp@100={h.honeypots_top100}  traps@50={h.traps_top50}  "
        f"inactive_worst_rank={worst}  [{inactive}]"
    )


def print_baseline(
    data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]
) -> None:
    print("\n=== Baseline (current ScoringConfig) ===")
    print(f"  availability_floor={cfg.availability_floor}  weights={base_weights(cfg)}")
    print(f"  tracked inactive-but-perfect ids: {list(inactive_ids)}")
    print("  " + _fmt(evaluate(data, cfg, inactive_ids)))


def print_floor_sweep(
    data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]
) -> None:
    print("\n=== availability_floor sweep (the main NDCG@50 lever) ===")
    for floor in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        swept = dataclasses.replace(cfg, availability_floor=floor)
        print(f"  floor={floor:.2f}  " + _fmt(evaluate(data, swept, inactive_ids)))


def print_weight_sweep(
    data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]
) -> None:
    print("\n=== per-weight coordinate sweep (others renormalized to keep Σ=1) ===")
    for term in BASE_TERMS:
        current = getattr(cfg, f"w_{term}")
        print(f"  -- {term} (current {current:.2f}) --")
        for value in sorted(
            {round(max(0.0, current + d), 3) for d in (-0.08, -0.04, 0.0, 0.04, 0.08)}
        ):
            swept = set_weight(cfg, term, value)
            print(f"     {term}={value:.3f}  " + _fmt(evaluate(data, swept, inactive_ids)))


def print_stability(
    data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]
) -> None:
    print("\n=== stability: perturb each weight / floor by ±10% and ±20% ===")
    baseline = evaluate(data, cfg, inactive_ids)
    print("  baseline      " + _fmt(baseline))
    worst_swing = 0.0
    for term in BASE_TERMS:
        current = getattr(cfg, f"w_{term}")
        for pct in (-0.20, -0.10, 0.10, 0.20):
            swept = set_weight(cfg, term, max(0.0, current * (1 + pct)))
            head = evaluate(data, swept, inactive_ids)
            swing = abs(head.weighted - baseline.weighted)
            worst_swing = max(worst_swing, swing)
            print(f"  {term:<16}{pct:+.0%}  " + _fmt(head))
    for pct in (-0.20, -0.10, 0.10, 0.20):
        swept = dataclasses.replace(cfg, availability_floor=cfg.availability_floor * (1 + pct))
        head = evaluate(data, swept, inactive_ids)
        worst_swing = max(worst_swing, abs(head.weighted - baseline.weighted))
        print(f"  {'avail_floor':<16}{pct:+.0%}  " + _fmt(head))
    print(f"\n  worst weighted-score swing under ±20% = {worst_swing:.4f}")


def print_ablation(
    data: CalibrationData, cfg: ScoringConfig, inactive_ids: tuple[str, ...]
) -> None:
    print("\n=== ablation: drop each base term (weight→0, others renormalized) ===")
    baseline = evaluate(data, cfg, inactive_ids)
    print("  full model    " + _fmt(baseline))
    for term in BASE_TERMS:
        ablated = set_weight(cfg, term, 0.0)
        head = evaluate(data, ablated, inactive_ids)
        delta = head.weighted - baseline.weighted
        print(f"  -{term:<15} Δweighted={delta:+.4f}  " + _fmt(head))


def tracked_inactive_ids(data: CalibrationData) -> tuple[str, ...]:
    """The strong-hire (tier >= 4) gold ids that the *baseline* config buries past top-100.

    Tracking this fixed id set (not a per-config availability threshold) lets the sweep
    show the ids being lifted as the availability floor softens — a threshold-based set
    would empty out the moment the floor raises their availability, hiding the effect.
    """
    baseline_ranks = {c.candidate_id: c.rank for c in rank(data, SCORING)}
    buried = [
        cid
        for cid, label in data.gold.items()
        if label.tier >= _STRONG_HIRE_TIER and baseline_ranks.get(cid, 10**9) > _BURIED_RANK
    ]
    return tuple(sorted(buried, key=lambda cid: baseline_ranks[cid]))


def candidate_config(cfg: ScoringConfig = SCORING) -> ScoringConfig:
    """The proposed locked configuration this session arrives at (see calibration_report.md).

    Single, principled change from the Session-06 defaults: soften the availability
    multiplier (floor 0.5 → 0.7) so a genuine ideal hire who is merely less active is
    *down-weighted, not buried*. The seven fit weights are left at their interpretable
    Session-06 values — the sweep shows the metrics are already at a stable plateau in
    the weights, so moving them would be overfitting ~66 labels.
    """
    return dataclasses.replace(cfg, availability_floor=0.70)


def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    gold_path: Path,
    views: Iterable[str],
) -> None:
    data = load_calibration_data(
        artifacts_dir=artifacts_dir, candidates_path=candidates_path, gold_path=gold_path
    )
    inactive_ids = tracked_inactive_ids(data)
    print_baseline(data, SCORING, inactive_ids)
    views = set(views)
    if "sweep" in views:
        print_floor_sweep(data, SCORING, inactive_ids)
        print_weight_sweep(data, SCORING, inactive_ids)
    if "stability" in views:
        print_stability(data, candidate_config(), inactive_ids)
    if "ablation" in views:
        print_ablation(data, candidate_config(), inactive_ids)
    if "candidate" in views:
        print("\n=== Proposed locked config (availability_floor 0.5 → 0.7) ===")
        cand = candidate_config()
        print(f"  availability_floor={cand.availability_floor}")
        print("  " + _fmt(evaluate(data, cand, inactive_ids)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--sweep", action="store_true", help="coordinate sweep only")
    parser.add_argument("--stability", action="store_true", help="±10/20%% perturbation only")
    parser.add_argument("--ablation", action="store_true", help="drop-one-term table only")
    parser.add_argument("--candidate", action="store_true", help="score the proposed config only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    selected = [
        name for name in ("sweep", "stability", "ablation", "candidate") if getattr(args, name)
    ]
    views = selected or ["sweep", "stability", "ablation", "candidate"]
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        gold_path=args.gold,
        views=views,
    )


if __name__ == "__main__":
    main()
