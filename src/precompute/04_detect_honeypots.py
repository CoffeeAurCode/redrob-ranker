"""Precompute 04 — deterministic honeypot detection over the full candidate pool.

OFFLINE step, but unlike Sessions 03/04 it uses **no LLM and no network** — the
rules are pure structural impossibilities (see ``src.honeypots``). It streams the
100k pool, applies every rule, and writes ``artifacts/honeypot_flags.json``::

    { "CAND_xxxxxxx": {"honeypot": true, "reasons": ["expert_zero_duration"]}, ... }

Only flagged candidates are stored (absence == not a honeypot), keyed by id and
sorted, so the artifact is tiny and trivial for ``scoring.py``/``rank.py`` to load:
a flagged id is forced to ``final = 0``. This is the single signal that *zeroes* a
candidate, so the rules favour precision (see ``src.honeypots`` and ``docs/schema.md``).

Run from the repo root::

    python src/precompute/04_detect_honeypots.py            # scan pool, write artifact
    python src/precompute/04_detect_honeypots.py --report   # re-print the existing artifact

The scan is deterministic, so re-running reproduces a byte-identical artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.honeypots import (  # noqa: E402
    HONEYPOT_FLAGS_FILE,
    HoneypotFlag,
    detect_honeypot,
)
from src.io_utils import load_candidates, use_utf8_stdout  # noqa: E402

logger = logging.getLogger("detect_honeypots")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
SHORTLIST_FILE = "shortlist_ids.json"


def scan_pool(candidates_path: Path) -> dict[str, HoneypotFlag]:
    """Stream the pool and return ``{candidate_id: HoneypotFlag}`` for flagged ids only.

    Streams one record at a time (never slurps the 465 MB pool) and keeps only the
    handful of impossible profiles. The result is rebuilt in ``candidate_id`` order
    by :func:`write_flags` so the artifact is deterministic.
    """
    flags: dict[str, HoneypotFlag] = {}
    total = 0
    for candidate in load_candidates(candidates_path):
        total += 1
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str):
            continue
        flag = detect_honeypot(candidate)
        if flag is not None:
            flags[cid] = flag
    logger.info("scanned %d candidates -- %d flagged", total, len(flags))
    return flags


def write_flags(flags: dict[str, HoneypotFlag], out_path: Path) -> None:
    """Write the flags artifact sorted by candidate_id (deterministic on-disk order)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {cid: flags[cid] for cid in sorted(flags)}
    out_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %d flags to %s", len(ordered), out_path.name)


def report(flags: dict[str, HoneypotFlag], artifacts_dir: Path) -> None:
    """Print per-reason counts, a few examples, and overlap with the shortlist."""
    if not flags:
        print("\n(no honeypot flags — run the scan first, or the pool is clean)\n")
        return

    reason_counts: Counter[str] = Counter(
        reason for flag in flags.values() for reason in flag["reasons"]
    )
    print(f"\n=== Honeypots flagged: {len(flags)} ===")
    print("by reason (a candidate may trip more than one):")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<26} {count:>4}")

    shortlist_path = artifacts_dir / SHORTLIST_FILE
    if shortlist_path.exists():
        shortlist = set(json.loads(shortlist_path.read_text(encoding="utf-8")))
        in_shortlist = sorted(cid for cid in flags if cid in shortlist)
        print(
            f"\nin shortlist (could reach scoring): {len(in_shortlist)} / {len(shortlist)}"
            + (f" -> {', '.join(in_shortlist)}" if in_shortlist else "")
        )

    print("\n=== Examples (first 5 by id) ===")
    for cid in sorted(flags)[:5]:
        print(f"  {cid}  reasons={flags[cid]['reasons']}")
    print()


def run(*, artifacts_dir: Path, candidates_path: Path, report_only: bool) -> None:
    """Scan the pool (or re-load the artifact) and print the report."""
    out_path = artifacts_dir / HONEYPOT_FLAGS_FILE

    if report_only:
        if not out_path.exists():
            raise SystemExit(f"{out_path} not found — run the scan first (without --report).")
        flags: dict[str, HoneypotFlag] = json.loads(out_path.read_text(encoding="utf-8"))
        report(flags, artifacts_dir)
        return

    flags = scan_pool(candidates_path)
    write_flags(flags, out_path)
    report(flags, artifacts_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the existing artifact's summary and exit (no scan)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        report_only=args.report,
    )


if __name__ == "__main__":
    main()
