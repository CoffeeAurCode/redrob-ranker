"""Precompute 03 — batched, cached LLM structured extraction over the shortlist.

OFFLINE step (the golden rule: this never runs inside ``rank.py`` — it is the
heaviest LLM user in the project and its *only* handoff to rank time is the cached
``artifacts/llm_signals.jsonl``). The pipeline is:

    shortlist_ids.json ─► (skip already-cached) ─► batch N profiles per Gemini
    call ─► defensive parse/clamp ─► append to llm_signals.jsonl (crash-safe)

All reusable, testable logic lives in ``src.llm_signals``; this file is the thin
CLI/orchestration layer — the only part that touches the network and the model.

The backend is pluggable (``--provider``) so we are not pinned to one free tier's
daily cap. Default is **Groq** (Llama-3.3-70B, ~1k free req/day) since Gemini-flash
turned out to allow only ~20 req/day on our key; the JD-rubric step (Session 03)
still used Gemini. Each provider's key is read from its own ``.env`` var.

Run from the repo root (needs the chosen provider's key in ``.env``)::

    python src/precompute/03_extract_signals.py                  # Groq (default), uncached only
    python src/precompute/03_extract_signals.py --provider gemini --limit 16   # Gemini smoke test
    python src/precompute/03_extract_signals.py --dry-run        # print one batch prompt, no call
    python src/precompute/03_extract_signals.py --report         # distributions + spot-checks

Re-running after a full pass makes **zero** API calls (everything is cached).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import load_candidates, use_utf8_stdout  # noqa: E402
from src.jd_reference import load_jd_reference  # noqa: E402
from src.llm_providers import (  # noqa: E402
    DEFAULT_PROVIDER,
    PROVIDERS,
    make_call_fn,
    require_api_key,
)
from src.llm_signals import (  # noqa: E402
    FAILURES_FILE,
    LLM_SIGNALS_FILE,
    LLMSignals,
    append_signals,
    build_batch_prompt,
    build_rubric_summary,
    extract_signals,
    load_signal_cache,
    pending_ids,
)
from src.profile_text import build_llm_profile  # noqa: E402

logger = logging.getLogger("extract_signals")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
SHORTLIST_FILE = "shortlist_ids.json"
DEFAULT_BATCH_SIZE = 8
# Gentle pacing between calls so bursts stay under each free tier's RPM/TPM. A
# call already takes tens of seconds, so this mainly guards the rare fast reply.
DEFAULT_SLEEP_SECONDS = 6.0
DEFAULT_SPOT_CHECK = 10

# The provider layer (Provider/PROVIDERS/make_call_fn/require_api_key) is shared
# with Session 09's reasoning CLI and lives in ``src/llm_providers.py``.


def load_profiles(candidates_path: Path, wanted: set[str]) -> dict[str, str]:
    """Stream the pool once and build a compact LLM profile for each wanted id.

    Only the ~1-3k shortlist profiles are held in memory (the 100k pool is
    streamed, never slurped). Stops early once every wanted id is found.
    """
    profiles: dict[str, str] = {}
    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id")
        if cid in wanted and cid not in profiles:
            profiles[cid] = build_llm_profile(candidate)
            if len(profiles) == len(wanted):
                break
    return profiles


def report(cache: dict[str, LLMSignals], *, spot_check: int) -> None:
    """Print archetype/domain/flag distributions + a spot-check sample (no network).

    The Phase-2 quality gate: distributions should be plausible (not e.g. 90%
    ``non_tech``), and the spot-checks let a human verify archetype/domain/flag
    calls against the raw profiles.
    """
    if not cache:
        print("\n(cache is empty — run the extraction first)\n")
        return

    n = len(cache)
    print(f"\n=== Coverage: {n} cached extractions ===")
    _histogram("role_archetype", (s["role_archetype"] for s in cache.values()), n)
    _histogram("domain", (s["domain"] for s in cache.values()), n)

    flag_counts: Counter[str] = Counter(
        flag for s in cache.values() for flag in s["disqualifier_flags"]
    )
    built = sum(1 for s in cache.values() if s["built_ranking_or_search"])
    print(f"\nbuilt_ranking_or_search = true: {built}/{n} ({built / n:.0%})")
    print("disqualifier_flags:")
    if flag_counts:
        for flag, count in flag_counts.most_common():
            print(f"  {flag:<22} {count:>5}  ({count / n:.0%})")
    else:
        print("  (none)")

    print(f"\n=== Spot-check (first {spot_check} by id) ===")
    for cid in sorted(cache)[:spot_check]:
        s = cache[cid]
        flags = ", ".join(s["disqualifier_flags"]) or "—"
        print(
            f"\n{cid}  [{s['role_archetype']} / {s['domain']}]  "
            f"prod={s['product_vs_services']:.2f} sen={s['seniority_band_fit']:.2f} "
            f"built={s['built_ranking_or_search']}  flags=[{flags}]"
        )
        print(f"  evidence: {s['evidence_span']}")
    print()


def _histogram(label: str, values: Iterable[str], total: int) -> None:
    """Print a count/percent histogram for one categorical field."""
    counts: Counter[str] = Counter(values)
    print(f"\n{label}:")
    for key, count in counts.most_common():
        bar = "█" * round(40 * count / total)
        print(f"  {key:<16} {count:>5}  {bar}")


def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    provider_name: str,
    model: str | None,
    batch_size: int,
    limit: int | None,
    sleep_seconds: float,
    spot_check: int,
    dry_run: bool,
    report_only: bool,
) -> None:
    """Orchestrate the extraction: cache-first, build profiles, batch-call, append."""
    cache = load_signal_cache(artifacts_dir / LLM_SIGNALS_FILE)

    if report_only:
        report(cache, spot_check=spot_check)
        return

    shortlist = json.loads((artifacts_dir / SHORTLIST_FILE).read_text(encoding="utf-8"))
    rubric_summary = build_rubric_summary(load_jd_reference(artifacts_dir)["rubric"])

    pending = pending_ids(shortlist, cache)
    logger.info("shortlist=%d, cached=%d, pending=%d", len(shortlist), len(cache), len(pending))
    if limit is not None:
        pending = pending[:limit]
        logger.info("limited to first %d pending", len(pending))
    if not pending:
        logger.info("nothing to do — shortlist fully cached (zero API calls)")
        report(cache, spot_check=spot_check)
        return

    profiles = load_profiles(candidates_path, set(pending))
    if missing := [cid for cid in pending if cid not in profiles]:
        logger.warning(
            "%d shortlist ids not found in the pool (skipped): %s",
            len(missing),
            ", ".join(missing[:5]),
        )
        profiles = {cid: profiles[cid] for cid in pending if cid in profiles}

    if dry_run:
        first = list(profiles.items())[:batch_size]
        print(build_batch_prompt(rubric_summary, first))
        return

    provider = PROVIDERS[provider_name]
    model_name = model or provider.default_model
    out_path = artifacts_dir / LLM_SIGNALS_FILE
    call_fn = make_call_fn(
        provider=provider,
        model_name=model_name,
        api_key=require_api_key(provider.env_key, env_path=REPO_ROOT / ".env"),
        label=provider_name,
    )

    logger.info(
        "extracting %d candidates in batches of %d via %s/%s ...",
        len(profiles),
        batch_size,
        provider_name,
        model_name,
    )
    new_count, failed = extract_signals(
        profiles=profiles,
        rubric_summary=rubric_summary,
        call_fn=call_fn,
        on_results=lambda records: append_signals(out_path, records),
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )
    logger.info("wrote %d new records to %s", new_count, out_path.name)
    if failed:
        fail_path = artifacts_dir / FAILURES_FILE
        fail_path.write_text(json.dumps(sorted(failed), indent=2) + "\n", encoding="utf-8")
        logger.warning("%d ids failed after fallback — logged to %s", len(failed), fail_path.name)

    report(load_signal_cache(out_path), spot_check=spot_check)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=DEFAULT_PROVIDER,
        help="LLM backend (free tier); key read from its .env var",
    )
    parser.add_argument(
        "--model", default=None, help="model id (defaults to the provider's default model)"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pending")
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="seconds to pace between calls (free-tier rate limit)",
    )
    parser.add_argument("--spot-check", type=int, default=DEFAULT_SPOT_CHECK)
    parser.add_argument("--dry-run", action="store_true", help="print one batch prompt and exit")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print distributions + spot-checks from the cache and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    for name in ("httpx", "httpcore", "urllib3", "google", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # google.generativeai is sunset and prints a FutureWarning on import; it still
    # works for precompute (see plan/PROGRESS.md). Silence it to keep run logs clean.
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        provider_name=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        sleep_seconds=args.sleep,
        spot_check=args.spot_check,
        dry_run=args.dry_run,
        report_only=args.report,
    )


if __name__ == "__main__":
    main()
