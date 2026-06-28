"""Precompute 05 — generate a grounded reasoning for each of the final top 100.

OFFLINE step (the golden rule: this never runs inside ``rank.py`` — its only handoff
to rank time is the cached ``artifacts/reasoning.jsonl``, which ``rank.py`` joins by
``candidate_id``). The pipeline is:

    rank the shortlist (shared eval.ranking path) ─► take the current top 100 ─►
    distil each into grounded facts ─► batch an LLM for 1-2 sentence reasonings ─►
    VALIDATE each against the candidate's real data (catch hallucinations) ─►
    regenerate or fall back to a deterministic grounded reasoning ─► append JSONL

The top 100 is computed from the *same* scorer ``rank.py`` uses (``build_ranking``),
so the reasoned set matches the shipped CSV. Reasoning must be regenerated **after**
the final calibration (Session 08) — if the weights change, the top 100 changes.

All reusable, testable logic lives in ``src.reasoning``; this file is the thin
CLI/orchestration layer (the only part that touches the network and the pool). The
provider layer is shared with Session 04 via ``src.llm_providers``.

Run from the repo root (needs the chosen provider's key in ``.env``)::

    python src/precompute/05_generate_reasoning.py                 # cerebras, uncached only
    python src/precompute/05_generate_reasoning.py --limit 6       # smoke a few
    python src/precompute/05_generate_reasoning.py --deterministic-only  # no network
    python src/precompute/05_generate_reasoning.py --dry-run       # print one prompt
    python src/precompute/05_generate_reasoning.py --report        # validate the artifact

Re-running after a full pass makes **zero** API calls (everything is cached).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.ranking import build_ranking  # noqa: E402
from src.io_utils import parse_json_safe, use_utf8_stdout  # noqa: E402
from src.jd_reference import load_jd_reference  # noqa: E402
from src.llm_providers import (  # noqa: E402
    DEFAULT_PROVIDER,
    PROVIDERS,
    make_call_fn,
    require_api_key,
)
from src.llm_signals import build_rubric_summary, chunk  # noqa: E402
from src.reasoning import (  # noqa: E402
    REASONING_FILE,
    ReasoningFacts,
    ReasoningRecord,
    append_reasoning,
    build_facts,
    build_reasoning_prompt,
    deterministic_reasoning,
    gap_notes,
    load_reasoning_cache,
    parse_reasoning_batch,
    too_similar,
    validate_reasoning,
)

logger = logging.getLogger("generate_reasoning")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_TOP = 100
DEFAULT_BATCH_SIZE = 6  # small so the model never crosses facts between candidates
DEFAULT_SLEEP_SECONDS = 4.0
DEFAULT_SAMPLE = 8


def compute_top_facts(
    *, artifacts_dir: Path, candidates_path: Path, top_n: int
) -> list[ReasoningFacts]:
    """Rank the shortlist with the shipped scorer and distil the top ``top_n`` to facts.

    Uses the same :func:`eval.ranking.build_ranking` path ``rank.py`` mirrors, so the
    reasoned set is exactly the prospective top 100 of the submission.
    """
    ranked = build_ranking(artifacts_dir=artifacts_dir, candidates_path=candidates_path)
    return [build_facts(rc.candidate, rc.signals, rc.result, rc.rank) for rc in ranked[:top_n]]


# --------------------------------------------------------------------------- #
# Generation: batch-call, validate, regenerate, fall back.                      #
# --------------------------------------------------------------------------- #
def generate(
    *,
    facts_list: Sequence[ReasoningFacts],
    jd_summary: str,
    call_fn: Callable[[str], str] | None,
    out_path: Path,
    batch_size: int,
    sleep_seconds: float,
) -> dict[str, int]:
    """Generate + validate a reasoning for every uncached candidate, appending JSONL.

    Returns counts ``{"llm", "regenerated", "deterministic", "cached"}``. A reasoning
    that fails the grounding validator (or reads as a near-duplicate of an accepted
    one) is regenerated once per-candidate; if it still fails, the deterministic
    grounded reasoning is used so no row ever ships ungrounded or empty.

    With ``call_fn=None`` (``--deterministic-only``) every reasoning is the
    deterministic fallback — a fully offline way to (re)build a valid artifact.
    """
    cache = load_reasoning_cache(out_path)
    accepted: list[str] = [record["reasoning"] for record in cache.values()]
    pending = [facts for facts in facts_list if facts.candidate_id not in cache]
    counts = {"llm": 0, "regenerated": 0, "deterministic": 0, "cached": len(cache)}
    logger.info("top=%d, cached=%d, pending=%d", len(facts_list), len(cache), len(pending))

    for batch in chunk([f.candidate_id for f in pending], batch_size):
        batch_facts = [f for f in pending if f.candidate_id in set(batch)]
        parsed = _call_and_parse(batch_facts, jd_summary, call_fn) if call_fn else {}
        if call_fn and sleep_seconds > 0:
            import time

            time.sleep(sleep_seconds)  # gentle pacing between calls (free-tier rate limit)

        records: list[ReasoningRecord] = []
        for facts in batch_facts:
            record = _finalize(
                facts,
                parsed.get(facts.candidate_id, ""),
                accepted=accepted,
                jd_summary=jd_summary,
                call_fn=call_fn,
                counts=counts,
            )
            records.append(record)
            accepted.append(record["reasoning"])
        append_reasoning(out_path, records)

    return counts


def _finalize(
    facts: ReasoningFacts,
    text: str,
    *,
    accepted: Sequence[str],
    jd_summary: str,
    call_fn: Callable[[str], str] | None,
    counts: dict[str, int],
) -> ReasoningRecord:
    """Accept a grounded reasoning, regenerate a failing one once, else fall back."""
    if _is_good(text, facts, accepted):
        counts["llm"] += 1
        return {"candidate_id": facts.candidate_id, "reasoning": text, "source": "llm"}

    if text:
        issues = validate_reasoning(text, facts).issues or ("too similar to another reasoning",)
        logger.info("regenerating %s -- %s", facts.candidate_id, "; ".join(issues))

    if call_fn is not None:
        retry = _call_and_parse([facts], jd_summary, call_fn).get(facts.candidate_id, "")
        if _is_good(retry, facts, accepted):
            counts["regenerated"] += 1
            return {"candidate_id": facts.candidate_id, "reasoning": retry, "source": "llm"}

    counts["deterministic"] += 1
    return {
        "candidate_id": facts.candidate_id,
        "reasoning": deterministic_reasoning(facts),
        "source": "deterministic",
    }


def _is_good(text: str, facts: ReasoningFacts, accepted: Sequence[str]) -> bool:
    """True if ``text`` is grounded, well-formed, and not a near-duplicate."""
    return bool(text) and validate_reasoning(text, facts).ok and not too_similar(text, accepted)


def _call_and_parse(
    facts_batch: Sequence[ReasoningFacts], jd_summary: str, call_fn: Callable[[str], str]
) -> dict[str, str]:
    """Call the model for one batch and parse ``{candidate_id: reasoning}`` (empty on error)."""
    prompt = build_reasoning_prompt(jd_summary, facts_batch)
    ids = [f.candidate_id for f in facts_batch]
    try:
        return parse_reasoning_batch(parse_json_safe(call_fn(prompt)), ids)
    except Exception as exc:  # provider/JSON errors → treat as missing, caller falls back
        logger.warning("batch [%s..%s] failed: %s", ids[0], ids[-1], exc)
        return {}


# --------------------------------------------------------------------------- #
# Reporting / validation of the finished artifact.                              #
# --------------------------------------------------------------------------- #
def report(facts_list: Sequence[ReasoningFacts], out_path: Path, *, sample: int) -> None:
    """Validate every top-100 reasoning against its candidate and print the verdict.

    The Session-09 quality gate: confirms full coverage, that the grounding validator
    passes on the saved text (no hallucinations slipped through), the llm/deterministic
    split, that gap-bearing candidates disclose a gap, and shows a sample.
    """
    cache = load_reasoning_cache(out_path)
    missing = [f.candidate_id for f in facts_list if f.candidate_id not in cache]
    failures: list[tuple[str, tuple[str, ...]]] = []
    sources = {"llm": 0, "deterministic": 0}
    for facts in facts_list:
        record = cache.get(facts.candidate_id)
        if record is None:
            continue
        sources[record["source"]] = sources.get(record["source"], 0) + 1
        verdict = validate_reasoning(record["reasoning"], facts)
        if not verdict.ok:
            failures.append((facts.candidate_id, verdict.issues))

    print(f"\n=== Reasoning coverage: {len(cache)}/{len(facts_list)} top candidates ===")
    print(f"  missing reasonings:   {len(missing)}  (must be 0)")
    print(f"  source: llm={sources.get('llm', 0)}  deterministic={sources.get('deterministic', 0)}")
    print(f"  grounding-validator failures: {len(failures)}  (must be 0)")
    for cid, issues in failures[:10]:
        print(f"    {cid}: {'; '.join(issues)}")

    dup = _duplicate_count(cache)
    print(f"  exact-duplicate reasonings: {dup}  (lower is better)")
    _report_gap_honesty(facts_list, cache)

    print(f"\n=== Sample (first {sample} by rank) ===")
    for facts in facts_list[:sample]:
        record = cache.get(facts.candidate_id)
        if record is None:
            continue
        print(f"\n#{facts.rank} {facts.candidate_id}  [{record['source']}]  {facts.title}")
        print(f"  {record['reasoning']}")
    print()


def _duplicate_count(cache: dict[str, ReasoningRecord]) -> int:
    """How many reasonings are an exact (whitespace-normalized) repeat of another."""
    seen: set[str] = set()
    dup = 0
    for record in cache.values():
        key = record["reasoning"].strip().lower()
        if key in seen:
            dup += 1
        seen.add(key)
    return dup


def _report_gap_honesty(
    facts_list: Sequence[ReasoningFacts], cache: dict[str, ReasoningRecord]
) -> None:
    """Of candidates that *have* a gap, how many name one in the reasoning."""
    with_gap = [f for f in facts_list if gap_notes(f)]
    disclosed = sum(1 for f in with_gap if _mentions_gap(cache.get(f.candidate_id), f))
    print(f"  candidates with a real gap: {len(with_gap)}  (disclosed in reasoning: {disclosed})")


def _mentions_gap(record: ReasoningRecord | None, facts: ReasoningFacts) -> bool:
    """Heuristic: does the reasoning mention notice/services/research/activity when relevant."""
    if record is None:
        return False
    text = record["reasoning"].lower()
    cues = (
        "notice",
        "services",
        "consult",
        "research",
        "vision",
        "activity",
        "recent",
        "hop",
        "tenure",
        "open to work",
        "engaged",
        "available",
        "framework",
    )
    return any(cue in text for cue in cues)


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #
def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    provider_name: str,
    model: str | None,
    top_n: int,
    limit: int | None,
    batch_size: int,
    sleep_seconds: float,
    sample: int,
    dry_run: bool,
    report_only: bool,
    deterministic_only: bool,
) -> None:
    """Compute the top facts, then generate / report depending on the flags."""
    out_path = artifacts_dir / REASONING_FILE
    jd_summary = build_rubric_summary(load_jd_reference(artifacts_dir)["rubric"])
    logger.info("computing the current top %d from the calibrated scorer ...", top_n)
    facts_list = compute_top_facts(
        artifacts_dir=artifacts_dir, candidates_path=candidates_path, top_n=top_n
    )

    if report_only:
        report(facts_list, out_path, sample=sample)
        return

    if dry_run:
        first = facts_list[:batch_size]
        print(build_reasoning_prompt(jd_summary, first))
        return

    if limit is not None:
        facts_list = facts_list[:limit]
        logger.info("limited to the first %d", len(facts_list))

    call_fn = None
    if not deterministic_only:
        provider = PROVIDERS[provider_name]
        model_name = model or provider.default_model
        logger.info(
            "generating via %s/%s in batches of %d ...", provider_name, model_name, batch_size
        )
        call_fn = make_call_fn(
            provider=provider,
            model_name=model_name,
            api_key=require_api_key(provider.env_key, env_path=REPO_ROOT / ".env"),
            label=provider_name,
        )
    else:
        logger.info("deterministic-only: building grounded reasonings with no network")

    counts = generate(
        facts_list=facts_list,
        jd_summary=jd_summary,
        call_fn=call_fn,
        out_path=out_path,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )
    logger.info(
        "done -- llm=%d regenerated=%d deterministic=%d (cached=%d)",
        counts["llm"],
        counts["regenerated"],
        counts["deterministic"],
        counts["cached"],
    )
    report(facts_list, out_path, sample=sample)


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
    parser.add_argument("--model", default=None, help="model id (default: the provider's default)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="reason over the top N")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pending")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="skip the LLM; build the deterministic grounded reasoning for every row",
    )
    parser.add_argument("--dry-run", action="store_true", help="print one batch prompt and exit")
    parser.add_argument(
        "--report", action="store_true", help="validate the saved artifact and exit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    for name in ("httpx", "httpcore", "urllib3", "google", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        provider_name=args.provider,
        model=args.model,
        top_n=args.top,
        limit=args.limit,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
        sample=args.sample,
        dry_run=args.dry_run,
        report_only=args.report,
        deterministic_only=args.deterministic_only,
    )


if __name__ == "__main__":
    main()
