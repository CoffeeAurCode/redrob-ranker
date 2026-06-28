"""Build the human review sheet for the gold set (Session 07, Part A).

The gold set is the single highest-value hour of the build, so this tool does the
*mechanical* work — pick a deliberately diverse, stratified sample of the ranked
shortlist and lay each candidate out for a human to judge — and leaves the judgment
itself to a person. It emits:

* ``eval/gold_review.csv`` — one row per candidate with id, title, the career
  excerpts, the LLM signals, the honeypot flag, the full score breakdown, a
  **suggested** tier + rationale, and a blank ``tier`` / ``note`` for the reviewer.
* ``eval/gold_review.md`` — the same set as readable cards.

The sample is chosen to span the categories that actually test a ranker (clear
fits, plausible-but-flawed, keyword stuffers, CV-primary bait, consulting-only,
honeypots, **inactive-but-perfect**) across the score distribution — see
:func:`select_gold_candidates`. Selection and the suggested tier are pure and
deterministic (stdlib only), so the sheet is byte-stable on re-run and both are
unit-tested. The ``suggested_tier`` is Claude's draft to **overwrite**, never the
final label (see ``eval/LABELING_GUIDE.md``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.ranking import RankedCandidate, build_ranking  # noqa: E402
from src.config import SCORING  # noqa: E402
from src.features import is_archetype_title  # noqa: E402
from src.honeypots import honeypot_reasons, load_flagged_ids  # noqa: E402
from src.io_utils import Candidate, load_candidates, use_utf8_stdout  # noqa: E402

logger = logging.getLogger("label_helper")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval"

# Disqualifier flags the JD names as hard exclusions (cap the suggested tier at 2)
# vs. soft signals (nudge one tier down). Keys are from llm_signals.DISQUALIFIER_FLAGS.
_HARD_FLAGS = frozenset({"cv_primary", "consulting_only", "pure_research", "langchain_only_recent"})
_SOFT_FLAGS = frozenset({"stale_coding", "job_hopper"})


@dataclass(frozen=True)
class GoldPick:
    """A candidate chosen for the gold set, with why it was included.

    ``rank`` is the candidate's 1-based position in the system ranking, or ``None``
    for a pool honeypot deliberately included from outside the ranked shortlist.
    """

    candidate_id: str
    category: str
    rank: int | None
    reason: str


# --------------------------------------------------------------------------- #
# Selection — deterministic, stratified across the categories that test a ranker.
# --------------------------------------------------------------------------- #
def select_gold_candidates(
    ranking: list[RankedCandidate], *, max_total: int = 90
) -> list[GoldPick]:
    """Pick a diverse, deterministic sample of the ranked shortlist to hand-label.

    Walks an ordered set of category selectors (clear fits → inactive-but-perfect →
    CV bait → consulting-only → job-hoppers/stale → semantic mismatches → title
    surprises → mid/low band → the in-ranking honeypot → the very bottom), taking a
    fixed quota from each over the already-sorted ranking and de-duplicating by id
    (first category wins). Pure and order-stable; capped at ``max_total``.
    """
    n = len(ranking)
    if n == 0:
        return []
    avail_floor = _quantile(sorted(c.features.availability for c in ranking), 0.20)

    picks: list[GoldPick] = []
    seen: set[str] = set()

    def take(
        category: str,
        items: list[RankedCandidate],
        quota: int,
        reason: str | Callable[[RankedCandidate], str],
    ) -> None:
        added = 0
        for c in items:
            if added >= quota or len(picks) >= max_total:
                break
            if c.candidate_id in seen:
                continue
            seen.add(c.candidate_id)
            note = reason(c) if callable(reason) else reason
            picks.append(GoldPick(c.candidate_id, category, c.rank, note))
            added += 1

    by_rank = ranking  # already sorted by final desc, id asc

    take("clear_fit", by_rank, 16, "top of the ranking — should be a clear fit")
    take(
        "inactive_but_perfect",
        sorted(
            (
                c
                for c in by_rank
                if c.result.base_fit >= 0.85 and c.features.availability <= avail_floor
            ),
            key=lambda c: c.features.availability,
        ),
        6,
        lambda c: f"strong fit (base={c.result.base_fit:.2f}) but low availability "
        f"({c.features.availability:.2f}) — ranker must still surface them",
    )
    take(
        "cv_bait",
        [c for c in by_rank if "cv_primary" in c.features.disqualifier_flags],
        6,
        "CV/speech-primary bait the JD excludes (highest-ranked first = most dangerous)",
    )
    take(
        "consulting_only",
        [c for c in by_rank if "consulting_only" in c.features.disqualifier_flags],
        4,
        "consulting/services-only career the JD excludes",
    )
    take(
        "job_hopper",
        [c for c in by_rank if "job_hopper" in c.features.disqualifier_flags],
        3,
        "job-hopper flag",
    )
    take(
        "stale_coding",
        [c for c in by_rank if "stale_coding" in c.features.disqualifier_flags],
        2,
        "stale-coding flag",
    )
    take(
        "semantic_mismatch",
        [c for c in by_rank if c.features.career_sim >= 0.80 and c.features.role_match <= 0.30],
        4,
        "embedding says relevant but the role/domain is off (CV/speech) — must NOT float up",
    )
    take(
        "title_surprise",
        [c for c in by_rank if c.rank <= 320 and not _archetype(c)],
        5,
        lambda c: f"title '{c.title}' is not an archetype, but career ranks #{c.rank} — "
        "a plain-language fit the title would hide",
    )
    take(
        "mid_band", _stride(by_rank, 0.25, 0.60, 10), 10, lambda c: f"mid-distribution (#{c.rank})"
    )
    take("low_band", _stride(by_rank, 0.72, 0.95, 6), 6, lambda c: f"low-distribution (#{c.rank})")
    take("honeypot", [c for c in by_rank if c.features.honeypot], 4, "honeypot in the ranking")
    take(
        "bottom_floor",
        list(reversed(by_rank)),
        3,
        lambda c: f"very bottom of the ranking (#{c.rank})",
    )

    return picks[:max_total]


def _archetype(c: RankedCandidate) -> bool:
    return is_archetype_title(c.candidate)


def select_pool_honeypots(
    flagged: frozenset[str], shortlist_ids: set[str], *, count: int = 2
) -> list[str]:
    """Deterministically pick a couple of honeypots that are NOT in the shortlist.

    Only one honeypot reaches the ranked shortlist, so to give the reviewer "a couple
    of honeypots" to label tier 0 we add the lowest-id flagged ids from the pool. They
    are correctly excluded by the pre-filter; ``evaluate`` reports them separately.
    """
    return sorted(flagged - shortlist_ids)[:count]


def _stride(
    ranking: list[RankedCandidate], lo: float, hi: float, count: int
) -> list[RankedCandidate]:
    """``count`` evenly-spaced candidates from the rank fraction window ``[lo, hi]``."""
    n = len(ranking)
    start, end = int(n * lo), int(n * hi)
    if end <= start or count <= 0:
        return []
    step = max(1, (end - start) // count)
    return [ranking[i] for i in range(start, end, step)][:count]


def _quantile(sorted_values: list[float], q: float) -> float:
    """Lower-interpolated quantile of an already-sorted list (empty → 0.0)."""
    if not sorted_values:
        return 0.0
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


# --------------------------------------------------------------------------- #
# Suggested tier — Claude's DRAFT, to be overwritten by the human reviewer.
# --------------------------------------------------------------------------- #
def suggest_tier(candidate: RankedCandidate) -> tuple[int, str]:
    """Draft a tier (0-5) and one-line rationale from the score breakdown.

    A transparent heuristic over the *fit* terms (role/domain/base_fit and the JD's
    disqualifier flags) — explicitly **not** availability, since a strong-but-inactive
    candidate is still a strong hire (see ``LABELING_GUIDE.md``). This is a starting
    point for the reviewer to overwrite, never the final label.
    """
    f = candidate.features
    if f.honeypot:
        return 0, "honeypot — impossible profile (deliberate trap) → always tier 0"

    base = candidate.result.base_fit
    if f.role_match <= 0.20 and f.domain_match <= 0.20:
        tier = 1 if base >= 0.45 else 0
        rationale = "off-target CV/speech or non-tech role+domain"
    elif (
        base >= 0.85 and f.role_match >= 0.90 and f.domain_match >= 0.90 and f.built_ranking >= 1.0
    ):
        tier, rationale = 5, "ideal: on-target role+domain, shipped ranking/search, strong fit"
    elif base >= 0.70 and f.role_match >= 0.90 and f.domain_match >= 0.90:
        tier, rationale = 4, "strong on-target production ML"
    elif base >= 0.55:
        tier, rationale = 3, "genuine applied ML; IR/ranking or product evidence thinner"
    elif base >= 0.40:
        tier, rationale = 2, "weak / adjacent fit"
    else:
        tier, rationale = 1, "very weak fit"

    hard = sorted(set(f.disqualifier_flags) & _HARD_FLAGS)
    if hard and tier > 2:
        tier = 2
        rationale = f"{rationale}; hard disqualifier {hard} the JD excludes → capped at 2"
    soft = sorted(set(f.disqualifier_flags) & _SOFT_FLAGS)
    if soft and tier > 1:
        tier -= 1
        rationale = f"{rationale}; soft flag {soft} → -1"

    return tier, f"{rationale} (base={base:.2f})"


# --------------------------------------------------------------------------- #
# Rendering the review sheet.
# --------------------------------------------------------------------------- #
REVIEW_COLUMNS: tuple[str, ...] = (
    "category",
    "rank",
    "candidate_id",
    "current_title",
    "years_experience",
    "country",
    "role_archetype",
    "domain",
    "built_ranking",
    "product_ratio",
    "seniority_fit",
    "disqualifier_flags",
    "honeypot",
    "final",
    "base_fit",
    "availability",
    "location",
    "career_sim",
    "lexical_evidence",
    "evidence_span",
    "summary",
    "career_history",
    "suggested_tier",
    "suggested_rationale",
    "tier",  # blank — the reviewer writes this
    "note",  # blank — the reviewer writes this
)


def review_row(pick: GoldPick, ranked: RankedCandidate) -> dict[str, object]:
    """Flatten a ranked candidate into one review-sheet row (blank ``tier``/``note``)."""
    f, r = ranked.features, ranked.result
    profile = ranked.candidate.get("profile") or {}
    signals: dict[str, object] = dict(ranked.signals) if ranked.signals else {}
    tier, rationale = suggest_tier(ranked)
    return {
        "category": pick.category,
        "rank": pick.rank,
        "candidate_id": pick.candidate_id,
        "current_title": ranked.title,
        "years_experience": profile.get("years_of_experience"),
        "country": profile.get("country"),
        "role_archetype": signals.get("role_archetype", ""),
        "domain": signals.get("domain", ""),
        "built_ranking": f"{f.built_ranking:.0f}",
        "product_ratio": f"{f.product_ratio:.2f}",
        "seniority_fit": f"{f.seniority_fit:.2f}",
        "disqualifier_flags": ", ".join(f.disqualifier_flags),
        "honeypot": f.honeypot,
        "final": f"{r.final:.3f}",
        "base_fit": f"{r.base_fit:.3f}",
        "availability": f"{f.availability:.3f}",
        "location": f"{f.location:.2f}",
        "career_sim": f"{f.career_sim:.3f}",
        "lexical_evidence": f"{f.lexical_evidence:.2f}",
        "evidence_span": _clip(signals.get("evidence_span", ""), 300),
        "summary": _clip(profile.get("summary"), 280),
        "career_history": _career_excerpt(ranked.candidate),
        "suggested_tier": tier,
        "suggested_rationale": f"[SUGGESTION — overwrite] {pick.reason}; {rationale}",
        "tier": "",
        "note": "",
    }


def pool_honeypot_row(candidate: Candidate, reasons: list[str]) -> dict[str, object]:
    """A review row for a pool honeypot not in the ranking (score columns blank)."""
    profile = candidate.get("profile") or {}
    row: dict[str, object] = {column: "" for column in REVIEW_COLUMNS}
    row.update(
        {
            "category": "honeypot_pool",
            "rank": "",
            "candidate_id": candidate.get("candidate_id", ""),
            "current_title": profile.get("current_title") or "(no title)",
            "years_experience": profile.get("years_of_experience"),
            "country": profile.get("country"),
            "honeypot": True,
            "summary": _clip(profile.get("summary"), 280),
            "career_history": _career_excerpt(candidate),
            "suggested_tier": 0,
            "suggested_rationale": f"[SUGGESTION — overwrite] honeypot ({', '.join(reasons)}) "
            "excluded by the pre-filter → tier 0",
        }
    )
    return row


def _career_excerpt(candidate: Candidate, max_roles: int = 3) -> str:
    """'Title @ Company (Nmo): desc…' for the most-recent roles, semicolon-joined."""
    history = candidate.get("career_history") or []
    parts: list[str] = []
    for entry in history[:max_roles]:
        title = (entry.get("title") or "").strip() or "(role)"
        company = (entry.get("company") or "").strip() or "(company)"
        months = entry.get("duration_months")
        tenure = f" ({months}mo)" if isinstance(months, int) else ""
        desc = _clip(entry.get("description"), 120)
        body = f": {desc}" if desc else ""
        parts.append(f"{title} @ {company}{tenure}{body}")
    return " | ".join(parts)


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the review sheet as CSV (csv module handles quoting/newlines)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    """Write the review sheet as readable cards grouped by category."""
    lines: list[str] = [
        "# Gold-set review sheet (Session 07)",
        "",
        "Read each card against `eval/LABELING_GUIDE.md` and record your tier in",
        "`eval/gold_labels.csv`. `suggested_tier` is Claude's draft — overwrite it.",
        "",
    ]
    current = ""
    for row in rows:
        category = str(row["category"])
        if category != current:
            lines += [f"## {category}", ""]
            current = category
        rank = row["rank"]
        rank_str = f"#{rank}" if rank not in ("", None) else "(not ranked)"
        lines.append(
            f"### {row['candidate_id']} — {row['current_title']} ({rank_str}) · "
            f"suggested tier **{row['suggested_tier']}**"
        )
        lines.append(
            f"- {row['years_experience']} yrs · {row['country']} · "
            f"archetype `{row['role_archetype']}` / domain `{row['domain']}` · "
            f"flags: {row['disqualifier_flags'] or '—'} · honeypot: {row['honeypot']}"
        )
        if row["final"] != "":
            lines.append(
                f"- score: final **{row['final']}** = base {row['base_fit']} "
                f"x avail {row['availability']} x loc {row['location']}  "
                f"(career_sim {row['career_sim']}, built_ranking {row['built_ranking']}, "
                f"lexical {row['lexical_evidence']})"
            )
        if row["summary"]:
            lines.append(f"- summary: {row['summary']}")
        if row["career_history"]:
            lines.append(f"- career: {row['career_history']}")
        if row["evidence_span"]:
            lines.append(f"- evidence: {row['evidence_span']}")
        lines.append(f"- _why included_: {row['suggested_rationale']}")
        lines.append("- **tier:** ____   **note:** ____")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def run(*, artifacts_dir: Path, candidates_path: Path, out_dir: Path, max_total: int) -> None:
    """Build the ranking, select the gold sample, and write the CSV + markdown sheets."""
    ranking = build_ranking(
        artifacts_dir=artifacts_dir, candidates_path=candidates_path, cfg=SCORING
    )
    logger.info("ranking built: %d scored candidates", len(ranking))

    picks = select_gold_candidates(ranking, max_total=max_total)
    by_id = {c.candidate_id: c for c in ranking}
    rows = [review_row(pick, by_id[pick.candidate_id]) for pick in picks]

    # A couple of honeypots from outside the shortlist, fetched from the pool.
    flagged = load_flagged_ids(artifacts_dir)
    pool_ids = select_pool_honeypots(flagged, set(by_id), count=2)
    if pool_ids:
        wanted = set(pool_ids)
        found: dict[str, Candidate] = {}
        for candidate in load_candidates(candidates_path):
            cid = candidate.get("candidate_id")
            if cid in wanted:
                found[cid] = candidate
                if len(found) == len(wanted):
                    break
        for cid in pool_ids:
            if cid in found:
                rows.append(pool_honeypot_row(found[cid], honeypot_reasons(found[cid])))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gold_review.csv"
    md_path = out_dir / "gold_review.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    _print_summary(rows, csv_path, md_path)


def _print_summary(rows: list[dict[str, object]], csv_path: Path, md_path: Path) -> None:
    from collections import Counter

    counts = Counter(str(row["category"]) for row in rows)
    print(f"\nWrote {len(rows)} candidates to:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print("\nBy category:")
    for category, count in counts.items():
        print(f"  {category:22} {count}")
    print(
        "\nNext: a human reads each row against eval/LABELING_GUIDE.md, writes the "
        "`tier` column,\nand saves final labels to eval/gold_labels.csv "
        "(candidate_id,tier[,note]). Do NOT ship the suggestions unreviewed.\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-total", type=int, default=90, help="cap on review-sheet size")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        out_dir=args.out_dir,
        max_total=args.max_total,
    )


if __name__ == "__main__":
    main()
