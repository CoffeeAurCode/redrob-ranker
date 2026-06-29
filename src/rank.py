"""src/rank.py — the offline ranking entrypoint (the Stage-3 reproduce command).

    python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

OFFLINE GUARANTEE (the golden rule — see ``plan/00_OVERVIEW.md``):
``rank.py`` never touches a network or an LLM. Every LLM/embedding computation
happened beforehand and is baked into ``artifacts/``; this runtime only *loads*
those files, computes the transparent weighted score (``features`` + ``scoring``),
and writes the ranked CSV. Its only third-party import is ``numpy`` (one vectorized
dot product for the cosine column) — the rest is the standard library and the pure
``src`` helpers. A test (``tests/test_rank.py::test_rank_import_graph_is_offline``)
asserts the import closure contains no ``requests`` / ``openai`` /
``google.generativeai`` / ``httpx`` / ``sentence_transformers`` / ``torch``.

The pipeline mirrors the path the offline harness (``score_shortlist.py``) and the
eval builder (``eval/ranking.py``) walk, but re-derives everything from the
*primary* artifacts so the entrypoint is self-contained (it does not depend on the
``shortlist_ids.json`` precompute intermediate):

1. Load ``candidate_embeddings`` (mmap) + ``jd_reference`` and compute every
   candidate's cosine to the JD reference in one vectorized pass; load
   ``llm_signals``, ``honeypot_flags`` and ``reasoning`` caches.
2. Stream the candidate pool one row at a time (never slurp 100k), apply the
   cheap pre-filter (``features.passes_prefilter``), and score the survivors
   (``features.assemble_features`` → ``scoring.score``). Honeypot-flagged → 0.
3. Select the top 100, sort by **score descending then candidate_id ascending**,
   assign ranks 1-100, and join the grounded reasoning (deterministic grounded
   fallback for any id the LLM stage did not cover).
4. Write exactly the official columns ``candidate_id,rank,score,reasoning`` as a
   deterministic, byte-identical CSV.

Determinism: the score written to the CSV is the same fixed-precision string the
rows are *sorted on*, so the validator's "score non-increasing, ties by id
ascending" invariant holds by construction and re-runs are byte-identical.

Fixed-pool assumption (risk #1 in the overview): the released 100k pool is fixed
(the submission spec requires every ``candidate_id`` to exist in the released
``candidates.jsonl``, and Session 02 verified the embeddings cover the pool 100%).
``rank.py`` does not embed at run time, so an id with no precomputed embedding is
*degraded gracefully* (treated as non-matching, logged) rather than crashing — the
run always emits a valid 100-row CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FILTER, SCORING, FilterConfig, ScoringConfig  # noqa: E402
from src.embedding import load_embeddings  # noqa: E402
from src.features import assemble_features, passes_prefilter  # noqa: E402
from src.honeypots import load_flagged_ids  # noqa: E402
from src.io_utils import Candidate, load_candidates  # noqa: E402
from src.jd_reference import assert_reference_dim, load_jd_reference  # noqa: E402
from src.llm_signals import LLM_SIGNALS_FILE, LLMSignals, load_signal_cache  # noqa: E402
from src.reasoning import (  # noqa: E402
    REASONING_FILE,
    ReasoningRecord,
    build_facts,
    deterministic_reasoning,
    load_reasoning_cache,
)
from src.scoring import ScoreResult, score  # noqa: E402

logger = logging.getLogger("rank")

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "candidates.jsonl"
DEFAULT_OUT = REPO_ROOT / "submission.csv"

# The official submission columns, in the official order (the validator pins this
# exact header — see plan/PROGRESS.md, Session 01: candidate_id FIRST, not rank).
SUBMISSION_HEADER: tuple[str, ...] = ("candidate_id", "rank", "score", "reasoning")
TOP_N = 100

# Fixed-precision score formatting. The rows are sorted on the *parsed-back* value
# of this exact string, so the CSV column and the sort key are identical floats —
# which makes the validator's non-increasing / tie-break checks hold by
# construction and the output byte-identical on re-run.
SCORE_DECIMALS = 6
# Cosine assigned to a candidate with no precomputed embedding (fixed-pool fallback):
# a value below any real cosine, so it never passes the similarity branch.
_MISSING_SIM = -1.0


@dataclass(frozen=True)
class SubmissionRow:
    """One output row, exactly the four official CSV fields (score is preformatted)."""

    candidate_id: str
    rank: int
    score: str
    reasoning: str


@dataclass(frozen=True)
class _Scored:
    """A scored survivor carried through the sort (keeps the record for fallback reasoning)."""

    candidate_id: str
    score_str: str  # the fixed-precision CSV string
    score_val: float  # that string parsed back to float — the sort key
    candidate: Candidate
    result: ScoreResult


def format_score(value: float) -> str:
    """Format a final score as the fixed-precision string written to the CSV.

    Using one canonical representation (and sorting on its parsed-back value) is what
    keeps re-runs byte-identical and the validator's score-ordering checks consistent.
    """
    return f"{value:.{SCORE_DECIMALS}f}"


def select_ranked(
    candidates: Iterable[Candidate],
    *,
    sim_by_id: Mapping[str, float],
    signals: Mapping[str, LLMSignals],
    flagged: frozenset[str] | set[str],
    reasoning_cache: Mapping[str, ReasoningRecord],
    top_n: int = TOP_N,
    cfg: ScoringConfig = SCORING,
    filter_cfg: FilterConfig = FILTER,
) -> list[SubmissionRow]:
    """Score the survivors, take the top ``top_n``, and build the submission rows.

    The pure core (no file I/O): consumes an iterable of candidates plus the
    already-loaded artifact maps, so it is fully unit-testable. Sorts by score
    descending then ``candidate_id`` ascending — the deterministic golden-rule order.
    """
    scored = _score_survivors(
        candidates,
        sim_by_id=sim_by_id,
        signals=signals,
        flagged=flagged,
        cfg=cfg,
        filter_cfg=filter_cfg,
    )
    scored.sort(key=lambda s: (-s.score_val, s.candidate_id))

    if len(scored) < top_n:
        raise SystemExit(
            f"only {len(scored)} candidate(s) passed the pre-filter; cannot produce a valid "
            f"{top_n}-row submission. Check the artifacts and the candidate pool."
        )

    rows: list[SubmissionRow] = []
    for rank, s in enumerate(scored[:top_n], start=1):
        reasoning = _reasoning_for(s, signals.get(s.candidate_id), rank, reasoning_cache)
        rows.append(SubmissionRow(s.candidate_id, rank, s.score_str, reasoning))
    return rows


def _score_survivors(
    candidates: Iterable[Candidate],
    *,
    sim_by_id: Mapping[str, float],
    signals: Mapping[str, LLMSignals],
    flagged: frozenset[str] | set[str],
    cfg: ScoringConfig,
    filter_cfg: FilterConfig,
) -> list[_Scored]:
    """Stream the pool, pre-filter, and score each survivor (defensive — never crashes).

    Skips unidentifiable or duplicate rows. A candidate with no precomputed embedding
    is degraded to a non-matching cosine (fixed-pool fallback) rather than dropped, so
    an archetype-titled straggler is still scored; it simply cannot float to the top.
    """
    scored: list[_Scored] = []
    seen: set[str] = set()
    missing_embedding = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            continue
        seen.add(cid)

        sim = sim_by_id.get(cid)
        if sim is None:
            missing_embedding += 1
            sim = _MISSING_SIM
        if not passes_prefilter(candidate, sim, filter_cfg):
            continue

        features = assemble_features(
            candidate,
            cosine=sim,
            signals=signals.get(cid),
            honeypot=cid in flagged,
            cfg=cfg,
        )
        result = score(features, cfg)
        score_str = format_score(result.final)
        scored.append(_Scored(cid, score_str, float(score_str), candidate, result))

    if missing_embedding:
        logger.warning(
            "%d input candidate(s) had no precomputed embedding (treated as non-matching)",
            missing_embedding,
        )
    logger.info("scored %d survivors after the pre-filter", len(scored))
    return scored


def _reasoning_for(
    scored: _Scored,
    signal: LLMSignals | None,
    rank: int,
    reasoning_cache: Mapping[str, ReasoningRecord],
) -> str:
    """The grounded reasoning for a top candidate: the cached LLM text, else the fallback.

    Any top-100 id the offline LLM stage did not cover falls back to the
    deterministic grounded reasoning (built only from the candidate's real facts and
    its score breakdown), so no row ever ships empty and ``rank.py`` stays LLM-free.
    """
    record = reasoning_cache.get(scored.candidate_id)
    if record is not None:
        text = " ".join(record["reasoning"].split())
        if text:
            return text
    facts = build_facts(scored.candidate, signal, scored.result, rank)
    return " ".join(deterministic_reasoning(facts).split())


# --------------------------------------------------------------------------- #
# Artifact loading (the I/O edge).                                              #
# --------------------------------------------------------------------------- #
def reference_similarities(artifacts_dir: Path) -> dict[str, float]:
    """Cosine of every pool candidate against the JD reference, keyed by ``candidate_id``.

    Vectors are L2-normalized, so cosine is a plain dot product of the float16
    candidate matrix (upcast to float32) with the float32 reference vector — one
    vectorized pass over the pool. The reference dim is guarded against the embedding
    dim first so a mismatched artifact fails loudly instead of producing garbage.
    """
    reference = load_jd_reference(artifacts_dir)
    assert_reference_dim(reference, artifacts_dir)
    ref_vec = np.asarray(reference["reference_embedding"], dtype=np.float32)
    ids, embeddings = load_embeddings(artifacts_dir, mmap=True)
    sims = np.asarray(embeddings, dtype=np.float32) @ ref_vec
    return dict(zip(ids.tolist(), sims.tolist(), strict=True))


def build_submission_rows(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    top_n: int = TOP_N,
    cfg: ScoringConfig = SCORING,
) -> list[SubmissionRow]:
    """Load the committed artifacts and produce the ranked top-``top_n`` rows."""
    sim_by_id = reference_similarities(artifacts_dir)
    signals = load_signal_cache(artifacts_dir / LLM_SIGNALS_FILE)
    flagged = load_flagged_ids(artifacts_dir)
    reasoning_cache = load_reasoning_cache(artifacts_dir / REASONING_FILE)
    logger.info(
        "artifacts: embeddings=%d signals=%d honeypot-flags=%d reasonings=%d",
        len(sim_by_id),
        len(signals),
        len(flagged),
        len(reasoning_cache),
    )
    candidates = load_candidates(candidates_path, skip_errors=True)
    return select_ranked(
        candidates,
        sim_by_id=sim_by_id,
        signals=signals,
        flagged=flagged,
        reasoning_cache=reasoning_cache,
        top_n=top_n,
        cfg=cfg,
    )


def write_submission(rows: Iterable[SubmissionRow], out_path: Path) -> None:
    """Write the rows to ``out_path`` as a deterministic, byte-identical CSV.

    ``lineterminator="\\n"`` (not the platform default ``\\r\\n``) and UTF-8 with no
    BOM keep the bytes identical across OSes and re-runs. ``csv.writer`` minimally
    quotes any reasoning containing a comma; reasonings are already single-line.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUBMISSION_HEADER)
        for row in rows:
            writer.writerow([row.candidate_id, row.rank, row.score, row.reasoning])


def run(
    *,
    artifacts_dir: Path,
    candidates_path: Path,
    out_path: Path,
    top_n: int = TOP_N,
    cfg: ScoringConfig = SCORING,
) -> list[SubmissionRow]:
    """Build and write the submission CSV; return the rows (for logging/tests)."""
    rows = build_submission_rows(
        artifacts_dir=artifacts_dir, candidates_path=candidates_path, top_n=top_n, cfg=cfg
    )
    write_submission(rows, out_path)
    logger.info("wrote %d rows -> %s", len(rows), out_path)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline top-100 candidate ranker (CPU-only, no network).",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--top", type=int, default=TOP_N, help="number of rows to emit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args(argv)
    run(
        artifacts_dir=args.artifacts,
        candidates_path=args.candidates,
        out_path=args.out,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
