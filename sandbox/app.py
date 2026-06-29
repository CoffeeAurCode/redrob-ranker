"""sandbox/app.py — a public, no-setup demo of the offline ranker.

A reviewer can run the *real* ``src/rank.py`` pipeline on a bundled 100-candidate
sample (or their own uploaded JSONL of pool candidates) straight to a ranked table
and a downloadable, validator-passing CSV — without cloning the repo or the 100k
pool.

The golden rule still holds here: this app loads only the precomputed
``artifacts/`` (an ids-restricted mirror of the committed artifacts) and imports
the same pure scoring code (``features`` + ``scoring``) that ``rank.py`` ships.
**No network, no LLM, no API key** — the only third-party imports are
``numpy``/``pandas`` (data wrangling) and ``gradio`` (the UI). The scoring logic is
imported, never forked.

Layout-agnostic: it finds ``src/`` next to this file (a flat Hugging Face Space)
or one directory up (the in-repo ``sandbox/`` folder), so the same file runs in
both places.

Run locally:   python sandbox/app.py
Deploy:        python sandbox/build_space.py  (see sandbox/README.md)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

HERE = Path(__file__).resolve().parent
# ``src`` sits beside this file in a flat Space, or one level up in the repo.
SRC_BASE = HERE if (HERE / "src").is_dir() else HERE.parent
if str(SRC_BASE) not in sys.path:
    sys.path.insert(0, str(SRC_BASE))

from src.config import FILTER  # noqa: E402
from src.features import passes_prefilter  # noqa: E402
from src.honeypots import load_flagged_ids  # noqa: E402
from src.io_utils import load_candidates  # noqa: E402
from src.llm_signals import LLM_SIGNALS_FILE, load_signal_cache  # noqa: E402
from src.rank import (  # noqa: E402
    SubmissionRow,
    reference_similarities,
    select_ranked,
    write_submission,
)
from src.reasoning import REASONING_FILE, load_reasoning_cache  # noqa: E402

ARTIFACTS = HERE / "artifacts"
BUNDLED_SAMPLE = HERE / "sample_candidates.jsonl"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "sample_submission.csv"

# Cosine assigned to a candidate with no precomputed embedding (the same fixed-pool
# fallback rank.py uses: a value below any real cosine, so it never matches).
_MISSING_SIM = -1.0
MAX_UPLOAD = 100


def _load_validator() -> ModuleType:
    """Load the official ``validate_submission.py`` by path (it lives beside ``src``).

    Loading dynamically (rather than ``import validate_submission``) keeps the
    vendored validator off the import path and makes the lookup work in both the
    in-repo and flat-Space layouts.
    """
    spec = importlib.util.spec_from_file_location(
        "validate_submission", SRC_BASE / "validate_submission.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VALIDATOR = _load_validator()


# --------------------------------------------------------------------------- #
# Core (no gradio import — unit-testable on its own).                          #
# --------------------------------------------------------------------------- #
def rank_candidates_file(candidates_path: Path) -> tuple[list[SubmissionRow], int, int]:
    """Run the offline rank pipeline over a JSONL file using the bundled artifacts.

    Returns ``(rows, survivors, n_input)``. ``top_n`` is sized to the survivor count
    so the call never raises on a small or heavily filtered sample; for the bundled
    100 (all shortlist members) survivors == 100, so the CSV is exactly the 100 rows
    the official validator requires.
    """
    sim_by_id = reference_similarities(ARTIFACTS)
    signals = load_signal_cache(ARTIFACTS / LLM_SIGNALS_FILE)
    flagged = load_flagged_ids(ARTIFACTS)
    reasoning_cache = load_reasoning_cache(ARTIFACTS / REASONING_FILE)

    candidates = list(load_candidates(candidates_path, skip_errors=True))

    # Size top_n to the prefilter survivors (a cheap pass using the same public
    # filter rank.py uses — not a re-implemented scorer) so select_ranked never
    # under-fills and raises.
    seen: set[str] = set()
    survivors = 0
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            continue
        seen.add(cid)
        if passes_prefilter(candidate, sim_by_id.get(cid, _MISSING_SIM), FILTER):
            survivors += 1

    rows = select_ranked(
        candidates,
        sim_by_id=sim_by_id,
        signals=signals,
        flagged=flagged,
        reasoning_cache=reasoning_cache,
        top_n=max(survivors, 1),
    )
    return rows, survivors, len(seen)


def rows_to_frame(rows: list[SubmissionRow]) -> pd.DataFrame:
    """A display table that surfaces the story: flagged honeypots + the reasoning source."""
    flagged = load_flagged_ids(ARTIFACTS)
    reasoned = set(load_reasoning_cache(ARTIFACTS / REASONING_FILE))
    records = [
        {
            "rank": row.rank,
            "candidate_id": row.candidate_id,
            "score": row.score,
            "note": "honeypot → 0" if row.candidate_id in flagged else "",
            "reasoning_source": "llm" if row.candidate_id in reasoned else "deterministic",
            "reasoning": row.reasoning,
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def run_ranking(candidates_path: Path) -> tuple[pd.DataFrame, str, str]:
    """Rank a JSONL file → (display table, status markdown, written CSV path)."""
    rows, survivors, n_input = rank_candidates_file(candidates_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_submission(rows, OUTPUT_CSV)

    errors = _VALIDATOR.validate_submission(str(OUTPUT_CSV))
    expected = _VALIDATOR.EXPECTED_DATA_ROWS
    if not errors:
        verdict = "✅ **Passes `validate_submission.py`** — a fully valid 100-row submission."
    elif len(rows) != expected:
        verdict = (
            f"📋 Produced **{len(rows)} ranked rows**. The official validator requires "
            f"exactly {expected}; the bundled sample meets that — upload ≥100 pool "
            "candidates for a submission-shaped CSV."
        )
    else:
        verdict = "⚠️ Validator issues:\n\n- " + "\n- ".join(errors)

    flagged = load_flagged_ids(ARTIFACTS)
    flagged_in = sum(1 for row in rows if row.candidate_id in flagged)
    status = (
        f"**Ranked {len(rows)} of {n_input} candidate(s)** "
        f"({survivors} passed the cheap pre-filter). "
        f"Honeypots forced to 0: **{flagged_in}**.\n\n{verdict}"
    )
    return rows_to_frame(rows), status, str(OUTPUT_CSV)


def _run_bundled() -> tuple[pd.DataFrame, str, str]:
    return run_ranking(BUNDLED_SAMPLE)


def _run_upload(file_obj: object) -> tuple[pd.DataFrame, str, str]:
    if file_obj is None:
        return pd.DataFrame(), "Upload a `.jsonl` file first, or run the bundled sample.", ""
    path = Path(getattr(file_obj, "name", str(file_obj)))
    n_lines = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if n_lines > MAX_UPLOAD:
        return (
            pd.DataFrame(),
            f"That file has {n_lines} rows; please keep the sample ≤ {MAX_UPLOAD}.",
            "",
        )
    return run_ranking(path)


# --------------------------------------------------------------------------- #
# UI (gradio imported lazily so the core stays importable without it).         #
# --------------------------------------------------------------------------- #
INTRO = """
# redrob-ranker — offline sandbox

Ranks candidates for a **Senior AI Engineer** JD with a transparent, CPU-only
scorer. All the heavy "understanding" (embeddings + LLM signal extraction) was
done **offline** and baked into `artifacts/`; this demo only loads those files and
computes a weighted score — **no network, no LLM, no API key at rank time.**

Click **Run bundled sample** to rank 100 real candidates (the strongest fits plus
a deliberate spread of traps — a honeypot, CV-bait, consulting-only, job-hoppers).
Watch the genuine fits rise and the traps sink (the honeypot is forced to
`0.000`). Then download a CSV that passes the official `validate_submission.py`.
"""

UPLOAD_HELP = (
    "Optional: upload your own `.jsonl` (≤100 candidates **from the released pool** — "
    "ids outside the bundled embedding subset degrade gracefully to non-matching, "
    "exactly as `rank.py` handles an unseen id)."
)


def build_ui() -> object:
    import gradio as gr

    with gr.Blocks(title="redrob-ranker sandbox") as demo:
        gr.Markdown(INTRO)
        run_btn = gr.Button("Run bundled sample", variant="primary")
        with gr.Accordion("Or rank your own sample", open=False):
            gr.Markdown(UPLOAD_HELP)
            upload = gr.File(label="candidates .jsonl", file_types=[".jsonl"])
            upload_btn = gr.Button("Rank uploaded sample")

        status = gr.Markdown()
        download = gr.File(label="Download ranked CSV")
        table = gr.Dataframe(
            headers=["rank", "candidate_id", "score", "note", "reasoning_source", "reasoning"],
            wrap=True,
            label="Ranked candidates",
        )

        run_btn.click(_run_bundled, outputs=[table, status, download])
        upload_btn.click(_run_upload, inputs=upload, outputs=[table, status, download])

    return demo


def main() -> None:
    build_ui().launch()


if __name__ == "__main__":
    main()
