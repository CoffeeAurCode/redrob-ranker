"""sandbox/build_sample.py — OFFLINE prep: carve a self-contained 100-candidate
sample out of the full pool + committed artifacts so the sandbox Space can run the
real ``rank.py`` pipeline without shipping the whole 100k pool (~152 MB of
artifacts) or any network/LLM dependency.

This is a *prep* tool, run once before deploying the Space — never at sandbox
runtime (the same offline/precompute split as ``src/precompute/*``). It reads the
parent repo's ``artifacts/`` and ``data/candidates.jsonl`` and writes a tiny,
ids-restricted mirror under ``sandbox/``:

    sandbox/
    ├── sample_candidates.jsonl        # the 100 full candidate records
    └── artifacts/                     # the SAME artifact files, restricted to those ids
        ├── candidate_embeddings_000.npy + candidate_ids.npy + embeddings_meta.json
        ├── jd_reference.json          # copied whole (the reference vector)
        ├── llm_signals.jsonl          # only the sample ids
        ├── honeypot_flags.json        # only flagged sample ids
        └── reasoning.jsonl            # only the sample ids the LLM stage covered

Sample design (so the demo tells the whole story, not just "100 strong CVs"):
the bundle is the top of the global ranking *plus* a deliberate spread of traps
drawn from the gold set — the honeypot, CV-bait, consulting-only, job-hoppers,
stale coders, semantic mismatches, and bottom-floor profiles — so a reviewer can
watch the transparent scorer push them down (and the honeypot to 0.000) live.

Every id is verified to be inside the precomputed shortlist (== the prefilter
survivors), so all 100 pass ``passes_prefilter`` and ``rank.py`` emits exactly the
100 rows the official ``validate_submission.py`` requires.

    python sandbox/build_sample.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embedding import load_embeddings, save_artifacts  # noqa: E402
from src.honeypots import HONEYPOT_FLAGS_FILE  # noqa: E402
from src.io_utils import load_candidates  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
DATA = REPO_ROOT / "data" / "candidates.jsonl"
SANDBOX = REPO_ROOT / "sandbox"
OUT_ARTIFACTS = SANDBOX / "artifacts"
OUT_SAMPLE = SANDBOX / "sample_candidates.jsonl"

SAMPLE_SIZE = 100

# A deliberate spread of traps from the Session-07 gold set (every id is a
# shortlist member, so it survives the prefilter and shows up in the ranked
# table). These should rank *below* the genuine fits — that is the demo.
TRAP_IDS: tuple[str, ...] = (
    "CAND_0037000",  # honeypot — 8y role vs fewer total years → final 0.000
    "CAND_0092278",  # inactive-but-perfect (Senior NLP) — strong fit, down-weighted
    "CAND_0041611",  # inactive-but-perfect (Staff ML)
    "CAND_0060072",  # inactive-but-perfect (Staff ML)
    "CAND_0073007",  # cv_bait — CV career under an AI title
    "CAND_0022127",  # cv_bait — Computer Vision Engineer
    "CAND_0081321",  # cv_bait — Senior SWE (ML) with a CV career
    "CAND_0029449",  # consulting_only — services firms, no product
    "CAND_0015582",  # consulting_only
    "CAND_0070930",  # job_hopper — Data Scientist, many short stints
    "CAND_0061175",  # stale_coding — AI Research, no recent shipping
    "CAND_0054703",  # semantic_mismatch — CV career, wrong sub-domain
    "CAND_0092468",  # semantic_mismatch
    "CAND_0052195",  # title_surprise — CV title, recsys career (plain-language fit)
    "CAND_0009722",  # title_surprise — Computer Vision Engineer
    "CAND_0093540",  # bottom_floor — consulting + cv flags stacked
    "CAND_0088602",  # bottom_floor — cv_primary
    "CAND_0078856",  # low_band — consulting_only
)


def _load_shortlist() -> set[str]:
    return set(json.loads((ARTIFACTS / "shortlist_ids.json").read_text(encoding="utf-8")))


def _top_global_ids() -> list[str]:
    """Candidate ids from the committed submission.csv, in rank order."""
    with (REPO_ROOT / "submission.csv").open(encoding="utf-8", newline="") as handle:
        return [row["candidate_id"] for row in csv.DictReader(handle)]


def _choose_sample_ids() -> list[str]:
    """The 100 sample ids: the traps first, then the strongest globals to fill 100.

    All are asserted to be inside the shortlist (the prefilter survivor set), so the
    run yields exactly 100 scored rows.
    """
    shortlist = _load_shortlist()

    traps = [cid for cid in TRAP_IDS if cid in shortlist]
    missing = [cid for cid in TRAP_IDS if cid not in shortlist]
    if missing:
        print(f"  ! dropping {len(missing)} trap id(s) not in shortlist: {missing}")

    chosen: list[str] = list(dict.fromkeys(traps))  # de-dup, keep order
    for cid in _top_global_ids():
        if len(chosen) >= SAMPLE_SIZE:
            break
        if cid in shortlist and cid not in chosen:
            chosen.append(cid)

    if len(chosen) != SAMPLE_SIZE:
        raise SystemExit(f"could not assemble {SAMPLE_SIZE} sample ids (got {len(chosen)})")
    assert all(cid in shortlist for cid in chosen)
    return chosen


def _write_embeddings_subset(sample: set[str]) -> int:
    """Mirror the embeddings/ids/meta for just the sample ids (sorted, one shard)."""
    ids, embeddings = load_embeddings(ARTIFACTS, mmap=True)
    index = {cid: row for row, cid in enumerate(ids.tolist())}
    ordered = sorted(cid for cid in sample if cid in index)
    rows = [index[cid] for cid in ordered]

    sub_ids = np.asarray(ordered, dtype=ids.dtype)
    sub_emb = np.asarray(embeddings[rows], dtype=np.float16)

    meta = json.loads((ARTIFACTS / "embeddings_meta.json").read_text(encoding="utf-8"))
    meta.pop("shards", None)
    meta["n"] = len(ordered)
    meta["source"] = "sandbox/build_sample.py (ids-restricted subset of candidates.jsonl)"
    save_artifacts(OUT_ARTIFACTS, sub_ids, sub_emb, meta)
    return len(ordered)


def _filter_jsonl(src: Path, dst: Path, sample: set[str]) -> int:
    """Copy only the JSONL lines whose ``candidate_id`` is in the sample."""
    kept = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("candidate_id") in sample:
                fout.write(line + "\n")
                kept += 1
    return kept


def _filter_honeypots(sample: set[str]) -> int:
    """Mirror honeypot_flags.json restricted to flagged sample ids (sorted keys)."""
    data = json.loads((ARTIFACTS / HONEYPOT_FLAGS_FILE).read_text(encoding="utf-8"))
    kept = {cid: data[cid] for cid in sorted(data) if cid in sample}
    (OUT_ARTIFACTS / HONEYPOT_FLAGS_FILE).write_text(
        json.dumps(kept, indent=2) + "\n", encoding="utf-8"
    )
    return len(kept)


def _write_sample_candidates(sample: set[str]) -> int:
    """Extract the full candidate records for the sample, sorted by id (deterministic)."""
    records = {
        cid: rec
        for rec in load_candidates(DATA, skip_errors=True)
        if (cid := rec.get("candidate_id")) in sample
    }
    with OUT_SAMPLE.open("w", encoding="utf-8", newline="\n") as fout:
        for cid in sorted(records):
            fout.write(json.dumps(records[cid], ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sample_ids = _choose_sample_ids()
    sample = set(sample_ids)
    print(f"chosen sample: {len(sample_ids)} ids ({len(TRAP_IDS)} traps + globals)")

    n_emb = _write_embeddings_subset(sample)
    print(f"  embeddings subset: {n_emb} rows")

    (OUT_ARTIFACTS / "jd_reference.json").write_text(
        (ARTIFACTS / "jd_reference.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print("  jd_reference.json: copied")

    n_sig = _filter_jsonl(
        ARTIFACTS / "llm_signals.jsonl", OUT_ARTIFACTS / "llm_signals.jsonl", sample
    )
    print(f"  llm_signals.jsonl: {n_sig} rows")

    n_hp = _filter_honeypots(sample)
    print(f"  honeypot_flags.json: {n_hp} flagged")

    n_rsn = _filter_jsonl(ARTIFACTS / "reasoning.jsonl", OUT_ARTIFACTS / "reasoning.jsonl", sample)
    print(f"  reasoning.jsonl: {n_rsn} rows (LLM-covered; the rest use the deterministic fallback)")

    n_rec = _write_sample_candidates(sample)
    print(f"  sample_candidates.jsonl: {n_rec} records")

    if n_emb != SAMPLE_SIZE or n_rec != SAMPLE_SIZE:
        raise SystemExit(
            f"coverage gap: {n_emb} embeddings / {n_rec} records for {SAMPLE_SIZE} ids"
        )
    print("done — sandbox sample is self-contained.")


if __name__ == "__main__":
    main()
