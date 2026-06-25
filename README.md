# redrob-ranker

Ranks the top 100 of ~100,000 candidates for a Senior AI Engineer JD. The heavy
"understanding" happens **offline** (LLM + embeddings, baked into `artifacts/`);
the shipped runtime is a fast, transparent, CPU-only scorer.

> **Status:** scaffolding (Session 00). Sections below are skeletons — filled in
> as later sessions land. `TODO` markers flag what is not yet wired up.

## Reproduce the submission

```bash
# TODO(session-06+): this is the single Stage-3 reproduce command.
# Must run on CPU, with no network, in under 5 minutes.
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

## Offline guarantee (the golden rule)

> **`rank.py` never touches a network or an LLM.** Every LLM/embedding
> computation happens beforehand and is saved under `artifacts/`. The runtime
> only loads those files, computes a transparent weighted score, and writes the
> ranked CSV.

This keeps the submission compliant at Stage 3 (organizers re-run `rank.py` in a
locked, network-less container), reproducible, and defensible line-by-line.

## Setup

Target **Python 3.11**.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # PowerShell (Windows)
# source .venv/bin/activate     # bash / macOS / Linux
pip install -r requirements.txt
```

Precompute scripts need API keys — copy `.env.example` to `.env` and fill in
`GEMINI_API_KEY`. The runtime (`rank.py`) needs neither keys nor network.

### Data

The 100k candidate pool (`data/candidates.jsonl`, ~465 MB) is **gitignored** —
obtain it from the challenge bundle (`candidates.jsonl[.gz]`) and place it at
`data/candidates.jsonl`. Confirm `wc -l data/candidates.jsonl` prints `100000`.
The real field layout (identity under `profile`, behavioral under
`redrob_signals`) is documented in [`docs/schema.md`](docs/schema.md).

## Architecture

```
OFFLINE (precompute, no limits, LLM + GPU allowed)
  candidates.jsonl ─► embeddings + jd_reference + llm_signals + honeypot_flags ─► artifacts/

AT RANK TIME (rank.py — CPU only, <5 min, no network)
  candidates.jsonl + artifacts/* ─► features ─► scoring ─► top 100 ─► submission.csv
```

TODO(session-12): per-artifact provenance table (which script produced each
file, when, with which model). See `plan/00_OVERVIEW.md` for the full design.

## Repository layout

```
redrob-ranker/
├── data/            # provided candidate pool (gitignored; see Setup)
├── docs/
│   ├── schema.md    # the REAL candidate schema + usage mapping + EDA findings
│   └── challenge/   # organizer bundle, vendored (JD, schema, signals doc, samples)
├── artifacts/       # PRECOMPUTED, committed — embeddings, signals, flags, reasoning
├── src/
│   ├── precompute/  # offline scripts (LLM + embeddings); never run at rank time
│   ├── features.py  # assembles feature vectors from artifacts        TODO
│   ├── scoring.py   # the transparent weighted score                  TODO
│   └── rank.py      # ENTRYPOINT — loads artifacts, scores, writes CSV TODO
├── eval/            # gold labels + NDCG/MAP/P@k evaluation            TODO
├── tests/           # unit tests for scoring, honeypots, io           TODO
└── validate_submission.py  # CSV format validator
```
