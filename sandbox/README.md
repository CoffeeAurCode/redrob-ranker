# Sandbox — try the offline ranker without local setup

A small [Gradio](https://www.gradio.app/) app that runs the **real** `src/rank.py`
pipeline on a bundled 100-candidate sample (or an uploaded JSONL) → a ranked table
+ a downloadable CSV that passes the official `validate_submission.py`.

It exists so a reviewer can try the ranker in a browser without cloning the repo
or the 100k pool.

## What it demonstrates

- **The golden rule, live.** The rank path loads only the precomputed
  `artifacts/` and imports the same pure scoring code (`features` + `scoring`)
  that `rank.py` ships. **No network, no LLM, no API key.** The only third-party
  imports are `numpy`/`pandas` and `gradio` (the UI). `tests/test_sandbox.py`
  mechanically asserts the rank path imports no network/LLM library, mirroring
  `rank.py`'s own offline guard.
- **Trap avoidance.** The bundled sample is the strongest fits plus a deliberate
  spread of traps from the gold set — a honeypot, CV-bait, consulting-only
  careers, job-hoppers, stale coders. The transparent scorer pushes them down and
  forces the honeypot to `0.000`, visible in the ranked table.
- **A valid submission.** All 100 bundled ids are inside the precomputed shortlist
  (the prefilter survivor set), so the run emits exactly the 100 rows the official
  validator requires, and the app shows the validator's verdict.

## Fixed-pool behavior (important)

The app does **not** embed at run time (that would need a model + network —
forbidden). It loads precomputed embeddings keyed by `candidate_id`, so it can
only score ids it has a vector for. The bundled `artifacts/` is an ids-restricted
mirror of the committed artifacts covering exactly the 100 bundled candidates.

When you **upload** your own JSONL, any id outside that subset degrades gracefully
to a non-matching cosine (`-1.0`) — exactly how `rank.py` handles an unseen id in
the fixed pool — rather than crashing. So uploads are most meaningful with ids
from the released pool that the bundle covers; arbitrary ids will simply score low.

## Run locally

```bash
pip install -r sandbox/requirements.txt
python sandbox/app.py            # opens http://127.0.0.1:7860
```

## Regenerate the bundled sample (offline prep)

The committed `sandbox/artifacts/` + `sandbox/sample_candidates.jsonl` are carved
from the full pool + committed artifacts by an offline prep script (needs the
~487 MB `data/candidates.jsonl` and the full `artifacts/`):

```bash
python sandbox/build_sample.py
```

## Deploy to a Hugging Face Space

A Space is its own git repo, so it must carry everything it imports. The packager
assembles a flat, self-contained directory (< 1 MB, no Git LFS):

```bash
python sandbox/build_space.py    # writes sandbox/space/
```

Then create a **Gradio** Space at huggingface.co and push `sandbox/space/`:

```bash
cd sandbox/space
git init
git remote add origin https://huggingface.co/spaces/<you>/redrob-ranker-sandbox
git add .
git commit -m "offline ranker sandbox"
git push -u origin main
```

The Space builds from `requirements.txt` (gradio + numpy + pandas only) and serves
`app.py`. Record the public URL in `submission_metadata.yaml` and `plan/PROGRESS.md`.

## Files

| File | Role |
| --- | --- |
| `app.py` | Gradio UI + the rank-path glue (imports `src`, never forks scoring). |
| `build_sample.py` | Offline: carves the 100-id artifact subset + sample from the pool. |
| `build_space.py` | Offline: packages a self-contained HF Space at `sandbox/space/`. |
| `requirements.txt` | The Space's tiny runtime deps (no torch/LLM libs). |
| `artifacts/` | The ids-restricted artifact subset (committed, ~260 KB). |
| `sample_candidates.jsonl` | The 100 bundled candidate records (committed). |
