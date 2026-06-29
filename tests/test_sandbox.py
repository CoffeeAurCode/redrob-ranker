"""Tests for the public sandbox app (``sandbox/app.py``).

The sandbox runs the *real* rank path on a bundled 100-candidate sample, so it must
honor the same guarantees as ``rank.py``:

* **Self-contained & valid** — the bundled sample + ids-restricted artifacts produce
  exactly the 100 rows the official ``validate_submission.py`` accepts, with the
  honeypot zeroed.
* **Offline import guard** — exercising the rank path pulls in no network/LLM library
  (mirrors ``test_rank.py::test_rank_import_graph_is_offline``).

Both run the app in a subprocess: the app lives under ``sandbox/`` (not on the test
import path) and imports ``gradio`` lazily, so a subprocess keeps the assertions
simple and the test file free of optional UI dependencies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "sandbox"

# The bundle is committed; if it's missing, the prep step (build_sample.py) hasn't run.
_BUNDLE_READY = (SANDBOX / "sample_candidates.jsonl").exists() and (
    SANDBOX / "artifacts" / "jd_reference.json"
).exists()
_needs_bundle = pytest.mark.skipif(not _BUNDLE_READY, reason="sandbox bundle not built")


def _run_app_snippet(body: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet with ``sandbox/`` on ``sys.path`` so ``import app`` resolves."""
    code = f"import sys\nsys.path.insert(0, r{str(SANDBOX)!r})\nimport app\n{body}\n"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


@_needs_bundle
def test_bundled_sample_is_a_valid_100_row_submission() -> None:
    """The bundled sample ranks to exactly 100 validator-passing rows, honeypot zeroed."""
    result = _run_app_snippet(
        "rows, survivors, n = app.rank_candidates_file(app.BUNDLED_SAMPLE)\n"
        "app.run_ranking(app.BUNDLED_SAMPLE)\n"
        "errors = app._VALIDATOR.validate_submission(str(app.OUTPUT_CSV))\n"
        "flagged = app.load_flagged_ids(app.ARTIFACTS)\n"
        "zeroed = [r.candidate_id for r in rows if r.candidate_id in flagged]\n"
        "assert len(rows) == 100, len(rows)\n"
        "assert survivors == 100 and n == 100, (survivors, n)\n"
        "assert not errors, errors\n"
        "assert all(rows[i].score >= rows[i + 1].score for i in range(len(rows) - 1))\n"
        "assert zeroed and all(r.score == '0.000000' for r in rows "
        "if r.candidate_id in flagged), 'honeypot not zeroed'\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


@_needs_bundle
def test_sandbox_rank_path_is_offline() -> None:
    """Exercising the sandbox rank path imports no network/LLM library (the golden rule)."""
    forbidden = [
        "requests",
        "httpx",
        "httpcore",
        "openai",
        "google.generativeai",
        "google.genai",
        "ollama",
        "sentence_transformers",
        "torch",
        "grpc",
    ]
    result = _run_app_snippet(
        "app.run_ranking(app.BUNDLED_SAMPLE)\n"
        f"bad = [m for m in {forbidden!r} if m in sys.modules]\n"
        "print(';'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    assert result.returncode == 0, (
        f"forbidden modules imported by the sandbox rank path: {result.stdout.strip()!r} "
        f"(stderr: {result.stderr.strip()!r})"
    )
