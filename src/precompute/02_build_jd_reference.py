"""Precompute 02 — build the JD rubric + "ideal candidate" reference embedding.

OFFLINE step (the golden rule: this never runs inside ``rank.py`` — it makes one
network LLM call and is cached to ``artifacts/``). The pipeline is:

    JD text ─► one Gemini call (rubric + ideal paragraph) ─► embed the paragraph
            with the SAME model + query prefix as the candidates ─► jd_reference.json

The output ``artifacts/jd_reference.json`` is the single file ``career_sim``,
``role_match`` and ``domain_match`` all key off in later sessions.

Run from the repo root (needs ``GEMINI_API_KEY`` in ``.env``)::

    python src/precompute/02_build_jd_reference.py                 # build it
    python src/precompute/02_build_jd_reference.py --force         # rebuild
    python src/precompute/02_build_jd_reference.py --dry-run       # print prompt, no call

All reusable, testable logic lives in ``src.jd_reference``; this file is the thin
CLI/orchestration layer (the only part that touches the network and the model).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBEDDING  # noqa: E402  (after sys.path bootstrap)
from src.embedding import encode_normalized, load_model, query_text  # noqa: E402
from src.io_utils import parse_json_safe, use_utf8_stdout  # noqa: E402
from src.jd_reference import (  # noqa: E402
    JD_REFERENCE_FILE,
    JDReference,
    assert_reference_dim,
    build_jd_reference,
    build_rubric_prompt,
    parse_jd_response,
    save_jd_reference,
)

logger = logging.getLogger("build_jd_reference")

DEFAULT_JD = REPO_ROOT / "docs" / "challenge" / "job_description.md"
DEFAULT_OUT = REPO_ROOT / "artifacts"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"


def call_gemini(prompt: str, *, model_name: str, api_key: str) -> str:
    """One JSON-mode, temperature-0 Gemini call. Returns the raw response text.

    The ``google.generativeai`` import is local so the rest of this module (and
    ``--dry-run``) works without the package or a key installed.
    """
    # google-generativeai ships partial type info that doesn't re-export these at
    # the package top level, so mypy flags attr-defined though the runtime is fine.
    import google.generativeai as genai

    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    model = genai.GenerativeModel(  # type: ignore[attr-defined]
        model_name,
        generation_config={"temperature": 0, "response_mime_type": "application/json"},
    )
    return str(model.generate_content(prompt).text)


def build_reference(
    *,
    jd_text: str,
    llm_model: str,
    api_key: str,
    jd_source: str,
) -> JDReference:
    """Run the full Part-A pipeline and return the assembled reference payload."""
    prompt = build_rubric_prompt(jd_text)

    logger.info("calling %s for the JD rubric …", llm_model)
    raw = call_gemini(prompt, model_name=llm_model, api_key=api_key)
    rubric, ideal_text = parse_jd_response(parse_json_safe(raw))
    logger.info(
        "rubric: %d archetypes, %d must-haves, %d disqualifiers; ideal paragraph %d chars",
        len(rubric["role_archetypes"]),
        len(rubric["must_haves"]),
        len(rubric["hard_disqualifiers"]),
        len(ideal_text),
    )

    logger.info("loading %s to embed the ideal-candidate paragraph …", EMBEDDING.model_id)
    model = load_model()
    vector = encode_normalized(model, [query_text(ideal_text)])[0]

    return build_jd_reference(
        rubric=rubric,
        ideal_text=ideal_text,
        reference_embedding=[float(x) for x in vector.tolist()],
        model_id=EMBEDDING.model_id,
        query_prefix=EMBEDDING.query_prefix,
        llm_model=llm_model,
        jd_source=jd_source,
        created=date.today().isoformat(),
    )


def run(
    *,
    jd_path: Path,
    out_dir: Path,
    llm_model: str,
    force: bool,
    dry_run: bool,
) -> None:
    """Orchestrate the build: load env/JD, call the LLM, embed, validate, save."""
    jd_text = jd_path.read_text(encoding="utf-8")

    if dry_run:
        print(build_rubric_prompt(jd_text))
        return

    out_path = out_dir / JD_REFERENCE_FILE
    if out_path.exists() and not force:
        logger.info("%s already exists — use --force to rebuild", out_path)
        return

    api_key = _require_api_key()
    jd_source = jd_path.relative_to(REPO_ROOT).as_posix()
    reference = build_reference(
        jd_text=jd_text,
        llm_model=llm_model,
        api_key=api_key,
        jd_source=jd_source,
    )

    # Guard the Session-03 gotcha before writing: the reference must live in the
    # candidates' vector space (same model ⇒ same dim) or every cosine is noise.
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_reference_dim(reference, out_dir)

    saved = save_jd_reference(out_dir, reference)
    logger.info("wrote %s (embedding dim %d)", saved, reference["embedding_dim"])


def _require_api_key() -> str:
    """Load ``.env`` and return ``GEMINI_API_KEY`` or fail with a clear message."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Google AI Studio key (this is precompute only; rank.py never reads it)."
        )
    return api_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jd", type=Path, default=DEFAULT_JD, help="path to the JD text")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifacts output directory")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Gemini model id")
    parser.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the prompt and exit (no API call)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = parse_args(argv)
    run(
        jd_path=args.jd,
        out_dir=args.out,
        llm_model=args.model,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
