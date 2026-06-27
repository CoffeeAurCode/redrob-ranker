"""Central configuration constants — defined once, defensible at a glance.

Per ``plan/CONVENTIONS.md`` ("no magic numbers"), model ids, prefixes and other
knobs live here rather than scattered as literals across the precompute scripts.
This module imports nothing heavy (no torch / sentence-transformers / pandas), so
even ``rank.py`` could import it without paying a startup cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    """How candidate and JD text become vectors — shared by Sessions 02 and 03.

    One model and one prefix convention are used on both sides so the candidate
    pool (Session 02) and the JD reference (Session 03) live in the same vector
    space. Vectors are L2-normalized, so cosine similarity is a plain dot product.

    BGE-*-en-v1.5 is asymmetric: it expects a retrieval *instruction* on the
    **query** side only. Candidate profiles are the corpus, so they get
    ``passage_prefix`` (empty); the JD and any hand-written sanity query get
    ``query_prefix``. Applying these consistently is what keeps the two artifacts
    comparable — see ``embeddings_meta.json`` for the recorded values.
    """

    model_id: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    normalize: bool = True
    # BGE v1.5 corpus/passage side takes no instruction.
    passage_prefix: str = ""
    # BGE v1.5 recommended query instruction for short-query→passage retrieval.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Encoder micro-batch size (CPU-friendly; offline, so throughput over latency).
    batch_size: int = 64


# The single embedding configuration imported everywhere it is needed.
EMBEDDING = EmbeddingConfig()


# The AI/ML/NLP/IR/DS archetype titles, lower-cased and whitespace-collapsed to
# match :func:`features.normalize_title`. Built from the Session-01 title EDA
# (``docs/schema.md`` → "EDA findings") and reconciled in Session 03 against the
# *literal* titles in candidates.jsonl (the EDA abbreviated a few, e.g. real data
# spells out "Senior Software Engineer (ML)", not "Senior SWE (ML)"). The pool has
# a *fixed* vocabulary of 47 distinct titles, of which these 19 are the genuine
# AI/ML/retrieval/ranking/DS roles (~1.0k candidates). Membership is therefore an
# exact-set test, not fuzzy matching — defensible and deterministic.
#
# Deliberately EXCLUDED:
#   * "computer vision engineer" — the JD's CV/speech/robotics bait (re-learning
#     IR fundamentals here); CV-primary profiles must not auto-pass on title.
#   * Broad adjacent titles (Data/Backend/Analytics Engineer, Data Analyst, SWE).
#     A Tier-5 fit hides among these, but most are filler, so they do NOT pass on
#     title alone — they enter the shortlist only via the similarity branch, i.e.
#     when their *career text* (not their label) is actually relevant.
_ARCHETYPE_TITLES: frozenset[str] = frozenset(
    {
        # Core AI/ML (exact literal titles as they appear in candidates.jsonl)
        "ml engineer",
        "machine learning engineer",
        "ai engineer",
        "ai research engineer",
        "ai specialist",
        "applied ml engineer",
        "junior ml engineer",
        "senior machine learning engineer",
        "staff machine learning engineer",
        "lead ai engineer",
        "senior ai engineer",
        "senior software engineer (ml)",
        "senior applied scientist",
        # NLP
        "nlp engineer",
        "senior nlp engineer",
        # Search / Recsys
        "search engineer",
        "recommendation systems engineer",
        # Data Science
        "data scientist",
        "senior data scientist",
    }
)


@dataclass(frozen=True)
class FilterConfig:
    """The cheap Session-03 pre-filter: archetype title OR similarity threshold.

    A candidate survives the pre-filter (and is sent to the expensive LLM
    extraction in Session 04) if *either* their normalized ``current_title`` is a
    known AI/ML archetype *or* their career-text cosine similarity to the JD
    reference embedding is at least :attr:`similarity_threshold`.

    The threshold is tuned offline in ``src/precompute/build_shortlist.py`` so the
    union lands at the ~1-3k shortlist the plan targets; the chosen value and the
    resulting survivor count are recorded in ``plan/PROGRESS.md``. The golden rule
    is to keep more when unsure — losing a true fit here is unrecoverable, and the
    LLM stage refines precision later.
    """

    archetype_titles: frozenset[str] = _ARCHETYPE_TITLES
    # Cosine-similarity cut for the similarity branch. Tuned in Session 03 against
    # the real JD reference (build_shortlist.py --scan): BGE cosine on this pool
    # has a sharp filler cliff — at 0.7125 ~133 obvious-filler titles (mechanical/
    # civil eng, HR, marketing) leak in; at 0.725 that collapses to ~3. 0.725 sits
    # just above the cliff and yields ~1.2k survivors (1047 archetype + ~146 by
    # similarity), comfortably inside the ~1-3k target. We keep generously here
    # (incl. ~129 CV-bait) since the LLM stage refines precision and losing a true
    # fit at the pre-filter is unrecoverable.
    similarity_threshold: float = 0.725


# The single filter configuration imported by features.py and build_shortlist.py.
FILTER = FilterConfig()
