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


@dataclass(frozen=True)
class HoneypotConfig:
    """Thresholds for the deterministic honeypot detector (Session 05).

    Honeypots are the ~80 "subtly impossible" profiles the challenge injected and
    forces to relevance tier 0; ranking >10% of them in the top 100 is an instant
    disqualification. This detector is *insurance* — a flagged candidate is zeroed
    in ``scoring.py``. Zeroing a real candidate is unrecoverable, so every knob
    here is tuned for **precision over recall**: strict about true impossibility,
    lenient about the ordinary messiness of real resumes (see ``docs/schema.md`` —
    e.g. ``skill.duration_months > yoe*12`` fires on 51% of the pool and is
    deliberately *not* a rule).
    """

    # "expert" proficiency in a skill with zero months of use ("claims mastery,
    # never used it"). Only "expert": in the real pool every such candidate has
    # 3-5 of these skills (never 1-2), matching the bundle's "expert in N skills,
    # 0 years used" honeypot; "advanced"+0mo never occurs and is a weaker claim.
    expert_proficiency: str = "expert"

    # A single career role lasting at least this many months *longer than the
    # candidate's entire claimed experience* is impossible ("8 years at a company,
    # 3 years total experience"). The 12-month margin absorbs rounding, part-time
    # spells, and the float ``years_of_experience`` so only clear contradictions
    # fire (20 candidates in the pool, disjoint from the expert+0mo set).
    role_excess_margin_months: int = 12


# The single honeypot configuration imported by honeypots.py and the detector CLI.
HONEYPOT = HoneypotConfig()
