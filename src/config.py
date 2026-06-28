"""Central configuration constants — defined once, defensible at a glance.

Per ``plan/CONVENTIONS.md`` ("no magic numbers"), model ids, prefixes and other
knobs live here rather than scattered as literals across the precompute scripts.
This module imports nothing heavy (no torch / sentence-transformers / pandas), so
even ``rank.py`` could import it without paying a startup cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


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


# --------------------------------------------------------------------------- #
# Scoring (Session 06). Every map/weight below is a defensible knob the         #
# Session-08 calibration tunes against the gold set — they live here, once, so   #
# no scoring literal is scattered across features.py / scoring.py.              #
# --------------------------------------------------------------------------- #

# role_archetype (from llm_signals) → role_match score. 1.0 = squarely the JD's
# target (applied-ML / search / ranking / recsys); partial credit for adjacent
# families; 0 for a clean mismatch. Keys are the controlled vocabulary in
# ``llm_signals.ROLE_ARCHETYPES`` — a test asserts this map covers it exactly.
_ROLE_MATCH_SCORES: Mapping[str, float] = MappingProxyType(
    {
        "recsys_search": 1.0,  # search / recommendation — the JD bullseye
        "ml_engineer": 1.0,  # applied ML engineer — the core role
        "ai_engineer": 0.9,  # adjacent-strong; the LLM-glue risk is a separate flag
        "data_scientist": 0.6,  # analytical, may lack production ranking systems
        "data_eng": 0.4,  # builds systems/pipelines, not ranking — adjacent
        "swe_generic": 0.3,  # generic SWE; could have built systems, not ML-focused
        "cv_speech": 0.2,  # the JD's CV/speech bait; cv_primary flag adds the bite
        "non_tech": 0.0,  # clean mismatch
    }
)

# domain (from llm_signals) → domain_match score. NLP/IR and recsys/search are the
# JD's problem space; everything else is adjacent-to-off. Keys = ``llm_signals.DOMAINS``.
_DOMAIN_MATCH_SCORES: Mapping[str, float] = MappingProxyType(
    {
        "nlp_ir": 1.0,  # natural language / information retrieval — the core domain
        "recsys_search": 1.0,  # ranking / recommendation / search
        "data_eng": 0.4,  # data infrastructure — adjacent
        "generic_swe": 0.3,  # general software — weakly adjacent
        "cv_speech": 0.1,  # computer vision / speech — off-domain bait
        "non_tech": 0.0,  # off-domain
    }
)

# disqualifier_flag (from llm_signals) → points subtracted from base_fit. base_fit
# itself is in [0, 1] (the term weights sum to 1), so these are calibrated relative
# to that scale: the JD's explicit "do NOT want" conditions (consulting-only, the
# CV-primary bait) bite hardest; softer signals (job-hopping) bite least. Flags
# stack additively — two strong disqualifiers can drive base_fit below 0 (floored).
# Keys = ``llm_signals.DISQUALIFIER_FLAGS``.
_PENALTY_PER_FLAG: Mapping[str, float] = MappingProxyType(
    {
        "consulting_only": 0.35,  # JD explicitly excludes pure IT-services careers
        "cv_primary": 0.35,  # CV/speech without NLP/IR — the headline bait
        "pure_research": 0.30,  # academia-only, no production deployment
        "langchain_only_recent": 0.30,  # only recent LLM-framework glue, no ML depth
        "stale_coding": 0.20,  # no hands-on production coding in ~18 months
        "job_hopper": 0.15,  # frequent title-chasing hops — softest signal
    }
)

# Career-text keyword evidence (the w7 term). These are matched against the
# candidate's CAREER FREE-TEXT (build_embedding_text — title + summary + role
# descriptions), NOT the structured ``skills`` list the dataset made into uniform
# noise. A description that says "built a learning-to-rank model evaluated with
# NDCG" is genuine narrative evidence of doing the work; a skills tag of "RAG" is
# not. lexical_evidence is the fraction of these categories with ≥1 hit, so breadth
# of real evidence is rewarded and saturates — and it carries the smallest weight.
_LEXICAL_KEYWORD_CATEGORIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "retrieval": (
            "retrieval",
            "rag",
            "semantic search",
            "dense retrieval",
            "embedding",
            "vector search",
            "nearest neighbor",
        ),
        "vector_db": (
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "faiss",
            "opensearch",
            "elasticsearch",
            "vector database",
            "vector db",
        ),
        "ranking": (
            "ranking",
            "learning to rank",
            "learning-to-rank",
            "recommendation",
            "recommender",
            "recsys",
            "relevance",
        ),
        "evaluation": (
            "ndcg",
            "mrr",
            "map@",
            "a/b test",
            "ab test",
            "offline metric",
            "offline eval",
            "precision@",
            "recall@",
        ),
    }
)

# Availability sub-weights — how the behavioral signals blend into the [floor, 1]
# multiplier. Weights sum to 1.0; recency and engagement dominate, identity
# verification is a tie-breaker. Keys are the component names produced by
# ``features._availability_components``.
_AVAILABILITY_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "recency": 0.25,  # last_active_date decayed against the snapshot date
        "recruiter_response_rate": 0.15,
        "open_to_work": 0.15,
        "interview_completion_rate": 0.15,
        "notice_period": 0.15,  # shorter notice → more available
        "offer_acceptance": 0.07,  # -1 sentinel (no history) treated as neutral
        "verified_email": 0.04,
        "verified_phone": 0.04,
    }
)


@dataclass(frozen=True)
class ScoringConfig:
    """Every knob of the transparent linear score (Session 06; tuned in Session 08).

    The contract (``plan/00_OVERVIEW.md``)::

        base_fit  = w1·career_sim + w2·role_match + w3·domain_match
                  + w4·product_ratio + w5·seniority_fit + w6·built_ranking
                  + w7·lexical_evidence
        penalties = Σ penalty[flag] for flag in disqualifier_flags
        final     = max(0, base_fit - penalties) · availability · location
        final     = 0.0 if honeypot

    The seven term weights sum to 1.0 so ``base_fit`` reads as a 0-1 fit before
    penalties — which makes the penalty and weight magnitudes interpretable at a
    glance (and defensible line-by-line at Stage 5).
    """

    # base_fit term weights (w1..w7). Sum to 1.0 (test_term_weights_sum_to_one).
    # FROZEN in Session 08: the gold metric sits on a flat plateau in weight-space (every
    # +/-20% single-weight perturbation moves the challenge-weighted score by < 0.008), so
    # these stay at their interpretable Session-06 values rather than overfit ~66 gold
    # labels. "abl" after each is the Session-08 ablation drop in the challenge-weighted
    # score (set the term to 0, renormalize the other six, re-measure) -- every term earns
    # its weight. See eval/calibration_report.md for the full sweep/stability/ablation.
    w_career_sim: float = 0.22  # w1: semantic career-vs-JD fit (cosine).      abl -0.0075
    w_role_match: float = 0.20  # w2: archetype match to target role family.   abl -0.0071
    w_domain_match: float = 0.15  # w3: NLP/IR/recsys problem domain.            abl -0.0066
    w_product_ratio: float = 0.12  # w4: product- vs services/consulting career.  abl -0.0059
    w_seniority_fit: float = 0.08  # w5: inside the JD experience band.           abl -0.0260 (top)
    w_built_ranking: float = 0.15  # w6: shipped a ranking/search system.         abl -0.0079
    w_lexical_evidence: float = (
        0.08  # w7: corroborating career-text evidence.   abl -0.0003 (least)
    )

    # The maps are immutable module-level constants; default_factory hands back the
    # shared proxy (a frozen dataclass forbids an unhashable mappingproxy as a bare
    # default, and these are read-only, so sharing one instance is correct).
    role_match_scores: Mapping[str, float] = field(default_factory=lambda: _ROLE_MATCH_SCORES)
    domain_match_scores: Mapping[str, float] = field(default_factory=lambda: _DOMAIN_MATCH_SCORES)
    # Conservative fallback for an off-vocabulary archetype/domain (llm_signals
    # already coerces unknowns to the generic bucket, so this is belt-and-braces).
    role_match_default: float = 0.2
    domain_match_default: float = 0.2

    penalty_per_flag: Mapping[str, float] = field(default_factory=lambda: _PENALTY_PER_FLAG)
    lexical_keyword_categories: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: _LEXICAL_KEYWORD_CATEGORIES
    )

    # career_sim: map the embedding cosine to [0, 1] by stretching the band where
    # this pool actually discriminates. BGE cosine here is high and compressed —
    # most of the 100k sit at 0.60-0.65 and the signal lives in 0.70-0.80 (Session
    # 03 EDA / shortlist threshold 0.725). A flat (cos+1)/2 would squash everyone
    # into ~0.8 and make the term near-constant; this linear window
    # (clamp((cos - floor)/(ceil - floor), 0, 1)) keeps the weight meaningful.
    cosine_floor: float = 0.60
    cosine_ceiling: float = 0.80

    # availability: blend of behavioral signals squashed to [floor, 1.0] so a weak
    # candidate is never zeroed by availability alone (it is a modifier, not fit).
    availability_weights: Mapping[str, float] = field(default_factory=lambda: _AVAILABILITY_WEIGHTS)
    # Session-08 calibration lever (0.50 → 0.70). The floor is the *gentleness* of the
    # availability multiplier: at 0.70 the worst down-weight is 30%, so a genuine ideal
    # hire who is merely less active is moved down, never buried. This is the one knob
    # Session 08 changed from the Session-06 defaults — it is both principled (the JD
    # ranks fit; reachability is a tie-breaker, not a disqualifier) and the gold-metric
    # peak: 0.50→0.70 lifts NDCG@10 0.987→1.000 and the challenge-weighted score
    # 0.985→0.992, pulls the six inactive-but-perfect tier-5/4 fits up ~50-80 ranks, and
    # 0.75 already regresses. honeypots-in-top-100 stays 0. See eval/calibration_report.md.
    availability_floor: float = 0.70
    # Recency decay: last_active is scored against a FIXED snapshot date (never
    # datetime.now()) so rank.py stays byte-identical on re-run. The window is the
    # dataset snapshot; activity older than the horizon decays to 0.
    snapshot_date: str = "2026-06-28"
    recency_horizon_days: int = 180
    notice_period_max_days: int = 180  # JD-stated 0-180 range; longer → less available
    open_to_work_false: float = 0.3  # not toggling "open" is weak-negative, not zero

    # location: India or willing-to-relocate is full credit; otherwise a mild
    # down-weight (the JD is Noida/Pune, India) — never an exclusion.
    home_country: str = "india"
    location_penalty: float = 0.85


# The single scoring configuration imported by features.py, scoring.py and the
# Session-08 calibration harness.
SCORING = ScoringConfig()
