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
