"""Ranking metrics — NDCG@k, MAP, P@k — the math the gold set is read through.

These are the only feedback signal in a no-leaderboard challenge, and the scored
metric for the challenge itself is ``0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP +
0.05·P@10`` (``plan/00_OVERVIEW.md``). An off-by-one in the discount or the @k
truncation would silently mislead every Session-08 weight decision, so each
function here is pure, tiny, and pinned to hand-computed fixtures in
``tests/test_metrics.py``.

Conventions, fixed and documented so they are defensible at Stage 5:

* **Graded relevance** is the gold **tier** (0-5). NDCG uses **linear gain**
  (``gain = tier``) with the standard ``log2(rank + 1)`` discount and 1-based ranks
  — i.e. ``DCG@k = Σ_{i=1..k} tier_i / log2(i + 1)`` (rank 1 → ``/log2(2)=1``). This
  matches scikit-learn's ``ndcg_score`` default and is the easiest to hand-verify.
* **IDCG** is the DCG of the *same evaluated tiers* sorted descending, so
  ``NDCG ∈ [0, 1]`` and ``NDCG = 0`` when no judged item is relevant.
* **Binary relevance** for MAP and P@k uses the cutoff ``tier >= 3`` (see
  :data:`RELEVANCE_CUTOFF`) — tiers 3-5 are "relevant", 0-2 are not.
* All inputs are the gold tiers/relevances **in system-ranked order**, restricted
  to judged candidates (the join lives in :mod:`eval.evaluate`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Binary relevance threshold for MAP and P@k: a candidate counts as "relevant" iff
# its hand-labeled tier is at least this. Tiers run 0-5 (see eval/LABELING_GUIDE.md);
# tier 3 = "solid maybe", the lowest tier we consider a genuine hire signal.
RELEVANCE_CUTOFF = 3


def dcg(tiers: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` ranked tiers (linear gain).

    ``DCG@k = Σ_{i=1..k} tier_i / log2(i + 1)`` with 1-based rank ``i`` (so the top
    position has discount ``log2(2) = 1``). ``k`` is clamped to the list length, so
    asking for more positions than exist simply sums what is there.
    """
    return sum(tier / math.log2(rank + 1) for rank, tier in enumerate(tiers[:k], start=1))


def ndcg(tiers: Sequence[float], k: int) -> float:
    """Normalized DCG@k: ``DCG@k`` divided by the ideal ``DCG@k`` of these tiers.

    ``tiers`` are the gold tiers in the system's ranked order. The ideal ranking is
    the same tiers sorted descending; if every judged tier is 0 (nothing relevant)
    the ideal is 0 and NDCG is defined as 0.0.
    """
    ideal = dcg(sorted(tiers, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg(tiers, k) / ideal


def binary_relevances(tiers: Sequence[float], cutoff: int = RELEVANCE_CUTOFF) -> list[bool]:
    """Map graded tiers to binary relevance (``tier >= cutoff``) in ranked order."""
    return [tier >= cutoff for tier in tiers]


def precision_at_k(relevances: Sequence[bool], k: int) -> float:
    """Fraction of the top ``k`` ranked items that are relevant: ``hits / k``.

    Uses the canonical ``/k`` denominator (not ``/min(k, n)``), so a short ranked
    list is penalized for the relevant items it could not supply. ``k <= 0`` → 0.0.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for relevant in relevances[:k] if relevant)
    return hits / k


def average_precision(relevances: Sequence[bool]) -> float:
    """Average precision for one ranked list of binary relevances.

    ``AP = (1/R) Σ_k [rel_k · P@k]`` over ranks ``k`` where item ``k`` is relevant,
    with ``R`` the total number of relevant items in the list. ``R = 0`` → 0.0. This
    is the per-query term; :func:`mean_average_precision` averages it over queries.
    """
    total_relevant = sum(1 for relevant in relevances if relevant)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_relevant


def mean_average_precision(queries: Sequence[Sequence[bool]]) -> float:
    """Mean of :func:`average_precision` over queries (one JD here ⇒ MAP == AP)."""
    if not queries:
        return 0.0
    return sum(average_precision(query) for query in queries) / len(queries)
