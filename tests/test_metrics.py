"""Correctness of the ranking metrics against hand-computed fixtures.

These guard the numbers Session 08 will tune against, so each expected value is
worked out by hand in the comments — a silent off-by-one in the discount or the @k
truncation would mislead every later weight decision.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    RELEVANCE_CUTOFF,
    average_precision,
    binary_relevances,
    dcg,
    mean_average_precision,
    ndcg,
    precision_at_k,
)


# --------------------------------------------------------------------------- #
# DCG — linear gain, log2(rank+1) discount, 1-based ranks.                       #
# --------------------------------------------------------------------------- #
def test_dcg_known_value() -> None:
    # tiers [3, 2, 3] @3: 3/log2(2) + 2/log2(3) + 3/log2(4)
    #                   = 3/1 + 2/1.5849625 + 3/2 = 3 + 1.2618595 + 1.5 = 5.7618595
    assert dcg([3, 2, 3], 3) == pytest.approx(5.7618595, abs=1e-6)


def test_dcg_truncates_at_k() -> None:
    # k=2 ignores the 3rd/4th items entirely: 3/1 + 2/log2(3) = 4.2618595.
    assert dcg([3, 2, 3, 5], 2) == pytest.approx(4.2618595, abs=1e-6)


def test_dcg_top_position_discount_is_one() -> None:
    # A single item at rank 1 has discount log2(2) = 1, so DCG == its gain.
    assert dcg([4], 1) == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# NDCG.                                                                          #
# --------------------------------------------------------------------------- #
def test_ndcg_known_value() -> None:
    # tiers [3,2,3,0,1,2] @3: DCG@3 = 5.7618595 (above).
    # ideal sort [3,3,2,2,1,0] @3: 3/1 + 3/log2(3) + 2/2 = 3 + 1.8927893 + 1 = 5.8927893.
    # NDCG@3 = 5.7618595 / 5.8927893 = 0.9777823.
    assert ndcg([3, 2, 3, 0, 1, 2], 3) == pytest.approx(0.9777823, abs=1e-6)


def test_ndcg_perfect_order_is_one() -> None:
    assert ndcg([5, 4, 3, 1, 0], 5) == pytest.approx(1.0)


def test_ndcg_all_zero_is_zero() -> None:
    # No relevant item ⇒ ideal DCG is 0 ⇒ NDCG defined as 0.0 (no ZeroDivisionError).
    assert ndcg([0, 0, 0], 10) == 0.0


def test_ndcg_empty_is_zero() -> None:
    assert ndcg([], 10) == 0.0


def test_ndcg_k_larger_than_list() -> None:
    # @50 on a 3-item list equals @3 (truncation clamps to length).
    tiers = [2, 0, 3]
    assert ndcg(tiers, 50) == pytest.approx(ndcg(tiers, 3))


# --------------------------------------------------------------------------- #
# Binary relevance cutoff.                                                       #
# --------------------------------------------------------------------------- #
def test_binary_relevances_cutoff() -> None:
    assert binary_relevances([5, 3, 2, 0]) == [True, True, False, False]
    assert RELEVANCE_CUTOFF == 3


def test_binary_relevances_custom_cutoff() -> None:
    assert binary_relevances([5, 4, 3, 2], cutoff=4) == [True, True, False, False]


# --------------------------------------------------------------------------- #
# Precision@k — canonical hits/k.                                               #
# --------------------------------------------------------------------------- #
def test_precision_at_k_known() -> None:
    rels = [True, False, True, True, False]
    assert precision_at_k(rels, 3) == pytest.approx(2 / 3)  # 2 of top 3
    assert precision_at_k(rels, 5) == pytest.approx(3 / 5)  # 3 of top 5


def test_precision_at_k_divides_by_k_not_n() -> None:
    # 3 relevant, but P@10 divides by 10 (a short list is penalized for what it lacks).
    assert precision_at_k([True, True, True], 10) == pytest.approx(0.3)


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k([True], 0) == 0.0


# --------------------------------------------------------------------------- #
# Average precision / MAP.                                                       #
# --------------------------------------------------------------------------- #
def test_average_precision_known() -> None:
    # rels [T,F,T,T,F], R=3. P@1=1.0, P@3=2/3, P@4=3/4 (only at relevant ranks).
    # AP = (1.0 + 0.6667 + 0.75) / 3 = 0.8055556.
    assert average_precision([True, False, True, True, False]) == pytest.approx(0.8055556, abs=1e-6)


def test_average_precision_perfect() -> None:
    assert average_precision([True, True, False, False]) == pytest.approx(1.0)


def test_average_precision_no_relevant_is_zero() -> None:
    assert average_precision([False, False]) == 0.0


def test_mean_average_precision_over_queries() -> None:
    # Q1 [T,F]: AP=1.0.  Q2 [F,T]: P@2=0.5 ⇒ AP=0.5.  MAP=(1.0+0.5)/2=0.75.
    assert mean_average_precision([[True, False], [False, True]]) == pytest.approx(0.75)


def test_mean_average_precision_empty_is_zero() -> None:
    assert mean_average_precision([]) == 0.0
