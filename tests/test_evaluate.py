"""Tests for the eval harness join + report (``eval/evaluate.py``).

Covers the label parser (blank tiers skipped, bad tiers fail loudly), the
ranking↔gold join (system order, judged subset), and the report's metrics and
targeted checks (honeypots in top, tier ranks, inactive-but-perfect, missing
relevant). Metric *math* is covered separately in ``test_metrics.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.evaluate import GoldLabel, build_report, judged_rows, parse_gold_labels
from eval.ranking import RankedCandidate
from tests.factories import make_ranked


def _good_ranking() -> list[RankedCandidate]:
    """12 ranked candidates, no honeypot in the top; CAND_02 is a low-availability fit."""
    ranking = [make_ranked(f"CAND_{i:02d}", i) for i in range(1, 13)]
    ranking[1] = make_ranked("CAND_02", 2, availability=0.70)  # strong but inactive
    return ranking


def _gold() -> dict[str, GoldLabel]:
    return {
        "CAND_01": GoldLabel("CAND_01", 5, ""),
        "CAND_02": GoldLabel("CAND_02", 4, "great fit, long notice period"),
        "CAND_05": GoldLabel("CAND_05", 3, ""),
        "CAND_08": GoldLabel("CAND_08", 1, ""),
        "CAND_10": GoldLabel("CAND_10", 0, ""),
        "CAND_99": GoldLabel("CAND_99", 5, "not in ranking"),  # dropped by pre-filter
    }


# --------------------------------------------------------------------------- #
# Label parsing.                                                                #
# --------------------------------------------------------------------------- #
def test_parse_gold_labels_basic(tmp_path: Path) -> None:
    path = tmp_path / "gold.csv"
    path.write_text(
        "candidate_id,tier,note\nCAND_01,5,ideal\nCAND_02,,not yet labeled\nCAND_03,0,trap\n",
        encoding="utf-8",
    )
    labels = parse_gold_labels(path)
    assert set(labels) == {"CAND_01", "CAND_03"}  # blank-tier row skipped
    assert labels["CAND_01"].tier == 5
    assert labels["CAND_01"].note == "ideal"


def test_parse_gold_labels_without_note_column(tmp_path: Path) -> None:
    path = tmp_path / "gold.csv"
    path.write_text("candidate_id,tier\nCAND_01,4\n", encoding="utf-8")
    assert parse_gold_labels(path)["CAND_01"].note == ""


def test_parse_gold_labels_rejects_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "gold.csv"
    path.write_text("candidate_id,tier\nCAND_01,7\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_gold_labels(path)


def test_parse_gold_labels_rejects_non_integer(tmp_path: Path) -> None:
    path = tmp_path / "gold.csv"
    path.write_text("candidate_id,tier\nCAND_01,high\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_gold_labels(path)


def test_parse_gold_labels_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_gold_labels(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# Join.                                                                          #
# --------------------------------------------------------------------------- #
def test_judged_rows_are_in_system_order_and_judged_only() -> None:
    rows = judged_rows(_good_ranking(), _gold())
    assert [r.candidate_id for r in rows] == ["CAND_01", "CAND_02", "CAND_05", "CAND_08", "CAND_10"]
    assert [r.rank for r in rows] == [1, 2, 5, 8, 10]
    assert [r.tier for r in rows] == [5, 4, 3, 1, 0]


# --------------------------------------------------------------------------- #
# Report.                                                                        #
# --------------------------------------------------------------------------- #
def test_report_metrics_on_perfectly_ordered_gold() -> None:
    report = build_report(_good_ranking(), _gold())
    metrics = report["metrics"]
    # Judged tiers [5,4,3,1,0] are already in ideal order ⇒ NDCG == 1.0.
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["ndcg_at_50"] == pytest.approx(1.0)
    # 3 relevant (tiers 5,4,3) at ranks 1-3 of the judged list ⇒ AP == 1.0.
    assert metrics["map"] == pytest.approx(1.0)
    # 3 relevant judged candidates, P@10 = 3/10.
    assert metrics["precision_at_10"] == pytest.approx(0.3)
    assert report["challenge_weighted_score"] == pytest.approx(0.965)


def test_report_gold_coverage_flags_missing_relevant() -> None:
    report = build_report(_good_ranking(), _gold())
    gold = report["gold"]
    assert gold["labeled"] == 6
    assert gold["judged_in_ranking"] == 5
    assert gold["missing_from_ranking"] == ["CAND_99"]
    assert gold["missing_but_relevant"] == ["CAND_99"]
    assert gold["tier_histogram"] == {"0": 1, "1": 1, "2": 0, "3": 1, "4": 1, "5": 1}


def test_report_tier_ranks_and_traps() -> None:
    checks = build_report(_good_ranking(), _gold())["checks"]
    assert checks["tier_ranks"]["5"] == [1]
    assert checks["tier_ranks"]["4"] == [2]
    assert checks["tier_ranks"]["0"] == [10]
    # tier <=1 judged candidates inside top 50.
    assert checks["traps_in_top_50"] == ["CAND_08", "CAND_10"]


def test_report_inactive_but_perfect_surfaced() -> None:
    checks = build_report(_good_ranking(), _gold())["checks"]
    inactive = checks["inactive_but_perfect"]
    assert [d["candidate_id"] for d in inactive] == ["CAND_02"]  # tier 4, availability 0.70
    assert inactive[0]["rank"] == 2


def test_report_honeypots_in_top_100_clean_vs_dirty() -> None:
    clean = build_report(_good_ranking(), _gold())
    assert clean["checks"]["honeypots_in_top_100"] == 0

    dirty = _good_ranking()
    dirty[1] = make_ranked("CAND_02", 2, honeypot=True)  # a honeypot leaks into the top
    report = build_report(dirty, _gold())
    assert report["checks"]["honeypots_in_top_100"] == 1
    assert report["checks"]["honeypot_ids_in_top_100"] == ["CAND_02"]
