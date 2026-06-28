"""Evaluation harness (Session 07): gold-set labeling tooling and ranking metrics.

This package is **offline only** — it imports the same pure ``src`` building blocks
the rank-time path uses (no network/LLM), scores the shortlist, and measures the
ranking against a hand-labeled gold set. Nothing here is imported by ``rank.py``.
"""
