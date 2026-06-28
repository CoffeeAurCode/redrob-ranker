"""Test package marker.

Present so shared test helpers import unambiguously as ``tests.factories`` (under
both pytest and mypy --strict); pytest still auto-discovers ``conftest.py`` here.
"""
