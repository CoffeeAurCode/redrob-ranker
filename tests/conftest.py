"""Shared test fixtures.

The ``sample_candidate`` mirrors the real ``candidate_schema.json`` (identity
under ``profile``, behavioral signals under ``redrob_signals``) and carries a
deliberately distinctive skill name so tests can prove the skills list never
leaks into the profile text.
"""

from __future__ import annotations

import pytest

from src.io_utils import Candidate

# A nonsense token that appears ONLY in the skills list, nowhere in the free
# text. If it shows up in profile text, the skills list leaked.
DISTINCTIVE_SKILL = "Flibbertigibbet"


@pytest.fixture
def sample_candidate() -> Candidate:
    """A single realistic candidate record in the real (nested) schema."""
    return {
        "candidate_id": "CAND_0000042",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "ML Engineer | Retrieval, Ranking",
            "summary": "Seven years building search and recommendation systems.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 54,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built a hybrid retrieval and ranking system in production.",
            },
            {
                "company": "SearchCo",
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2020-12-31",
                "duration_months": 36,
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Owned the offline evaluation harness measuring NDCG and MAP.",
            },
        ],
        "education": [],
        "skills": [
            {
                "name": DISTINCTIVE_SKILL,
                "proficiency": "expert",
                "endorsements": 99,
                "duration_months": 60,
            },
            {
                "name": "Python",
                "proficiency": "advanced",
                "endorsements": 12,
                "duration_months": 84,
            },
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "interview_completion_rate": 0.9,
            "notice_period_days": 30,
            "willing_to_relocate": True,
        },
    }
