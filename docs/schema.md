# Candidate schema — the real data

**Source:** `data/candidates.jsonl` (100,000 records, ~465 MB), provided in the
challenge bundle. **Verified against the actual data on 2026-06-26** — this doc
reflects the *real* field names and null-rates, not the prose in the planning
docs (several of which were wrong; see "Corrections" below).

Authoritative machine-readable schema: [`challenge/candidate_schema.json`](challenge/candidate_schema.json).
First 50 records pretty-printed: [`challenge/sample_candidates.json`](challenge/sample_candidates.json).

Every later session reads candidates through `src/io_utils.py` (typed
`Candidate` view) and `src/profile_text.py`, never raw dicts.

## Usage legend

How each field is consumed downstream:

- **EMB** — feeds the embedding text (`build_embedding_text`). Career semantics.
- **LLM** — feeds the compact LLM profile (`build_llm_profile`) for extraction.
- **SCORE** — a scoring term / multiplier in `scoring.py` (Session 06).
- **HONEY** — input to deterministic honeypot detection (Session 05).
- **—** — not used (noise or out of scope).

## Top-level structure

Each line of the JSONL is one object with 8 keys, **all present in 100% of
records** (0 nulls):

```
candidate_id      str            "CAND_0000001"   (regex ^CAND_[0-9]{7}$, 100% conform)
profile           object         identity block (see below)
career_history    array[1..10]   roles, newest-relevant first — the PRIMARY signal
education         array[0..5]
skills            array          the NOISE list — excluded from text (see Gotchas)
certifications    array          often []
languages         array
redrob_signals    object         23 behavioral signals
```

> **The two facts the planning docs got wrong:** identity fields are **nested
> under `profile`** (not top-level), and the behavioral fields — including
> `willing_to_relocate` — live **under `redrob_signals`** with names like
> `last_active_date`, `open_to_work_flag`, `notice_period_days`.

## `profile` (identity) — all fields 100% present

| field | type | example | usage |
|---|---|---|---|
| `anonymized_name` | str | "Ira Vora" | — |
| `headline` | str | "Backend Engineer \| SQL, Spark, Cloud" | — *(skill-ish; excluded from EMB, see Gotchas)* |
| `summary` | str | "Software / data professional with 6.9 years…" | **EMB, LLM** |
| `location` | str | "Toronto" / "Indore, Madhya Pradesh" | **SCORE** (location factor), LLM |
| `country` | str | "India" | **SCORE** (location factor), LLM |
| `years_of_experience` | number | 6.9 | **SCORE** (seniority fit), LLM |
| `current_title` | str | "Backend Engineer" | **EMB, LLM, SCORE** (archetype filter) |
| `current_company` | str | "Mindtree" | **SCORE** (product-vs-services), LLM |
| `current_company_size` | enum | "10001+" | SCORE (weak) |
| `current_industry` | str | "IT Services" | **SCORE** (services/consulting penalty) |

`current_company_size` / `company_size` enum: `1-10, 11-50, 51-200, 201-500,
501-1000, 1001-5000, 5001-10000, 10001+`.

## `career_history[]` — the primary signal (avg 3.0 roles/candidate)

| field | type | example | usage |
|---|---|---|---|
| `company` | str | "Dunder Mifflin" | LLM, SCORE (product-vs-services) |
| `title` | str | "Analytics Engineer" | LLM, SCORE |
| `start_date` | date | "2019-07-03" | HONEY (timeline consistency) |
| `end_date` | date \| null | null = current role | HONEY |
| `duration_months` | int | 55 | LLM, HONEY (tenure vs company age) |
| `is_current` | bool | true | SCORE (stale-coding check) |
| `industry` | str | "Paper Products" | SCORE |
| `company_size` | enum | "201-500" | SCORE (weak) |
| `description` | str (avg ~396 chars) | "Built and maintained data pipelines…" | **EMB, LLM** — the richest signal; 0% empty |

## `education[]` (0..5)

| field | type | example | usage |
|---|---|---|---|
| `institution` | str | "Lovely Professional University" | — |
| `degree` | str | "B.E." | — |
| `field_of_study` | str | "Computer Science" | SCORE (weak, optional) |
| `start_year` / `end_year` | int | 2017 / 2020 | — |
| `grade` | str \| null | "8.24 CGPA" | — |
| `tier` | enum | `tier_1..tier_4`, `unknown` | SCORE (weak, optional) |

## `skills[]` — **NOISE. Excluded from all text.** (avg 9.6/candidate)

| field | type | example | usage |
|---|---|---|---|
| `name` | str | "Fine-tuning LLMs", "Milvus", "Photoshop" | **HONEY only** (never EMB/LLM) |
| `proficiency` | enum | `beginner/intermediate/advanced/expert` | **HONEY** (expert + 0 months) |
| `endorsements` | int | 21 | — |
| `duration_months` | int | 36 | **HONEY** (but see caveat below) |

Why excluded: the dataset makes skills near-uniform across the whole pool — every
AI keyword appears on ~12% of profiles regardless of real fit, so matching them
walks straight into the trap the JD describes. See `src/profile_text.py`.

## `certifications[]` / `languages[]`

`certifications[]`: `{name, issuer, year}` — frequently `[]`. Usage: —.
`languages[]`: `{language, proficiency∈basic/conversational/professional/native}`. Usage: —.

## `redrob_signals` (23 behavioral signals) — all 100% present

Full descriptions in [`challenge/redrob_signals_doc.md`](challenge/redrob_signals_doc.md).
These form the **availability multiplier** in scoring (the JD: a perfect-on-paper
candidate who's inactive with a 5% response rate is *not actually available*).

| field | type / range | example | usage |
|---|---|---|---|
| `profile_completeness_score` | 0–100 | 86.9 | SCORE (weak) |
| `signup_date` | date | "2025-10-16" | — |
| `last_active_date` | date | "2026-05-20" | **SCORE** (recency), LLM |
| `open_to_work_flag` | bool | true (35.3% true) | **SCORE**, LLM |
| `profile_views_received_30d` | int ≥0 | 23 | SCORE (weak) |
| `applications_submitted_30d` | int ≥0 | 2 | SCORE (weak) |
| `recruiter_response_rate` | 0.0–1.0 | 0.34 | **SCORE**, LLM |
| `avg_response_time_hours` | number ≥0 | 177.8 | SCORE (weak) |
| `skill_assessment_scores` | dict[str,0–100] | {"NLP": 38.8, …} | SCORE (optional) |
| `connection_count` | int ≥0 | 356 | — |
| `endorsements_received` | int ≥0 | 35 | — |
| `notice_period_days` | int 0–180 | 60 | **SCORE**, LLM |
| `expected_salary_range_inr_lpa` | {min,max} | {18.7, 36.1} | — |
| `preferred_work_mode` | enum | onsite/hybrid/remote/flexible | SCORE (weak) |
| `willing_to_relocate` | bool | false (28.8% true) | **SCORE** (location factor), LLM |
| `github_activity_score` | **−1**..100 (−1 = no GitHub) | 9.2 | SCORE (weak) |
| `search_appearance_30d` | int ≥0 | 249 | SCORE (weak) |
| `saved_by_recruiters_30d` | int ≥0 | — | SCORE (weak) |
| `interview_completion_rate` | 0.0–1.0 | 0.9 | **SCORE**, LLM |
| `offer_acceptance_rate` | **−1**..1.0 (−1 = no history) | 0.5 | **SCORE** |
| `verified_email` | bool | true | SCORE (weak) |
| `verified_phone` | bool | true | SCORE (weak) |
| `linkedin_connected` | bool | true | — |

## EDA findings (single pass over all 100k, 2026-06-26)

- **100,000 rows.** `candidate_id` 100% conform to `^CAND_[0-9]{7}$`. No
  top-level, profile, or `redrob_signals` nulls. `summary`, `headline`, and every
  `career_history.description` are non-empty (0%).
- **Title distribution** (47 distinct titles): ~68k are 12 filler titles
  (Business Analyst, HR Manager, Mechanical/Civil Engineer, Accountant, …, each
  ~5.5–5.8k). Generic SWE titles (~2.7–3.5k each). The genuine AI/ML roles are
  rare — collected for the Session 03 archetype set:
  - **Core AI/ML:** ML Engineer (167), AI Research Engineer (153), Senior SWE (ML)
    (142), Junior ML Engineer (131), AI Specialist (130), Machine Learning
    Engineer (24), Applied ML Engineer (23), AI Engineer (21), Senior ML Engineer
    (6), Staff ML Engineer (6), Senior AI Engineer (4), Senior Applied Scientist
    (4), Lead AI Engineer (3).
  - **NLP:** NLP Engineer (14), Senior NLP Engineer (6).
  - **Search/Recsys:** Recommendation Systems Engineer (26), Search Engineer (23).
  - **Data Science:** Data Scientist (145), Senior Data Scientist (19).
  - **Bait (CV-primary, JD-excluded):** Computer Vision Engineer (132).
  - **Adjacent (filter-in, score lower):** Data Engineer (744), Senior Data
    Engineer (687), Analytics Engineer (764), Backend Engineer (704), Data Analyst
    (728), Software/Senior Software Engineer.
- **Geography:** India 75.1k (75%); rest spread across USA/Australia/Canada/UK/
  Germany/Singapore/UAE (~2.5k each). `willing_to_relocate` true for 28.8% overall.
- **Industry:** IT Services 29.9k (the consulting-only trap pool — TCS/Infosys/…),
  Software 22.4k, Manufacturing 22.3k.
- **Experience:** peaks at 5–9y (33.6k) and 9–15y (32.3k); JD targets 5–9.

## Gotchas (carry into later sessions)

1. **`headline` is excluded from embedding text** even though it's free-form,
   because it is frequently a skill-stuffed tail (`"… | SQL, Spark, Cloud"`). A
   deliberate, documented choice in `profile_text.py`; revisit if Session 03's
   sanity check wants it. The fit signal lives in `summary` + career descriptions.
2. **`−1` is a sentinel, not a low score** for `github_activity_score` (64.6% are
   −1, no GitHub) and `offer_acceptance_rate` (59.6% are −1, no history). The
   availability normalization in Session 06 must treat −1 as *unknown/neutral*,
   not as "worst possible," or it will wrongly bury 60% of the pool.
3. **Honeypot preview (for Session 05):**
   - `proficiency == "expert"` with `duration_months == 0`: **84 occurrences** —
     matches the bundle's "~80 impossible profiles." A good honeypot rule.
   - `skill.duration_months > years_of_experience × 12`: fires on **51% of the
     pool** — this is *normal noise here, NOT a honeypot signal.* Do not use it
     naively as an impossibility rule.
4. **Mojibake in free text:** some summaries/descriptions contain `�` (U+FFFD)
   where an em-dash was mis-encoded at dataset generation. Harmless for embeddings;
   `profile_text` normalizes whitespace but leaves these bytes as-is.

## Corrections vs. the planning docs

| planning doc said | reality |
|---|---|
| `title` / `current_title` at top level | under `profile.current_title` |
| `last_active`, `open_to_work`, `notice_period` | `last_active_date`, `open_to_work_flag`, `notice_period_days` |
| top-level `willing_to_relocate` | `redrob_signals.willing_to_relocate` |
| `offer_acceptance` | `offer_acceptance_rate` (with −1 sentinel) |
| submission header `rank,candidate_id,score,reasoning` | **official** is `candidate_id,rank,score,reasoning` (see `validate_submission.py`) |
