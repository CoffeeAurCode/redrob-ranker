# Gold-set labeling guide (Session 07)

This gold set is the **only feedback signal** in a no-leaderboard challenge. The
weights in `src/config.py` are tuned *to these labels* (Session 08), so the labels
must be **human judgment**, not the model's own score read back to itself. Claude
builds the review sheet and a *suggested* tier per candidate; a person must read
each profile against the rubric and write the final `tier`.

## What you are labeling

Relevance of the candidate to **this** JD: a Senior Applied ML / Search / Ranking /
Recommendation engineer (see `artifacts/jd_reference.json` for the full rubric).
You are judging **fit / "how good a hire is this"**, read from the *career history*
— not keyword presence, and not the system's score.

The "ideal" (rubric `ideal_text`): 6–8 yrs total, ~4–5 in applied ML at **product**
companies; shipped an **end-to-end search / ranking / recommendation** system to
real users; embeddings-based retrieval, vector search, rigorous **evaluation**
(NDCG/MRR/MAP, A/B tests); hands-on coding in the last 18 months; NLP/IR domain.

## The tiers (write one of 0–5 in the `tier` column)

| Tier | Meaning | Typical profile |
|------|---------|-----------------|
| **5** | Ideal hire — would fast-track | Applied-ML / search / recsys at product cos; shipped end-to-end ranking/retrieval to real users at scale; evaluation rigor; in-band seniority; no disqualifiers |
| **4** | Strong — clearly interview | Squarely on-target role + domain, real production ML systems; maybe one soft gap (lighter eval story, slightly off seniority, modest scale) |
| **3** | Solid maybe — borderline interview | Genuine applied ML, but the IR/ranking evidence is thin or adjacent (e.g. general ML/NLP, data science with some production), or product signal is mixed |
| **2** | Weak — probably not | Real engineer but off-target (generic SWE/data-eng), or on-target but carrying a hard-disqualifier the JD names (CV/speech-primary, consulting-only) that outweighs the fit |
| **1** | Very weak | Mostly off-domain (CV/speech with no NLP/IR), thin production evidence, or stacked disqualifiers |
| **0** | Irrelevant / trap | Non-tech, pure keyword-stuffer with no real evidence, **or any honeypot** (impossible profile — always 0) |

### Binary cutoff (used by MAP and P@10)

**`tier >= 3` is "relevant".** Tiers 3–5 are genuine hire signals; 0–2 are not.
This is fixed in `eval/metrics.py` (`RELEVANCE_CUTOFF`) and must not drift.

## Rules of thumb (the calls that matter)

- **Label fit, not availability.** A perfect-fit candidate who is inactive / has a
  long notice period / isn't "open to work" is **still a strong hire** — tier 4–5.
  Availability is a *minor* multiplier in the scorer, not a relevance signal; the
  gold set must encode "great hire" independent of how reachable they are. This is
  exactly the **inactive-but-perfect** case the harness checks: the ranker should
  still surface them, and the labels are what prove it.
- **Honeypots are always 0.** If a profile is flagged impossible (`honeypot=true`
  in the sheet — e.g. "expert in N skills, 0 months used"; one role longer than the
  whole career), it is a deliberate trap → tier 0, regardless of how good it reads.
- **Hard disqualifiers cap the tier.** The JD explicitly excludes: pure-research /
  no production, recent-LLM-glue-only, stale coding (no hands-on in ~18 mo),
  title-chasing job hopping, framework-demo-only, **consulting/services-only**
  careers, and **CV/speech/robotics-primary without NLP/IR**. A profile that is
  otherwise strong but sits in one of these buckets is tier ≤ 2 — the trait the JD
  says "do NOT want" dominates.
- **Career text over title and skills.** Titles are noisy (a "Computer Vision
  Engineer" whose history is all retrieval/ranking can be a real fit; an "AI
  Engineer" who only ran LangChain tutorials is not). The `skills` list is uniform
  noise by design — ignore it. Read the role descriptions and the evidence span.
- **Reward plain-language fits.** A great career history that never says "RAG" or
  "Pinecone" but clearly built and evaluated a ranking system is tier 4–5. Do not
  reward buzzword density.

## Workflow

1. Open `eval/gold_review.csv` (one row per candidate). Columns include id, title,
   years, summary/career excerpts, the LLM signals, honeypot flag, the full score
   breakdown, and a **`suggested_tier`** + **`suggested_rationale`** (Claude's draft
   — a starting point to overwrite, *not* an answer).
2. Read each profile. Write your tier in the **`tier`** column; add a short **`note`**
   for any tricky call (these become the deck's honest examples).
3. Save the final labels to **`eval/gold_labels.csv`** with columns
   `candidate_id,tier[,note]` (one row per candidate; rows with a blank `tier` are
   ignored by the evaluator). Keep the gold set **out of any fitting** — it is
   evaluation only.
4. Run `python eval/evaluate.py` to see NDCG@10/@50, MAP, P@10 and the
   honeypot/stuffer/inactive-but-perfect checks.

## Coverage to aim for (the sheet is pre-stratified for this)

A good gold set deliberately spans: clear fits (top), plausible-but-flawed,
keyword stuffers, **CV-primary bait**, consulting-only, a couple of **honeypots**,
and **inactive-but-perfect** profiles, pulled from across the score distribution
(top / middle / low) — so the metrics actually test the ranker rather than
rubber-stamping its top.
