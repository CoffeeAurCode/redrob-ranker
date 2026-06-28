# Calibration report — Session 08

**Goal:** tune the scoring weights/thresholds against the gold set until they are
strong **and stable**, honeypots-in-top-100 is 0, and the qualitative traps behave.

**Outcome:** one deliberate change — soften the availability multiplier
(`availability_floor` 0.50 → 0.70). The seven base-fit weights stay at their
Session-06 values. Challenge-weighted gold score **0.9854 → 0.9924**; honeypots in
top 100 stay **0**; the configuration is stable to ±20% perturbation (worst weighted
swing **0.0071**).

Reproduce every table below from the repo root:

```bash
python eval/evaluate.py                       # the locked metrics → eval/metrics.json
python eval/calibrate.py                       # baseline + sweep + stability + ablation
python eval/calibrate.py --sweep               # just the coordinate sweep
```

The gold set is **66 judged** of 68 labels (2 pool-honeypots are pre-filtered out),
tier histogram `{0:1, 1:14, 2:18, 3:6, 4:9, 5:18}`. Metrics are computed over the
system ranking **restricted to judged candidates** (standard pooled eval); binary
cutoff for MAP/P@10 is tier ≥ 3.

---

## 1. Before / after

| metric | baseline (floor 0.50) | **locked (floor 0.70)** | Δ |
|---|---|---|---|
| NDCG@10 | 0.9873 | **1.0000** | +0.0127 |
| NDCG@50 | 0.9784 | **0.9791** | +0.0007 |
| MAP | 0.9882 | **0.9911** | +0.0029 |
| P@10 | 1.0000 | **1.0000** | 0 |
| **challenge-weighted** (`.50·N@10+.30·N@50+.15·MAP+.05·P@10`) | 0.9854 | **0.9924** | **+0.0070** |
| honeypots in top 100 | 0 | **0** | 0 |
| disqualifier-flagged in top 10/50/100 | 0/0/0 | **0/0/0** | 0 |

**What moved.** Raising the floor lifts NDCG@10 to 1.0 (a tier-5 ideal hire now
correctly sits above a tier-4 at the very top of the judged list) and pulls the
inactive-but-perfect cohort up the full ranking (§5). NDCG@50 and MAP nudge up; P@10
was already saturated.

---

## 2. The one lever — `availability_floor` (the main NDCG@50 / inactive-but-perfect knob)

`availability = floor + (1 − floor)·blend`. The floor is how *gentle* the multiplier
is: at 0.70 the worst possible down-weight is 30%, so a genuine ideal hire who is
merely less active is **moved down, not buried**. This is principled independent of
the gold set — the JD ranks *fit*; reachability is a tie-breaker, not a
disqualifier — and it is also the gold-metric peak:

| floor | NDCG@10 | NDCG@50 | MAP | weighted | inactive worst rank |
|---|---|---|---|---|---|
| 0.50 (Session-06) | 0.9873 | 0.9784 | 0.9882 | 0.9854 | 340 |
| 0.55 | 0.9873 | 0.9784 | 0.9882 | 0.9854 | 327 |
| 0.60 | 0.9873 | 0.9783 | 0.9889 | 0.9855 | 318 |
| 0.65 | 1.0000 | 0.9787 | 0.9897 | 0.9921 | 308 |
| **0.70 (locked)** | **1.0000** | **0.9791** | **0.9911** | **0.9924** | **304** |
| 0.75 | 0.9873 | 0.9788 | 0.9911 | 0.9859 | 297 |
| 0.80 | 0.9867 | 0.9789 | 0.9921 | 0.9858 | 285 |

NDCG@10 = 1.0 holds across the band **[0.65, 0.70]**; the weighted score peaks at
0.70 and **regresses at 0.75** (the top tier-5/tier-4 pair flips back). 0.70 is the
top of the stable plateau, so it is the chosen value. Honeypots-in-top-100 = 0 at
every floor.

---

## 3. Why the seven base weights were **not** retuned

A per-weight coordinate sweep (each weight swept ±0.08 with the other six
renormalized to keep Σ = 1.0) moves the challenge-weighted score by only ~0.005–0.007
in either direction — the gold metric sits on a **flat plateau** in weight-space. With
only 66 judged labels, chasing those 4th-decimal gains would overfit. The weights keep
their interpretable Session-06 values, each justified by a JD requirement (see
`src/config.py` → `ScoringConfig`) and corroborated by the ablation in §6. This is the
session's explicit guidance: *favor a simple, stable, explainable configuration over a
fragile high score.*

---

## 4. Stability — ±10% and ±20% perturbation of the locked config

Each weight (and the floor) was perturbed by ±10% and ±20%, the other six weights
renormalized to keep Σ = 1.0, and the gold metrics recomputed.

* **Worst challenge-weighted swing across all ±20% perturbations: 0.0071.**
* **honeypots-in-top-100 = 0 and traps-in-top-50 = 0 for every single perturbation.**
* NDCG@10 stays 1.0 under most perturbations; it dips to 0.987 only when `career_sim`
  or `lexical_evidence` is pushed up ≥ +10%, or the floor up ≥ +10% (i.e. toward the
  0.75 cliff in §2) — and even then the weighted score stays ≥ 0.985.

The model is not balanced on a knife-edge: no single knob, moved a fifth of its value,
breaks a check or materially moves the score.

---

## 5. Inactive-but-perfect — down-weighted, not buried

The six gold tier-5/4 *inactive-but-perfect* fits (strong career, low availability)
were the headline Session-07 finding: the harsh floor-0.50 multiplier buried them at
ranks ~220–284. Softening to 0.70 lifts each ~30–60 places while still ranking them
**below** the equally-strong-but-reachable hires in the top ~16 — exactly the intended
"gentle multiplier" behaviour.

| candidate | tier | base_fit | availability | rank @0.50 | **rank @0.70** |
|---|---|---|---|---|---|
| CAND_0060072 | 5 | 0.937 | 0.83 | 220 | **190** |
| CAND_0041611 | 5 | ~0.94 | 0.83 | 235 | **202** |
| CAND_0092278 | 5 | 0.942 | 0.81 | 256 | **203** |
| CAND_0052498 | 4 | — | 0.83 | 233 | **206** |
| CAND_0088438 | 5 | — | 0.82 | 269 | **218** |
| CAND_0065927 | 4 | — | 0.82 | 284 | **223** |

---

## 6. Ablation — does every term earn its weight?

Each base term is dropped (weight → 0, the other six renormalized) from the locked
config and the gold metric re-measured. Every term has a **negative** Δ when removed,
so every term pulls its weight:

| dropped term | weight | Δ challenge-weighted | note |
|---|---|---|---|
| seniority_fit | 0.08 | **−0.0260** | most load-bearing (drops NDCG@10 to 0.956) |
| built_ranking | 0.15 | −0.0079 | the JD must-have; buries the inactive set (worst rank → 392) |
| career_sim | 0.22 | −0.0075 | the semantic backbone |
| role_match | 0.20 | −0.0071 | |
| domain_match | 0.15 | −0.0066 | |
| product_ratio | 0.12 | −0.0059 | |
| lexical_evidence | 0.08 | −0.0003 | least load-bearing on the gold sample |

`lexical_evidence` is nearly inert on these 66 labels. It is **kept** deliberately:
(a) a −0.0003 gold delta is noise at this sample size — dropping a principled term on
it would itself be overfitting; (b) it is a cheap, transparent keyword-evidence guard
on the **unlabeled 99.9%** of the pool, where the embedding/LLM terms have no gold
oversight. It carries the smallest weight precisely because it is corroboration, not a
primary signal.

---

## 7. Trap verification (qualitative, required)

All checked under the **locked** config (`python eval/calibrate.py` for the harness
view; per-candidate breakdowns reproduced from the rank-time path).

| trap class | example | result |
|---|---|---|
| **Honeypot** (reads ideal, impossible) | CAND_0037000 — base_fit **0.914**, "Search Engineer" | honeypot rule → **final 0.000, rank #1167** |
| **Honeypots in top 100** | the 41-flag set | **0** in top 100 (the single shortlisted honeypot sits at #1167; the other 40 never made the shortlist) |
| **Keyword stuffers out** | top-30 of the ranking | all genuine Search/Recsys/Applied-ML/NLP/AI archetypes; **0** disqualifier-flagged in top 100; skills-list excluded from all text |
| **Plain-language fit in** | CAND_0052195 — title "Computer Vision Engineer", career is recsys/ranking (tier 4) | **#146**, no penalty — the misleading title does not bury it (career_sim + LLM role/domain rescue it) |
| consulting_only | CAND_0065835 | penalty 0.35 → **#870** |
| cv_primary bait | CAND_0081709 | penalty 0.35 → **#847** |
| stale_coding | CAND_0061175 | penalty 0.20 → **#681** |
| job_hopper | CAND_0070930 | penalty 0.15 → **#855** |
| **Inactive-but-perfect** handled | §5 cohort | down-weighted to ranks ~190–223, **not buried** |

---

## 8. Locked configuration

`src/config.py → ScoringConfig` (each weight annotated with its rationale and its
ablation Δ):

```
w_career_sim     0.22     cosine_floor       0.60      availability_floor   0.70  ← changed
w_role_match     0.20     cosine_ceiling     0.80      location_penalty     0.85
w_domain_match   0.15     snapshot_date  2026-06-28    home_country        india
w_product_ratio  0.12     recency_horizon_days 180
w_seniority_fit  0.08     notice_period_max_days 180
w_built_ranking  0.15     penalties: consulting_only/cv_primary 0.35, pure_research/
w_lexical_evid.  0.08                langchain_only_recent 0.30, stale_coding 0.20, job_hopper 0.15
   Σ = 1.00
```

Determinism: the sweep is over a fixed gold file, a fixed shortlist, and the fixed
2026-06-28 recency snapshot — `eval/metrics.json` is byte-identical on re-run.
