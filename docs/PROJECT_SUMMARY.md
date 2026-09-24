# Project Summary — for Resume & Interviews

A ready-to-use summary of this project for a resume, portfolio site, or interview
prep. Every number here is pulled from the actual pipeline output committed in this
repo (`models/model_metadata.json`, `models/evaluation_report.json`), not estimated.

## Resume bullets (pick the length that fits)

**One line:**
> Built an end-to-end ML system (data pipeline → hyperparameter-tuned ensemble → SHAP explainability → hardened FastAPI → Streamlit) predicting student term-over-term persistence on a synthetic dataset, with MLflow tracking and Docker deployment.

**Two lines, more technical:**
> Designed and built a full ML system predicting student re-enrollment likelihood: SQL/pandas feature engineering with leakage guardrails, Optuna-tuned Logistic Regression/Random Forest/XGBoost combined into a calibrated soft-voting ensemble, SHAP explanations with an empirically-detected score scale, and a production-hardened FastAPI (auth, rate limiting, Prometheus metrics) + Streamlit deployment, containerized with Docker and covered by 52 automated tests.

**Bullet-point resume format:**
> **Student Journey Intelligence Engine** — Personal project
> - Built a synthetic-data ML pipeline (SQLite + SQL window functions + pandas) simulating realistic student enrollment, including stop-out-and-return behavior, with leakage-checked, time-aware features
> - Tuned Logistic Regression, Random Forest, and XGBoost with Optuna and combined them into a calibrated soft-voting ensemble, selected by validation ROC-AUC and evaluated on a genuinely held-out future-term test set
> - Diagnosed and fixed a probability-calibration issue from class-weighted imbalance handling (verified Brier score improvement via `CalibratedClassifierCV`), and built a model-agnostic SHAP explainer for the ensemble after finding the closed-form explainer would have crashed on it
> - Hardened the FastAPI service with API-key auth, per-endpoint rate limiting, structured access logging, and Prometheus metrics; served through a Streamlit dashboard with a business-case sensitivity analysis, all containerized with Docker

## Interview talking points

### The one-sentence pitch
"I built an end-to-end ML system — from a data pipeline through a hyperparameter-tuned ensemble, explainability, a hardened API, and a dashboard — that predicts whether a student will re-enroll next term and explains why, using synthetic data since I don't have access to real student records, but built with the same rigor (leakage checks, time-aware evaluation, calibration analysis, fairness slicing) I'd want in production."

### Why synthetic data, and why that's not a cop-out
I don't have access to real student records, and public academic datasets (OULAD, the UCI dropout dataset) don't have genuine per-term longitudinal structure — repeated GPA, credits, and withdrawals across terms for the same student — which is exactly what's needed for the time-based features the project called for (GPA trend, academic momentum, credit completion rate). So I wrote a forward-in-time stochastic simulation: for each synthetic student, term-by-term, this term's outcomes (GPA, credits, withdrawals) are generated from the student's trajectory so far, and *those same-term values* — never anything from the future — decide via a logistic function whether they persist to the next term, including a realistic chance of stopping out for a term or two before returning. That mirrors exactly how the real label has to be constructed in production, and it gave me a dataset where I could deliberately test leakage guardrails, not just claim to have them.

### Bugs I found and fixed (the strongest interview material in this project)

**1. The calibration/risk-threshold bug.** My first pass at risk categories used fixed probability cutoffs (e.g., "risk ≥ 0.35 = high"). When I checked it against the actual data, it flagged 44% of students as high-risk against a real attrition rate of about 14% — completely unusable for an advisor tool; nobody can act on a list that's half the student body. I traced it to `class_weight="balanced"`, which I'd used to handle class imbalance: it improves precision/recall on the minority class but distorts probability calibration. Two fixes, at two different points in the project: first, I derived risk categories from percentiles of the model's own score distribution instead of fixed cutoffs — an immediate, usable fix. Later, I went back and fixed the underlying calibration itself with `CalibratedClassifierCV`, and *measured* the improvement rather than assuming it worked (Brier score 0.208 → 0.177 on the held-out test set, verified with a before/after calibration-curve plot).

**2. A generator bug that silently did nothing.** When I added stop-out-and-return simulation (students skipping 2-3 terms before re-enrolling), I wrote the probability logic, ran the generator, and got zero actual gaps in the output — the feature I'd just built had no effect at all. The bug: my simulation loop recomputed `term_order` from `entry_term_order + term_number - 1` at the top of every iteration, silently discarding the gap I'd advanced it by in the previous one. I only caught it because I checked the generated data for gaps instead of assuming the code worked because it ran without errors. Fixed by switching to a `while` loop with explicitly tracked state, and wrote a regression test for exactly this failure mode.

**3. An explainability integration gap I found by not avoiding it.** When I added hyperparameter tuning and a soft-voting ensemble, the ensemble genuinely won the model comparison. But my SHAP explainability code assumed a single `sklearn.Pipeline` with a `.named_steps` attribute — a `VotingClassifier` has neither, and would have crashed the moment the ensemble was selected (which it was, in the actual run). I could have added an artificial rule blocking the ensemble from winning to dodge the problem, but that would have been avoiding the interesting part. Instead I built a second explainer path: a model-agnostic SHAP `PermutationExplainer` on raw features, with categorical columns encoded to fixed integer codes because SHAP's masker calls `np.isclose` on the background data and raises on raw strings. Along the way I also found and fixed a real indexing bug — reading the wrong class's base value out of a multi-class SHAP output — by writing a test that checks the SHAP values reconstruct the model's actual probability exactly, not approximately.

### Design decisions I'd defend
- **Demographic fields excluded from the model, kept for evaluation only.** `age_band`, `gender`, `first_gen_flag`, and `distance_from_campus_band` are in the dataset but never used as model inputs — only to slice performance metrics after the fact. This avoids encoding indirect discrimination into the score while still supporting a fairness review.
- **Time-aware split, not a random one.** Training only ever sees earlier calendar terms than what it's evaluated on — a random row split would leak future information into "past" predictions, which is unrealistic for a system that only ever has the past to work with.
- **The ensemble had to earn its spot.** I evaluated it as one more candidate alongside the individually-tuned models, selected purely by validation ROC-AUC — it won by a genuine (if narrow) margin, not because ensembling is assumed to be better. I'd rather report an honest small win than manufacture a bigger one.
- **SHAP explainer's score scale is detected empirically, not assumed per model name.** `LinearExplainer` is exact on log-odds; `TreeExplainer`'s scale actually depends on the specific tree algorithm (probability-space for scikit-learn's Random Forest, typically log-odds for gradient-boosted trees like XGBoost) — since I couldn't even verify XGBoost locally (environment-blocked, see below), I didn't want to bake in an assumption I couldn't check. So the code checks which formula reconstructs a real `predict_proba` call and uses that.
- **API takes engineered features, not raw history.** Computing cumulative/trend features requires a student's full history, which belongs in the feature pipeline, not duplicated in the serving layer — a realistic separation between feature engineering and model serving.
- **The intervention-impact analysis exposes its assumptions as sliders, not defaults.** There's no recorded outreach history in this dataset to estimate a real effect size from, so participation rate and effectiveness are parameters the user sets, not numbers presented as findings — the difference between a sensitivity analysis and a fabricated prediction.

### Environment limitations I worked around honestly, instead of hiding
- **XGBoost needs `libomp`**, unavailable on this dev machine (no Homebrew) — the code detects this at import time, skips XGBoost gracefully, and logs why. It's fully wired up and runs in the Docker image (Linux) and CI.
- **This session's browser tools can't reach a `localhost` server I start myself** — so the dashboard is verified with Streamlit's own headless test harness (`AppTest`), which actually executes the script and clicks the Predict form's submit button, rather than a real screenshot. I said so explicitly rather than presenting a mockup as a real screenshot.
- **No Docker daemon in this environment** — the Dockerfile/compose setup is written against standard patterns and reviewed carefully, but flagged as the first thing to verify rather than claimed as tested.

### What I'd do differently / next steps
- A real deployment using this ensemble would want to budget for the Permutation explainer's higher per-request cost (roughly 10-20x more model evaluations than the closed-form explainers) or cache explanations.
- Widen the Optuna search (currently 15-25 trials per model type, sized for reasonable local runtime) if this were a real retraining job with more compute budget.
- If this were real, none of it ships without a privacy/consent/fairness review appropriate to real student data — that's explicitly out of scope here and stated in the model card.

## Numbers to have ready

| | |
|---|---|
| Dataset | 4,000 synthetic students, 17,581 student-term rows, 16,328 labeled (incl. stop-out-and-return behavior) |
| Split | Time-aware: train ≤ term 12, val 13–14, test > 14 |
| Models compared | Logistic Regression, Random Forest, (XGBoost when available) — each Optuna-tuned, plus a soft-voting ensemble |
| Selected model | Ensemble (LR + RF), won validation ROC-AUC 0.706 vs. 0.704 for either individual tuned model |
| Test precision / recall / F1 | 0.767 / 0.946 / 0.847 |
| Test ROC-AUC / avg. precision | 0.689 / 0.836 |
| Calibration fix | Brier score 0.208 → 0.177 (verified before/after) |
| Risk gradient (test set) | low 18.9% actual attrition / medium 36.3% / high 62.6% |
| Subgroup ROC-AUC range | ~0.62–0.72 (synthetic proxies, illustrative only) |
| Automated tests | 52, covering data validation, feature leakage, model training/tuning, explainability (incl. the ensemble path), API (incl. auth, rate limiting, graceful degradation), dashboard, intervention-analysis arithmetic |

Full technical detail: [`docs/model_card.md`](model_card.md). Architecture: [`docs/architecture.md`](architecture.md).
