# Project Summary — for Resume & Interviews

A ready-to-use summary of this project for a resume, portfolio site, or interview
prep. Every number here is pulled from the actual pipeline output committed in this
repo (`models/model_metadata.json`, `models/evaluation_report.json`), not estimated.

## Resume bullets (pick the length that fits)

**One line:**
> Built an end-to-end ML system (data pipeline → model → SHAP explainability → FastAPI → Streamlit) predicting student term-over-term persistence on a synthetic dataset, with MLflow experiment tracking and Docker deployment.

**Two lines, more technical:**
> Designed and built a full ML system predicting student re-enrollment likelihood: SQL/pandas feature engineering with leakage guardrails, time-aware train/val/test evaluation (ROC-AUC 0.785, F1 0.854), SHAP-based per-prediction explanations, and a FastAPI + Streamlit deployment, containerized with Docker and tested with 29 automated tests (pytest, FastAPI TestClient, Streamlit AppTest).

**Bullet-point resume format:**
> **Student Journey Intelligence Engine** — Personal project
> - Built a synthetic-data ML pipeline (SQLite + SQL window functions + pandas) generating leakage-checked, time-aware features for a student persistence prediction model
> - Compared Logistic Regression, Random Forest, and XGBoost with MLflow tracking; selected by validation ROC-AUC and evaluated on a genuinely held-out future-term test set (precision 0.930, recall 0.790, ROC-AUC 0.785)
> - Diagnosed and fixed a probability-calibration issue from class-weighted imbalance handling by deriving risk categories from the model's own score distribution instead of fixed thresholds
> - Implemented exact SHAP explanations (verified via automated test that reconstructed probabilities match the model's own output) and served predictions through a FastAPI service and Streamlit dashboard, containerized with Docker

## Interview talking points

### The one-sentence pitch
"I built an end-to-end ML system — from a data pipeline through a served API and dashboard — that predicts whether a student will re-enroll next term and explains why, using synthetic data since I don't have access to real student records, but built with the same rigor (leakage checks, time-aware evaluation, calibration analysis, fairness slicing) I'd want in production."

### Why synthetic data, and why that's not a cop-out
I don't have access to real student records, and public academic datasets (OULAD, the UCI dropout dataset) don't have genuine per-term longitudinal structure — repeated GPA, credits, and withdrawals across terms for the same student — which is exactly what's needed for the time-based features the project called for (GPA trend, academic momentum, credit completion rate). So I wrote a forward-in-time stochastic simulation: for each synthetic student, term-by-term, this term's outcomes (GPA, credits, withdrawals) are generated from the student's trajectory so far, and *those same-term values* — never anything from the future — decide via a logistic function whether they persist to the next term. That mirrors exactly how the real label has to be constructed in production, and it gave me a dataset where I could deliberately test leakage guardrails, not just claim to have them.

### The most interesting bug I found (this is the one to lead with)
My first pass at risk categories used fixed probability cutoffs (e.g., "risk ≥ 0.35 = high"). When I checked it against the actual data, it flagged 44% of students as high-risk against a real attrition rate of about 14% — completely unusable for an advisor tool; nobody can act on a list that's half the student body. I traced it to `class_weight="balanced"`, which I'd used to handle the class imbalance: it improves precision/recall on the minority class but distorts the model's probability calibration (I confirmed this with a calibration curve — predicted probabilities ran well below actual observed rates). The fix was to stop treating the risk category as a probability cutoff at all, and instead derive it from percentiles of the model's own score distribution on the validation set. That gave a real, monotonic gradient: students the model calls "low risk" didn't persist 6.1% of the time, "medium" 23.0%, "high" 48.5% — a usable signal instead of noise.

### Design decisions I'd defend
- **Demographic fields excluded from the model, kept for evaluation only.** `age_band`, `gender`, `first_gen_flag`, and `distance_from_campus_band` are in the dataset but never used as model inputs — only to slice performance metrics after the fact. This avoids encoding indirect discrimination into the score while still supporting a fairness review.
- **Time-aware split, not a random one.** Training only ever sees earlier calendar terms than what it's evaluated on — a random row split would leak future information into "past" predictions, which is unrealistic for a system that only ever has the past to work with.
- **SHAP explainer choice tied to model type, verified exact.** I used `LinearExplainer` for the selected logistic regression (exact on the log-odds scale, not an approximation), and wrote a test asserting `base_value + sum(shap_values)` reconstructs the model's own predicted probability exactly — so the explanations are provably trustworthy, not just plausible-looking.
- **API takes engineered features, not raw history.** Computing cumulative/trend features requires a student's full history, which belongs in the feature pipeline, not duplicated in the serving layer — a realistic separation between feature engineering and model serving.

### What I'd do differently / next steps
- Add a calibration step (Platt scaling or isotonic regression) if the system needed to expose raw probabilities to end users, not just risk categories.
- Simulate gap terms / stop-out-and-return behavior — right now every student's enrollment is consecutive, so `prior_term_enrolled_flag` isn't actually informative yet, which I flagged rather than hid.
- Add real monitoring (the dashboard's drift-check section demonstrates the *method* using train-vs-test distributions, since there's no live production traffic to actually watch).
- If this were real, none of it ships without a privacy/consent/fairness review appropriate to real student data — that's explicitly out of scope here and stated in the model card.

## Numbers to have ready

| | |
|---|---|
| Dataset | 4,000 synthetic students, 17,897 student-term rows, 16,674 labeled |
| Split | Time-aware: train ≤ term 12, val 13–14, test > 14 |
| Selected model | Logistic Regression (beat Random Forest on validation ROC-AUC: 0.764 vs 0.753) |
| Test precision / recall / F1 | 0.930 / 0.790 / 0.854 |
| Test ROC-AUC / avg. precision | 0.785 / 0.951 |
| Risk gradient (test set) | low 6.1% actual attrition / medium 23.0% / high 48.5% |
| Subgroup ROC-AUC range | ~0.755–0.818 (synthetic proxies, illustrative only) |
| Automated tests | 29, covering data validation, feature leakage, model training, explainability, API (incl. graceful degradation), dashboard |

Full technical detail: [`docs/model_card.md`](model_card.md). Architecture: [`docs/architecture.md`](architecture.md).
