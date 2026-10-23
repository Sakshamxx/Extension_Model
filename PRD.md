# PRD — Student Learning Progress Model (LLM Extension Integration)

## Overview

A scikit-learn Random Forest pipeline (`model.pkl`) that classifies each study session's
learning progress as one of three categories, so the LLM extension can ground its
feedback and recommendations in a per-session progress signal.

- Trained in `main.ipynb` on `student_learning_progress_900_rows.csv` (900 sessions, 60 students).
- Held-out test performance (student-grouped split): **Random Forest — accuracy 0.876, macro F1 0.878**,
  vs Logistic Regression — accuracy 0.831, macro F1 0.840. Random Forest selected.

## Model

- `model.pkl`: full preprocessing + classifier pipeline (joblib).
  - Preprocessing inside the pipeline: one-hot encoded `subject`, median-imputed numerics.
  - Classifier: `RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced', random_state=42)`.
- Classes: `On Track`, `Improving`, `Needs Attention`.

## Inputs

One record per study session, exactly these fields (any other columns are ignored):

| Field                   | Type   | Notes                                                                      |
| ----------------------- | ------ | -------------------------------------------------------------------------- |
| `subject`             | string | e.g. `Mathematics`, `Computer Science` (case-normalized to Title Case) |
| `session_number`      | int    | 1–5                                                                       |
| `study_duration_min`  | float  | 0 or > 180 are treated as invalid (imputed)                                |
| `questions_attempted` | int    | > 0                                                                        |
| `questions_completed` | int    | ≤ `questions_attempted`; `completion_rate = completed / attempted`    |
| `completion_rate`     | float  | 0–1; recompute from question counts when both are available               |
| `previous_test_score` | float  | may be missing (first session)                                             |
| `current_test_score`  | float  | 0–100                                                                     |

Excluded and must never be sent: `student_id`, `session_date`, `next_test_score`, `score_change`
(identifier / no predictive meaning / future information / label leakage respectively).

## Outputs

- `predicted_status`: one of `On Track` / `Improving` / `Needs Attention`.
- `probabilities`: per-class probability dict (sums to 1), for confidence-based LLM prompting.

Example:

```python
import joblib, pandas as pd
mdl = joblib.load('model.pkl')
record = pd.DataFrame([{ 'subject': 'Computer Science', 'session_number': 3,
    'study_duration_min': 38, 'questions_attempted': 18, 'questions_completed': 13,
    'completion_rate': 0.722, 'previous_test_score': 64, 'current_test_score': 73 }])
mdl.predict(record)          # -> ['Improving']
mdl.predict_proba(record)    # -> [[Improving 0.774, Needs Attention 0.022, On Track 0.204]]
```

## Integration steps

1. Ship `model.pkl` with the extension's backend (or bundle it and load locally).
2. Load once at startup: `joblib.load('model.pkl')` — do not reload per request.
3. Before calling, map the session the extension already has to the input schema above;
   compute `completion_rate` from question counts when possible, otherwise omit and let
   the pipeline impute.
4. Call `predict` + `predict_proba`, and pass the label plus probabilities (e.g.
   "predicted: Improving, confidence 0.77") to the LLM as context for its coaching response.
5. Confidence below ~0.5: have the LLM avoid stating a progress category and ask the
   user for more details instead.
6. Data handling: never send prohibited fields (see Inputs); they break the contract and
   in the case of `score_change` would leak the label.

## Limitations

- Labels are rule-generated on a synthetic dataset. Accuracy reflects consistency with those
  rules, not real-world educational effectiveness.
- 60 students; grouped-split scores may not generalize to new populations.
