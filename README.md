<div align="center">

# 🎓 Student Learning Progress — Model Comparison

**Binary-to-3-class progress classifier** · Logistic Regression vs Random Forest
Predicts a study session's `progress_status` — `On Track` · `Improving` · `Needs Attention`

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Better model: **Random Forest** · Accuracy **0.876** · Macro F1 **0.878**

</div>

---

## 📋 Overview

Learning platforms generate study-session logs that need interpreting. This project trains and
compares two classifiers on **900 sessions from 60 students** and ships the better model as a
ready-to-load pipeline (`model.pkl`). A companion PRD (`prd.md`) documents exactly how the
existing LLM extension should call it.

Built with an end-to-end pipeline: **data cleaning → grouped train/test split → preprocessing →
model training → evaluation → artifact export**.

## ✨ Key Features

- **Leakage-safe feature selection** — only fields available at prediction time; `score_change`,
  `next_test_score` (future), and `student_id` (identifier) are excluded
- **Student-grouped split** — `GroupShuffleSplit` guarantees no student's sessions appear in both
  train and test sets
- **Defensive data cleaning** — mixed date formats, case-inconsistent subjects, invalid study
  durations, and impossible question counts are all detected and repaired
- **Deployment-ready artifact** — the saved `model.pkl` bundles preprocessing + classifier, so
  new records need no separate cleaning code

## 📂 Project Structure

```
extension_model/
├── main.ipynb                              # End-to-end training & evaluation notebook
├── model.pkl                               # Trained Random Forest pipeline (preprocess + model)
├── prd.md                                  # Product spec: inputs, outputs, LLM extension integration
├── student_learning_progress_900_rows.csv  # Source dataset (900 sessions, 60 students)
├── requirements.txt                        # Dependencies
├── .gitignore
└── README.md
```

## 📊 Dataset

| | |
|---|---|
| **Rows** | 900 study sessions |
| **Students** | 60 (5 sessions each) |
| **Subjects** | Mathematics, Physics, Chemistry, Biology, English, Computer Science |
| **Target** | `progress_status` → `On Track` (381) · `Improving` (288) · `Needs Attention` (231) |

**Known data quality issues handled in cleaning:**
- 10 subject strings with inconsistent casing → normalized to 6 canonical subjects
- Two date formats (`YYYY-MM-DD` and `DD/MM/YYYY`) → parsed to a single datetime
- 3 durations > 180 min and 2 of 0 min → treated as invalid and imputed
- 9 missing test scores, 10 incomplete question counts → median-imputed;
  `completion_rate` recomputed from question counts wherever they are valid

## 🧠 Model Selection

| Model | Preprocessing | Config |
|---|---|---|
| **Logistic Regression** | OneHotEncode `subject`, median-impute + StandardScale numerics | `max_iter=2000`, multinomial |
| **Random Forest** ✅ | OneHotEncode `subject`, median-impute numerics (no scaling needed) | `n_estimators=300`, `min_samples_leaf=3`, `class_weight=balanced`, `random_state=42` |

### 📈 Test Results

Evaluated on the same held-out test set (225 sessions, 15 unseen students):

| Metric | Logistic Regression | Random Forest 🏆 |
|---|:---:|:---:|
| Accuracy | 0.831 | **0.876** |
| Macro Precision | 0.838 | 0.875 |
| Macro Recall | 0.841 | 0.886 |
| **Macro F1** | 0.840 | **0.878** |

**Why Random Forest wins:** it captures non-linear interactions between study behavior and
test outcomes, improves recall on the minority `Needs Attention` class (helped by class
weighting), and wins on both primary (macro F1) and secondary (accuracy) criteria.

**Trade-off acknowledged:** Linear Regression loses 4.5 accuracy points but stays fully
interpretable — its one-hot coefficients are directly readable. The forest is chosen for
performance; interpretability is recovered through the feature-importance plot in
`main.ipynb`.

### Graphs produced (rendered in `main.ipynb`)

1. Class distribution
2. Metrics heatmap (both models side by side)
3. Confusion matrix — Logistic Regression
4. Confusion matrix — Random Forest
5. Random Forest feature importance

## 🚀 Getting Started

### Setup

```bash
# Clone the repository
git clone https://github.com/Sakshamxx/Extension_Model.git
cd Extension_Model

# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# One-shot: clean, train, evaluate, render graphs, and re-save model.pkl
jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Or explore interactively:

```bash
jupyter notebook   # opens on http://localhost:8888 — run main.ipynb cell by cell
```

### Quick sanity check (no notebook needed)

`model.pkl` ships pre-trained, so predictions work immediately:

```bash
python3 -c "import joblib, pandas as pd; m = joblib.load('model.pkl'); \
print(dict(zip(m.classes_, m.predict_proba(pd.DataFrame([{'subject': 'Mathematics', 'session_number': 3, 'study_duration_min': 38, 'questions_attempted': 18, 'questions_completed': 13, 'completion_rate': 0.722, 'previous_test_score': 64, 'current_test_score': 73}]))[0])))"
```

```text
{'Improving': 0.776, 'Needs Attention': 0.025, 'On Track': 0.199}
```

## 💻 Model API

**Input** — one row per study session, exactly these fields:

```python
new = pd.DataFrame([{
    'subject': 'Computer Science',        # string
    'session_number': 3,                  # int, 1-5
    'study_duration_min': 38.0,           # float; 0 or >180 → imputed
    'questions_attempted': 18,            # int, > 0
    'questions_completed': 13,            # int, ≤ attempted
    'completion_rate': 0.722,             # float, 0-1
    'previous_test_score': 64.0,          # float, may be missing on first session
    'current_test_score': 73.0,           # float, 0-100
}])
```

```python
import joblib
mdl = joblib.load('model.pkl')
mdl.predict(new)            # → ['Improving']
mdl.predict_proba(new)      # → array([[0.204, 0.774, 0.022]])  # order: On Track, Improving, Needs Attention
```

**Never send:** `student_id`, `session_date`, `next_test_score`, `score_change` — the first
is an identifier, the last is a label leak, and the rest violate the at-prediction-time contract.

Full integration guidance (confidence thresholds, prompt-context format) lives in [`prd.md`](prd.md).

## ⚗️ Feature Set

| Feature | Type | Handling |
|---|---|---|
| `subject` | categorical | one-hot (6 subjects) |
| `session_number` | numeric | passthrough |
| `study_duration_min` | numeric | invalids → NaN → median-imputed |
| `questions_attempted` | numeric | imputed; ≤0 treated as invalid |
| `questions_completed` | numeric | imputed; > attempted treated as invalid |
| `completion_rate` | numeric | recomputed = completed / attempted |
| `previous_test_score` | numeric | median-imputed (missing on first sessions) |
| `current_test_score` | numeric | median-imputed (9 missing) |

## ⚠️ Limitations

- **Synthetic, rule-generated labels.** `progress_status` was produced by performance rules;
  accuracy reflects consistency with those rules, not real-world educational effectiveness.
- **Small student population** (60 students). Grouped-split scores may not generalize to new
  cohorts.
- **Rule-derived risk.** Only `completion_rate` was factually recomputed during cleaning;
  other score fields are reported as recorded.

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas · NumPy |
| Modeling | scikit-learn (`Pipeline`, `ColumnTransformer`) |
| Evaluation & viz | matplotlib · seaborn |
| Export | joblib · Jupyter nbconvert |

## 📜 License

MIT — free to use, modify, and distribute.
