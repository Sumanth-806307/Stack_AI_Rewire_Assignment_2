# Churn EDA & Model-Family Recommendation

Exploratory data analysis of an e-commerce customer churn dataset across three
heterogeneous data sources — a CSV, a SQLite database, and a Parquet extract —
producing a verified findings report, a defensible model-family recommendation,
and a data-issues log covering leakage risk, class imbalance, time drift, and
multi-source disagreement.

This repository implements the analysis confirmed in `eda_review.md` — it does
not originate one. The review process (Stage 0 → learner review → build) is
described in `Helper Guide.docx` in the assignment package.

---

## Repo Structure

```
.
├── data/
│   ├── customers.csv                  # 1,500 customers with churn labels
│   ├── transactions.db                # 11,742 row-level transactions (SQLite)
│   └── transactions_monthly_agg.parquet  # 8,086 pre-aggregated records
│
├── eda/                               # Analysis modules (pure functions)
│   ├── load.py                        # One loader per data source
│   ├── profile.py                     # Schema, distributions, outliers, zero-activity
│   ├── hypotheses.py                  # Hypothesis tests (Mann-Whitney U, chi-square)
│   ├── time_checks.py                 # Monthly trend + multi-source agreement
│   └── leakage_checks.py              # Label-circularity + as-of-date checks
│
├── scripts/                           # Runnable runners over eda/ modules
│   ├── run_profile.py                 # Full data profile (schema, distributions, churn rates)
│   ├── run_hypothesis_tests.py        # All 8 hypothesis tests with results table
│   ├── run_time_and_leakage_checks.py # Time drift, multi-source, and leakage checks
│   └── verify_findings.py             # Reproduces every EDA_FINDINGS.md claim from raw data
│
├── notebooks/
│   └── eda_churn.ipynb                # Narrative notebook, runs top-to-bottom
│
├── tests/                             # 5 test files, 55 tests total
│   ├── test_load.py                   # 9 tests — schema, dtypes, row counts
│   ├── test_profile.py                # 7 tests — skew detection, IQR outliers, zero-activity
│   ├── test_hypotheses.py             # 5 tests — known-answer fixtures
│   ├── test_time_and_leakage_checks.py # 10 tests — trend, multi-source, leakage flags
│   └── test_findings_reproduce.py     # 23 tests — every claim in EDA_FINDINGS.md reproducible
│
├── outputs/artifacts/
│   ├── EDA_FINDINGS_DRAFT.md          # Stage 0 draft (for reference)
│   ├── MODEL_RECOMMENDATION_DRAFT.md  # Stage 0 draft (for reference)
│   ├── DATA_ISSUES_DRAFT.md           # Stage 0 draft (for reference)
│   └── test_results.txt               # Saved pytest output from verified run
│
├── EDA_FINDINGS.md                    # Verified findings (all claims reproducible)
├── MODEL_RECOMMENDATION.md            # Defensible model-family recommendation
├── DATA_ISSUES.md                     # Verified data quality issues log
├── ARTIFACT_NOTES.md                  # Self-critique log
├── review_log.md                      # Step 3 self-review log
├── eda_review.md                      # Corrected review of Stage 0 drafts
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Setup

### Prerequisites

- Python 3.10+
- SQLite (built into Python's stdlib — nothing extra to install)

### 1. Clone and enter the repo

```bash
git clone <your-repo-url> churn-eda
cd churn-eda
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies installed:

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, aggregation |
| `numpy` | Numerical operations |
| `scipy` | Hypothesis tests (Mann-Whitney U, chi-square) |
| `pyarrow` | Parquet file I/O |
| `matplotlib` | Notebook visualizations |
| `jupyter` | Notebook runner |
| `pytest` | Test suite runner |

SQLite access uses Python's built-in `sqlite3` module — nothing extra to install.

### 3. Activate the environment

```bash
source .venv/bin/activate    # every new terminal session
```

---

## Running the Analysis

Run these scripts from the repo root directory (`churn-eda/`). All scripts use
`python -m scripts.<name>` form so the `eda/` package resolves as an import.

> **Note:** Use `python -m scripts.run_profile`, **not** `python scripts/run_profile.py` —
> the `-m` form runs it as part of the `scripts` package so `eda/` resolves as an import;
> running the `.py` file directly raises `ModuleNotFoundError`.

### 1. Profile the data sources

```bash
python -m scripts.run_profile
```

Outputs: schema and dtypes per source, distribution stats, outlier counts,
churn rates by channel and region, zero-activity customer count.

### 2. Run hypothesis tests

```bash
python -m scripts.run_hypothesis_tests
```

Outputs: results table for all 8 candidate churn-driver tests — test name,
statistic, p-value, significance at α=0.05, effect size, and direction.

### 3. Run time-trend and leakage checks

```bash
python -m scripts.run_time_and_leakage_checks
```

Outputs: monthly transaction volume trend, churn rate by signup cohort,
multi-source DB-vs-parquet comparison, days-since-last-txn separation check,
and as-of-date discipline check.

### 4. Run the notebook

```bash
jupyter notebook notebooks/eda_churn.ipynb
```

Or in a headless environment:

```bash
jupyter nbconvert --to notebook --execute notebooks/eda_churn.ipynb --output eda_churn_executed.ipynb
```

The notebook runs all 12 sections top-to-bottom with no manual intervention.

---

## Verifying Findings

### Reproduce every claim in EDA_FINDINGS.md

```bash
python -m scripts.verify_findings
```

This recomputes every numeric claim in `EDA_FINDINGS.md` directly from the raw
data sources. Exit code 0 means all claims verified. Exit code 1 means at least
one claim doesn't reproduce — the specific failure is printed.

### Mutation check (optional, recommended)

To confirm the verification is sensitive enough, deliberately corrupt one number
in `EDA_FINDINGS.md`, re-run the verification, and confirm it fails:

```bash
# Make a scratch copy
cp EDA_FINDINGS.md EDA_FINDINGS_BACKUP.md
# Change one number (e.g. churn rate)
sed 's/13.5%/12.0%/g' EDA_FINDINGS_BACKUP.md > EDA_FINDINGS.md
python -m scripts.verify_findings   # Should FAIL
# Restore the real file
cp EDA_FINDINGS_BACKUP.md EDA_FINDINGS.md
python -m scripts.verify_findings   # Should PASS again
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Run all 55 tests across 5 test files:

| Test file | Coverage | Tests |
|-----------|----------|-------|
| `test_load.py` | Loaders return correct columns, dtypes, and row counts | 9 |
| `test_profile.py` | Skew detection, IQR outlier counting, zero-activity, churn rates | 7 |
| `test_hypotheses.py` | Each hypothesis function against known-answer fixtures | 5 |
| `test_time_and_leakage_checks.py` | Trend, multi-source agreement, leakage flagging | 10 |
| `test_findings_reproduce.py` | Every claim in EDA_FINDINGS.md reproduces from raw data | 23 |
| **Total** | | **55** |

### Expected output

```
============================== 55 passed in 3.36s ==============================
```

---

## Key Findings Summary

| Finding | Detail |
|---------|--------|
| **Churn rate** | 13.5% (202/1,500) — moderate imbalance, 6.4:1 ratio |
| **Zero-activity population** | 81 customers (5.4%) with zero transactions — **100% churned**, 40.1% of all churn |
| **Time non-stationarity** | Churn triples from 9.1% (2024 cohorts) to 27.8% (2026 cohorts), p<0.0001 |
| **Best predictors** | Lower total spend (p=0.0001), fewer txns (p=0.0002), fewer categories (p=0.0006) |
| **Not predictive** | Avg transaction value (p=0.18), return count (p=0.15), channel (p=0.12), region (p=0.29) |
| **🔴 CRITICAL — Leakage** | `days_since_last_txn` perfectly separates churned (≥90 days) from non-churned (≤89) — correlation 0.74. Definitionally circular with the churn label. **Exclude from all models.** |
| **🔴 Multi-source mismatch** | Parquet file is incomplete for July 2026 (19.3% shortfall vs DB). **Use DB as source of truth.** |

### Recommended Model Family

**Gradient Boosted Trees (XGBoost / LightGBM)** — handles the non-linear
channel×spend interaction, class imbalance natively, robust to skewed monetary
variables, and supports mixed data types without manual feature engineering.

**Excluded (leakage-tainted):** `days_since_last_txn`, `trailing_90d_spend`,
`trailing_90d_txn_count`, `inactive_90d_flag`.

---

## Output Deliverables

All deliverable documents are at the repo root and in `outputs/artifacts/`:

| File | Purpose |
|------|---------|
| `EDA_FINDINGS.md` | Verified findings — distributions, hypothesis results, trends, leakage |
| `MODEL_RECOMMENDATION.md` | Defensible model-family recommendation with reasoning |
| `DATA_ISSUES.md` | Verified data quality issues with severity, impact, and action items |
| `ARTIFACT_NOTES.md` | Self-critique — what was reconsidered during the build |
| `review_log.md` | Step 3 self-review — how each finding was resolved |
| `outputs/artifacts/test_results.txt` | Saved `pytest -v` run output |
| `eda_review.md` | Corrected review of Stage 0 drafts (reference) |

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
|---------|--------------------|
| `ModuleNotFoundError: No module named 'eda'` | Script run directly as `python scripts/run_profile.py` instead of `python -m scripts.run_profile` — use the `-m` form |
| `ModuleNotFoundError: No module named 'pandas'` | Virtual environment not activated — run `source .venv/bin/activate` |
| A hypothesis test shows near-zero p-value with huge separation | The driver may be leakage-tainted (trailing-90-day window). Check against the label definition |
| `verify_findings.py` fails on a claim you know is correct | The expected value in `scripts/verify_findings.py` may be stale — update it to match the recomputed value, or update `EDA_FINDINGS.md` if the claim was incorrect |
| Notebook cell errors on re-run | A cell depends on state left from a previous run — restart the kernel and run all cells fresh |