# Data Quality Issues — Verified

> Each issue is backed by a number computed directly from the raw data sources.
> Issues are ordered by severity.

---

## Issue 1: [CRITICAL] Label Leakage — Definitional Circularity with 90-Day Window

### Description
There is perfect separation between churned and non-churned customers on the
`days_since_last_txn` feature: **all 121 churned customers with transaction data
have days_since_last_txn ≥ 90** (min=90); **all 1,298 non-churned customers have
days_since_last_txn ≤ 89** (max=89). The separation gap is exactly 1 day.
Pearson correlation with churn = **0.741** — the strongest single predictor by far.

Additionally, **81 customers (5.4% of total) have zero transactions** in the DB,
and **100% of them (81/81) are labeled churned** — representing **40.1% of all
churned customers**.

### Root Cause
The churn label appears to be defined as *"no transactions in the trailing 90-day
window"* (or zero lifetime transactions). Any feature encoding the same 90-day
inactivity window is definitionally circular with the target.

### Impact
A model including `days_since_last_txn`, `trailing_90d_spend`, `trailing_90d_txn_count`,
or any `inactive_90d_flag` would achieve near-perfect training performance but would
simply encode the label definition. In production, this would fail immediately if the
prediction window or label definition changed.

### Recommended Action
1. Confirm the label definition with the data owner.
2. **Exclude** all features derived from the trailing 90-day window from the feature set.
3. Document this exclusion clearly in the model card and feature schema.
4. Use only full-history aggregate features (total spend, lifetime txn count, etc.)
   which do not overlap the label's defining window.

---

## Issue 2: [HIGH] Multi-Source Disagreement — Parquet Incomplete for July 2026

### Description
The `transactions_monthly_agg.parquet` file disagrees with the raw `transactions.db`
source for July 2026:

| Metric | DB (raw) | Parquet (pre-aggregated) | Gap |
|--------|----------|-------------------------|-----|
| Rows at customer×month grain | 8,188 | 8,086 | — |
| Rows only in DB (left_only) | 102 | — | **102 rows** |
| Rows only in parquet (right_only) | 0 | — | 0 |
| Matched rows with count mismatch | — | — | **110 rows** |
| Matched rows with amount mismatch | — | — | **110 rows** |
| July 2026 total amount | **$62,084.62** | **$50,116.67** | **$11,967.95 (19.28%)** |

- All 102 left-only rows and all 110 mismatches are in **July 2026**.
- 10 customers with July-only activity are entirely missing from the parquet.
- All other months (Feb 2024 — Jun 2026) match perfectly.

### Root Cause
The parquet file was generated mid-July 2026, missing the final ~19% of the month's
transactions. The DB has the complete record.

### Impact
- Any analysis or model relying solely on the parquet will undercount recent activity.
- Newer customers (2026 signups) are disproportionately affected.
- Since 2026 signups have a 27.8% churn rate (vs 9-10% for 2024-2025), this biases
  feature importance *against* recent engagement metrics.

### Recommended Action
- **Use the raw transactions DB** as the source of truth for all feature engineering.
- Note the parquet discrepancy in the feature pipeline documentation.

---

## Issue 3: [HIGH] Time Non-Stationarity

### Description
Churn rate rises sharply by signup cohort:

| Signup Year | Customers | Churned | Churn Rate |
|-------------|-----------|---------|------------|
| 2024 | 570 | 52 | **9.1%** |
| 2025 | 613 | 62 | **10.1%** |
| 2026 | 317 | 88 | **27.8%** |

**Chi-square test: χ²=70.72, p<0.0001** (Cramér's V = 0.22 — medium effect).

Monthly resolution confirms a sharp inflection starting March 2026 (30.8% churn),
peaking at 57.1% in June 2026.

### Impact
- A model trained on 2024-2025 data alone will severely under-predict churn for
  2026 cohorts (predicted ~9-10% vs actual ~28%).
- Random train/test splits leak the time trend, overestimating performance.
- The relationship between features and churn may also drift (e.g., organic channel
  at 15% in 2024 vs probably higher in 2026).

### Recommended Action
- Time-based train/test split (cutoff ≈ 2026-01-01).
- Include `signup_year` or `signup_quarter` as a feature.
- Monitor feature distributions and churn rates over time in production.

---

## Issue 4: [MEDIUM] Class Imbalance

### Description
- Churned: **202 customers (13.5%)**
- Non-churned: **1,298 customers (86.5%)**
- Ratio: **~6.4 : 1**

### Impact
- A classifier predicting "not churned" for every customer achieves 86.5% accuracy
  with **zero recall** on churn — completely useless for a retention team.
- Standard unweighted models will be biased toward the majority class.

### Recommended Action
- Use `scale_pos_weight` (XGBoost) or `class_weight='balanced'` (RF/scikit-learn).
- Always report precision, recall, F1, and precision-recall AUC — never accuracy alone.
- Consider the zero-activity subpopulation as a separate modeling concern (see Issue 5).

---

## Issue 5: [MEDIUM] Zero-Activity Subpopulation

### Description
**81 customers (5.4% of total) have zero transactions.** All 81 are churned,
representing **40.1% of all churned customers**. These customers signed up but
never transacted — they are not "missing data."

### Impact
- For these customers, all transaction-derived features are undefined (NaN).
- A model cannot learn engagement-based signals from them — there is nothing to
  learn from.
- They inflate the apparent separation in `days_since_last_txn` and other recency
  features.

### Recommended Action
- Add a binary `zero_activity_flag` or treat these customers as a separate modeling
  segment with a distinct feature set (signup attributes only).
- Do not impute zero for their transaction features — that conflates "never engaged"
  with "low engagement."

---

## Issue 6: [LOW] Skewed Transaction Amounts

### Description
Transaction amounts are strongly right-skewed:
- Mean = $49.72, median = $33.82 (mean > median by ~47%).
- **783 outliers (6.67%)** on the high side by the IQR rule (>$125.93).
- Max = $1,684.21 vs 75th percentile = $61.39.

### Impact
- Linear models and distance-based methods will be sensitive to these high spends.
- Tree-based models are naturally unaffected — no preprocessing needed for the
  recommended family (GBM).

### Recommended Action
- No action needed for tree-based models.
- If linear models are used for baseline comparison, apply log-transform or
  winsorization at the 99th percentile.

---

## Issue 7: [LOW] No Missing Values

### Description
All three data sources have **zero missing values** across all columns.

### Note
This is unusual for real-world customer data (where signup dates, regions, or
channels are often incomplete). This is likely synthetic data. In production,
implement null-handling logic even if it currently appears unnecessary.