# Data Quality Issues (Stage 0 Draft)

> Generated 2026-07-31 from raw data sources. **Draft — not authoritative.**
> Learner must verify each issue against the actual data per Helper Guide.docx.

## Issue 1: Multi-Source Disagreement — Parquet Incomplete for July 2026

**Severity: HIGH** — affects all models using the parquet file as a feature source.

### Description
The `transactions_monthly_agg.parquet` file disagrees with the raw `transactions.db` source for the month of July 2026:

- **102 rows** present in the DB but missing entirely from the parquet (10 customers who only transacted in July 2026).
- **212 count mismatches** across July 2026 (DB has 1,231 transactions; parquet aggregates to 994 — a **19% shortfall**).
- July 2026 amount total: DB = $62,084.62 vs Parquet = $50,116.67 ($11,967.95 missing, a **19% discrepancy**).

### Root Cause
The parquet file appears to have been generated before the end of July 2026 (likely at a cutoff ~July 25), so it excludes transactions occurring in the final days of the month. The DB has the complete record.

### Impact
- Any analysis or model relying solely on the parquet will undercount July 2026 activity.
- Newer customers (who signed up in 2026) are disproportionately affected.
- Since 2026 signups have a much higher churn rate (27.8% vs 9-10%), this could bias feature importance *against* recent engagement metrics.

### Recommended Action
- **Use the raw transactions DB** as the source of truth for all feature engineering.
- If the parquet must be used, note it as incomplete and document the bias.
- For modeling, ensure July 2026 transactions are fully represented before computing any rolling or trailing features.

---

## Issue 2: Potential Leakage — Definitional Circularity with 90-Day Window

**Severity: CRITICAL** — if unaddressed, the model will learn a circular rule and fail in production.

### Description
There is near-perfect separation between churned (days_since_last_txn ≥ 90) and non-churned (days_since_last_txn ≤ 89) customers. This strongly suggests the churn label is **defined** as "no transactions in 90 days" (or a similar window). If so, any feature that directly or indirectly encodes the same 90-day window (days_since_last_txn, trailing_90d_spend, trailing_90d_txn_count, `inactive_90d` flag) is definitionally correlated with the target.

### Evidence
- All 202 churned customers have days_since_last_txn ≥ 90.
- All 1,298 non-churned customers have days_since_last_txn ≤ 89.
- Pearson correlation with churn: 0.74 — the highest of any single feature.

### Impact
- A model including days-since-last-txn would achieve near-perfect training performance but would simply encode the label definition.
- In production, this would fail as soon as the prediction window or definition changes.
- The model would provide no real insight into *why* customers churn — it just detects the label.

### Root Cause
Label definition convention. The churn flag is set for customers who have not transacted in the observation window (likely the last 90 days prior to the data snapshot date).

### Recommended Action
1. Confirm the label definition with the data owner.
2. **Exclude** all features that encode the same 90-day inactivity window from the feature set.
3. If the label window is shorter or longer than 90 days, clip the recency feature accordingly (but still flag the risk).
4. Document this clearly in the model card and feature schema.

---

## Issue 3: Class Imbalance

**Severity: MEDIUM** — manageable with appropriate techniques.

### Description
- Churned: 202 customers (13.5%)
- Non-churned: 1,298 customers (86.5%)
- Ratio: ~6.4:1

### Impact
- Accuracy is misleading (predicting all non-churned yields 86.5% accuracy but 0% recall on churn).
- Standard logistic regression and unweighted trees will be biased toward the majority class.
- Precision-recall curves and F1-score are needed instead of accuracy.

### Recommended Action
- Use `scale_pos_weight` or `class_weight='balanced'` in tree models.
- Consider SMOTE or random undersampling if using scikit-learn models without built-in weighting.
- Always report precision, recall, F1, and ROC-AUC — never accuracy alone.

---

## Issue 4: Time Non-Stationarity (Drift)

**Severity: HIGH** — the churn generation process changes over time.

### Description
Churn rate rises sharply by signup cohort:
- 2024: 9.1%
- 2025: 10.1%
- 2026 H1: 27.8%

Monthly breakdown shows a clear inflection point starting March 2026 (30.8% churn) peaking in June 2026 (57.1%).

### Impact
- A model trained on 2024-2025 data alone will under-predict churn for 2026 cohorts.
- Random train/test splits will overestimate performance (by leaking the time trend into both sets).
- The relationships between features and churn may also drift (e.g., "organic channel" predicted churn at 15% in 2024 but 35% in 2026).

### Recommended Action
- Mandatory time-based train/test split (cutoff ~2026-01-01).
- Include signup cohort as a feature.
- Monitor feature distributions and churn rates over time in production.
- Consider a time-aware model (e.g., survival analysis, or models with built-in time decay).

---

## Issue 5: No Missing Values — But Validate Completeness

**Severity: LOW** — all tables have complete data, so this is a note, not a problem.

### Description
- customers.csv: 0 missing values across 5 columns.
- transactions.db: 0 missing values across 6 columns.
- transactions_monthly_agg.parquet: 0 missing values across 5 columns.

### Note
While the data is complete, this is unusual for real-world customer data (where signup dates, regions, or channels are often missing). This may be synthetic data. In production, missingness should be re-checked and null-handling logic added even if currently unnecessary.

---

## Issue 6: Skewed Transaction Amount Distributions

**Severity: LOW** — not a data quality error, but a modeling consideration.

### Description
- Total spend has high outliers (max $1,753 vs median $364; IQR outliers ~2.1%).
- Average transaction value: 4.9% IQR outliers at the high end.
- Transaction count: only 0.3% outliers.

### Impact
- Linear models and distance-based methods will be sensitive to these high spends.
- Tree-based models are naturally robust, but preprocessing (log-transform or winsorization) may help if linear models are needed.

### Recommended Action
- Use robust scaling or log-transform for continuous features if using linear models.
- Winsorize at the 99th percentile if outliers are considered noise rather than signal.
- Tree models (recommended) do not require this preprocessing.