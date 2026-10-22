# EDA Findings (Stage 0 Draft)

> Generated 2026-07-31 from raw data sources. **Draft — not authoritative.**
> Learner must spot-check each claim per Helper Guide.docx before treating as verified.

## 1. Data Overview

### 1.1 Sources

| Source | Format | Rows | Columns | Key Fields |
|--------|--------|------|---------|------------|
| `customers.csv` | CSV | 1,500 | 5 | customer_id, signup_date, acquisition_channel, region, churned |
| `transactions.db` | SQLite (transactions table) | 11,742 | 6 | customer_id, txn_timestamp, category, channel, amount, is_returned |
| `transactions_monthly_agg.parquet` | Parquet | 8,086 | 5 | customer_id, txn_month, txn_count, total_amount, return_count |

### 1.2 Customer Demographics

- **Signup date range:** 2024-02-14 to 2026-07-01
- **Acquisition channels:** organic (609), paid_search (431), referral (295), social (165)
- **Regions:** east (385), north (381), west (380), south (354)
- **No missing values** in any customer attribute.

## 2. Churn Analysis

### 2.1 Class Imbalance

- **Overall churn rate: 13.5%** (202 churned vs 1,298 not churned)
- **Ratio (non-churned : churned) ≈ 6.4 : 1**
- This is moderate imbalance. Accuracy is misleading; F1, precision-recall, and ROC-AUC are the appropriate evaluation metrics.

### 2.2 Churn by Acquisition Channel

| Channel | Customers | Churned | Churn Rate |
|---------|-----------|---------|------------|
| organic | 609 | 93 | **15.3%** |
| social | 165 | 24 | 14.5% |
| referral | 295 | 41 | 13.9% |
| paid_search | 431 | 44 | **10.2%** |

- Organic channel has the highest churn rate; paid_search the lowest.
- **Chi-square test: χ²=5.84, p=0.120** — not statistically significant at α=0.05.
- Channel alone is a weak predictor; needs to be used as interaction with other features.

### 2.3 Churn by Region

| Region | Customers | Churned | Churn Rate |
|--------|-----------|---------|------------|
| east | 385 | 63 | **16.4%** |
| south | 354 | 45 | 12.7% |
| west | 380 | 47 | 12.4% |
| north | 381 | 47 | 12.3% |

- East region has a modestly higher churn rate (~16% vs ~12%).
- **Chi-square test: χ²=3.76, p=0.289** — not statistically significant.

### 2.4 Churn by Signup Year (Time Drift — Critical Finding)

| Signup Year | Customers | Churned | Churn Rate |
|-------------|-----------|---------|------------|
| 2024 | 570 | 52 | **9.1%** |
| 2025 | 613 | 62 | **10.1%** |
| 2026 | 317 | 88 | **27.8%** |

- **Chi-square test: χ²=70.72, p < 0.0001** — strongly significant.
- Churn rate triples from 2024 (~9%) to 2026 H1 (~28%).
- Monthly breakdown confirms a clear upward drift starting around March 2026, with June 2026 at 57% and July 2026 at 67%.
- **This is non-stationarity** — the churn generation process is time-dependent. Models must account for this (time-based features, cohort effects, or time-aware validation).

## 3. Transaction Profiling

### 3.1 Transaction Patterns by Churn Status

| Feature | Not Churned (mean) | Churned (mean) | Mann-Whitney p-value | Significant? |
|---------|-------------------|----------------|---------------------|--------------|
| Total spend | $420.41 | $315.06 | **0.0001** | ✅ |
| Transaction count | 8.43 | 6.58 | **0.0002** | ✅ |
| Distinct categories | 4.17 | 3.64 | **0.0006** | ✅ |
| Return count | 0.42 | 0.33 | 0.147 | ❌ |
| Avg transaction value | $50.49 | $50.36 | — | ❌ (near-identical) |

- Churned customers consistently transact **less often, spend less total, and engage with fewer categories**.
- Average transaction value is essentially the same — churned customers buy *less frequently*, not *cheaper*.
- Return rate is not significantly different.

### 3.2 Recency (Days Since Last Transaction)

| Status | Mean | Median | Min | Max | Std |
|--------|------|--------|-----|-----|-----|
| Not churned | 27.6 | 22.0 | -1 | 89 | 23.2 |
| Churned | **131.5** | **112.0** | **90** | 364 | 48.7 |

- **Stark separation:** All churned customers have days_since_last_txn ≥ 90. All non-churned have ≤ 89.
- This is a near-perfect separation threshold. **LEAKAGE WARNING:** If `days_since_last_txn` or a closely related feature (e.g., "90-day inactivity flag", "trailing 90-day spend") is used, the model will trivially predict churn as "no transactions in 90 days" — which may be definitionally circular with how churn is labeled.

### 3.3 Outlier Detection (IQR Method)

| Feature | Outliers | %
|---------|----------|---|
| Total spend (high) | 30 (2.1%) | > $1,139 |
| Transaction count (high) | 4 (0.3%) | > 24 |
| Avg transaction value (high) | 69 (4.9%) | > $92.90 |
| Max transaction value (high) | 64 (4.5%) | > $304 |

- Outliers are mostly high-value — legitimate high-spending customers, not data errors.
- Robust/scaling methods are recommended over raw z-score for these variables.

### 3.4 Category & Channel Distribution

- **Six categories** (apparel, beauty, electronics, grocery, home, sports) — nearly uniform distribution (~1,900-2,000 transactions each).
- **Three channels** — web (5,278), mobile_app (4,146), store (2,318).
- Churned customers show lower engagement across **all** categories, with grocery showing the biggest relative drop (churned mean=1.57 vs non-churned mean=2.10).

### 3.5 Transaction Amount Distribution

- Amounts range from $3.00 to $1,753.23 (max outlier).
- Right-skewed distribution (mean ≈ $50, median ≈ $35-50 range).
- Non-parametric tests (Mann-Whitney U) are appropriate for hypothesis testing.

## 4. Multi-Source Disagreement (DB vs Parquet)

- **102 rows** present in the DB but absent from the parquet (all July 2026 transactions for 10 customers).
- **212 count mismatches** — mostly in July 2026 (DB has 1,231 July transactions; parquet aggregates to 994).
- Total July 2026 amount discrepancy: **DB $62,084.62 vs Parquet $50,116.67**.
- All 10 fully-missing customers have zero churned rate (they signed up in Jan-Jun 2026 and only transacted in July 2026).
- **The parquet file is incomplete for July 2026** — likely truncated at a cutoff date.
- **Impact:** Any analysis using the parquet alone will undercount recent transaction activity, particularly for newer customers. For modeling, this means features derived from the parquet will underestimate July 2026 engagement.

## 5. Leakage Risk Assessment

### 5.1 Critical Leakage: Days-Since-Last-Transaction / 90-Day Inactivity

- **Pearson correlation with churn: 0.74** — the strongest individual predictor by far.
- All churned customers have days_since_last_txn ≥ 90; all non-churned have ≤ 89.
- This is very likely definitional: if churn = "no transactions in the last 90 days", then any feature encoding the same 90-day window is circular.
- **Recommendation:** Exclude or heavily caveat `days_since_last_txn`, `trailing_90d_spend`, `trailing_90d_txn_count`, or any binary "inactive_90d" flag from feature sets used for prediction.

### 5.2 Low Risk Features

- Total spend, txn_count, distinct_categories — these are behavioral summaries over the full history. Not definitionally circular, but caution is needed if they also cover the 90-day window.

## 6. Summary of Key Findings

| Finding | Severity | Action |
|---------|----------|--------|
| Churn rate ~13.5% (moderate imbalance) | Medium | Use F1/ROC-AUC, not accuracy |
| Signup year / time drift (p < 0.0001) | High | Time-aware modeling; cohort features |
| Channel not significant alone (p=0.12) | Low | Use as interaction feature only |
| Region not significant alone (p=0.29) | Low | Optional inclusion |
| Lower spend + engagement → higher churn | Medium | Strong predictive signals |
| Days-since-last-txn near-perfect separation | **Critical** | **Exclude leakage-tainted features** |
| Parquet incomplete for July 2026 | Medium | Use DB as source of truth; note in docs |