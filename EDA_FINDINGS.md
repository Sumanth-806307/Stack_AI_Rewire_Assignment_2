# EDA Findings — Verified

> Generated from `scripts/run_profile.py`, `scripts/run_hypothesis_tests.py`,
> `scripts/run_time_and_leakage_checks.py`. Every numeric claim below is
> reproducible by `scripts/verify_findings.py`.

---

## 1. Data Overview

| Source | Format | Rows | Columns | Missing Values |
|--------|--------|------|---------|----------------|
| `customers.csv` | CSV | 1,500 | 5 | 0 in all columns |
| `transactions.db` | SQLite (`transactions`) | 11,742 | 6 | 0 in all columns |
| `transactions_monthly_agg.parquet` | Parquet | 8,086 | 5 | 0 in all columns |

### 1.1 Customer Demographics

- **Signup date range:** 2024-02-14 to 2026-07-01
- **Acquisition channels:** organic (609), paid_search (431), referral (295), social (165)
- **Regions:** east (385), north (381), west (380), south (354)

### 1.2 Zero-Activity Population

**81 customers (5.4% of total) have zero rows in `transactions.db`.** These are not
missing data — they are a distinct subpopulation who signed up but never transacted.
**All 81 (100%) are labeled churned**, representing **40.1% of all churned customers**.
This population must be analyzed separately from active-churned customers.

---

## 2. Churn Analysis

### 2.1 Class Imbalance

- **Overall churn rate: 13.5%** (202 churned / 1,500 total)
- Non-churned: 1,298 (86.5%)
- Ratio (non-churned : churned) ≈ **6.4 : 1**
- A dummy classifier predicting "not churned" always achieves 86.5% accuracy with
  **zero recall** on churn. F1-score, precision-recall AUC, and ROC-AUC are the
  appropriate evaluation metrics — accuracy alone is misleading.

### 2.2 Churn by Acquisition Channel

| Channel | Customers | Churned | Churn Rate |
|---------|-----------|---------|------------|
| organic | 609 | 93 | 15.3% |
| social | 165 | 24 | 14.5% |
| referral | 295 | 41 | 13.9% |
| paid_search | 431 | 44 | 10.2% |

- **Chi-square test: χ²=5.84, p=0.120** — not statistically significant at α=0.05.
- Channel alone is a weak predictor, but channel × spend interaction is strong
  (see §2.5).

### 2.3 Churn by Region

| Region | Customers | Churned | Churn Rate |
|--------|-----------|---------|------------|
| east | 385 | 63 | 16.4% |
| south | 354 | 45 | 12.7% |
| west | 380 | 47 | 12.4% |
| north | 381 | 47 | 12.3% |

- **Chi-square test: χ²=3.76, p=0.289** — not statistically significant.

### 2.4 Churn by Signup Year (Time Non-Stationarity)

| Signup Year | Customers | Churned | Churn Rate |
|-------------|-----------|---------|------------|
| 2024 | 570 | 52 | **9.1%** |
| 2025 | 613 | 62 | **10.1%** |
| 2026 | 317 | 88 | **27.8%** |

- **Chi-square test: χ²=70.72, p<0.0001** — strongly significant (Cramér's V = 0.22).
- Monthly resolution confirms a sharp inflection starting March 2026 (30.8% churn),
  peaking in June 2026 (57.1%).
- **This is non-stationarity**: the churn generation process changes over time.
  Time-based train/test split is mandatory.

### 2.5 Channel × Spend Interaction

Low-spending customers (below median total spend of ~$364) churn at **15-22% across
all channels**, while high-spending customers churn at only **4-8%**:

| Channel | Low-Spend Churn Rate | High-Spend Churn Rate |
|---------|---------------------|----------------------|
| organic | 21.6% | 8.3% |
| paid_search | 15.7% | 4.3% |
| referral | 21.7% | 4.5% |
| social | 20.5% | 7.8% |

This interaction is a strong candidate for tree-based modeling, which natively
captures it.

---

## 3. Transaction Profiling

### 3.1 Aggregates by Churn Status

| Feature | Not Churned (mean) | Churned (mean) | Mann-Whitney p | Significant? | Effect (Cohen's d) |
|---------|-------------------|----------------|----------------|--------------|-------------------|
| Total spend | $420.41 | $315.06 | **0.0001** | ✅ YES | 0.36 |
| Transaction count | 8.43 | 6.58 | **0.0002** | ✅ YES | 0.36 |
| Distinct categories | 4.17 | 3.64 | **0.0006** | ✅ YES | 0.34 |
| Avg transaction value | $50.49 | $50.36 | 0.184 | no | 0.01 |
| Return count | 0.42 | 0.33 | 0.147 | no | 0.14 |
| Distinct channels | 2.55 | 2.27 | — | — | — |

**Findings:**
- Churned customers consistently transact **less often, spend less total, and engage
  with fewer categories and channels**.
- Average transaction value is **not significantly different** (p=0.184). Churned
  customers buy *less frequently*, not *cheaper*.
- Return count is not significantly different (p=0.147).

### 3.2 Recency (Days Since Last Transaction)

| Status | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Not churned | 27.6 days | 22.0 | -1 | 89 |
| Churned (active) | 131.5 days | 112.0 | 90 | 364 |

- **Perfect separation at threshold 90 days**: all 121 active churned customers have
  days_since_last_txn ≥ 90; all 1,298 non-churned have ≤ 89.
- **Correlation with churn: 0.74** — the strongest single predictor and definitionally
  circular with the churn label (see §5 — Leakage).

### 3.3 Transaction Amount Distribution

| Stat | Value |
|------|-------|
| Mean | $49.72 |
| Median | $33.82 |
| Std | $55.24 |
| Min | $3.00 |
| Max | $1,684.21 |
| Skew | **Right** (mean > median by ~47%) |
| IQR outliers (high side) | 783 (6.67%) |

- Strong right skew with legitimate high-value outliers (not data errors).
- Non-parametric tests (Mann-Whitney U) are appropriate for hypothesis testing.

### 3.4 Category & Channel Distributions

**Categories** (nearly uniform, ~1,900-2,000 transactions each):
apparel, beauty, electronics, grocery, home, sports

**Channels**: web (5,278), mobile_app (4,146), store (2,318)

**Category engagement drops universally for churned customers**, with grocery
showing the largest relative decline (churned mean=1.57 vs non-churned mean=2.10
transactions per customer).

### 3.5 Transaction Volume Trend

| Metric | First Month | Last Month | Growth |
|--------|-------------|------------|--------|
| Monthly txns | 1 (Feb 2024) | 1,231 (Jul 2026) | ~1,200× |
| Monthly customers | 1 (Feb 2024) | 787 (Jul 2026) | ~787× |

- Monotonically increasing volume reflects the customer base growing over time.
- Raw volume growth does not indicate a churn signal in itself, but the rapidly
  increasing churn rate among later cohorts (see §2.4) combined with higher volume
  means more absolute churn risk in recent periods.

---

## 4. Multi-Source Agreement (DB vs Parquet)

| Metric | Value |
|--------|-------|
| DB re-aggregated rows | 8,188 |
| Parquet rows | 8,086 |
| Rows only in DB (left_only) | **102** (10 customers, all July 2026-only) |
| Rows only in parquet (right_only) | 0 |
| Matched rows with count mismatch | **110** |
| Matched rows with amount mismatch | **110** |
| Matched rows with return mismatch | **5** |
| Mismatched months | **Only July 2026** |
| July 2026 DB total | $62,084.62 |
| July 2026 Parquet total | $50,116.67 |
| July 2026 shortfall | **19.28%** |

**Conclusion:** The parquet file is incomplete for July 2026, missing ~19% of
transaction activity compared to the raw DB. The DB should be used as the source
of truth for all feature engineering.

---

## 5. Leakage Risk Assessment

### 5.1 Critical: Days-Since-Last-Transaction (Circular with Label)

- All churned customers with transaction data have days_since_last_txn ≥ 90.
- All non-churned customers have days_since_last_txn ≤ 89.
- Pearson correlation with churn: **0.741**.
- Additionally, all 81 zero-activity customers are churned — the label captures
  "no activity in the observation window" as a definitional property.

**The churn label appears to be defined as "no transactions in the trailing 90-day
window (or zero transactions ever)."** Any feature that directly or indirectly encodes
the same 90-day window is definitionally circular and must be excluded from modeling.

**Excluded variables:** `days_since_last_txn`, `trailing_90d_spend`,
`trailing_90d_txn_count`, `trailing_90d_return_count`, `inactive_90d_flag`.

**Safe variables** (full-history aggregates, no time-window overlap):
`total_spend`, `txn_count`, `avg_txn_value`, `max_txn_value`,
`distinct_categories`, `distinct_channels`, `return_count`, `return_rate`,
`customer_tenure_days`, `signup_year`/`signup_quarter`,
`acquisition_channel`, `region`.

### 5.2 As-of-Date Discipline

All transactions fall within the expected period (max timestamp: 2026-07-31
23:50:29). No future data leaks detected.

---

## 6. Summary of Key Findings

| Finding | Evidence | Severity | Action |
|---------|----------|----------|--------|
| Churn rate 13.5% (6.4:1 ratio) | 202/1,500 | Medium | Use F1/AUC, not accuracy |
| Zero-activity: 81 customers, 100% churned | 5.4% of total, 40.1% of all churn | High | Model separately or flag as distinct population |
| Time non-stationarity (p<0.0001) | 9.1% → 27.8% churn by cohort | High | Time-based validation; cohort features |
| Channel not significant alone (p=0.12) | χ²=5.84 | Low | Use as interaction only |
| Region not significant alone (p=0.29) | χ²=3.76 | Low | Optional inclusion |
| Lower spend/engagement → higher churn | p<0.001, d~0.36 | Medium | Strong predictive signal |
| Avg txn value NOT significant (p=0.18) | d=0.01 | Low | Drop from feature set |
| Days-since-last-txn: perfect separation | Gap=1, correlation=0.74 | **Critical** | Exclude all leakage-tainted features |
| Multi-source: parquet incomplete Jul 2026 | 19.28% shortfall | Medium | Use DB as source of truth |