# Model-Family Recommendation — Verified

> Generated from the verified EDA findings. Every claim ties to specific,
> checked characteristics of this dataset.

---

## 1. Problem Type

**Binary classification** — predict whether a customer will churn
(`churned` = 1) within the label window (appears to be 90 days of
inactivity or zero lifetime activity).

## 2. Key Data Characteristics (Verified)

| Characteristic | Number | Modeling Implication |
|---------------|--------|---------------------|
| **Class imbalance** | 13.5% churn (6.4:1 ratio) | Need class-weight support; avoid accuracy |
| **Non-stationarity** | Churn rate triples: 9.1% (2024) → 27.8% (2026) | Time-based validation mandatory |
| **Non-linear interactions** | Channel×spend: low-spend churns at 15-22%, high-spend at 4-8% | Tree models capture natively; linear models need manual engineering |
| **Feature types** | Mixed: categorical (channel, region), skewed numeric (spend, counts), temporal | Model must handle heterogeneous types |
| **Leakage-tainted variables** | days_since_last_txn (corr=0.74, perfect separation) | Exclude from feature set entirely |
| **Multi-source mismatch** | Parquet incomplete for July 2026 (19.28% shortfall) | Use DB as feature source |

## 3. Recommended Model Family

### Primary Recommendation: Gradient Boosted Trees (XGBoost / LightGBM)

**Why GBT is the right fit for *this* dataset:**

1. **Non-linear channel×spend interaction ($2.5 in EDA_FINDINGS.md).**
   Low-spend customers across all 4 channels churn at 15-22%; high-spend
   customers at 4-8%. A tree-based model captures this automatically;
   logistic regression requires an explicit interaction term.

2. **Class imbalance is native.** XGBoost's `scale_pos_weight` = 6.4 (or
   `sum(negative) / sum(positive)`) handles the 13.5% churn rate without
   manual resampling.

3. **Robust to right-skewed monetary variables.** Total spend (mean=$420,
   median=$364, max=$1,753) is strongly right-skewed with legitimate
   outliers. Tree splits are unaffected by scale — no log-transform or
   winsorization needed.

4. **Mixed data types.** The feature set includes 4-level categorical
   (channel, region), 6-level categorical (category engagement), and
   continuous variables (spend, count, tenure). GBT handles these natively.

5. **Feature importance for interpretability.** For a retention team,
   understanding *why* a customer is predicted to churn is as important
   as the prediction itself.

**Configuration considerations:**
- `scale_pos_weight` ≈ 6.4 or `sum(negative) / sum(positive)`
- Early stopping on validation set (N=1,500 is moderate — risk of overfitting)
- Time-based validation split (pre-2026 train / 2026+ test), **not random**

### Secondary Recommendation: Random Forest

- More robust to outliers — though GBT already handles this well.
- Easier to debug and explain to non-technical stakeholders.
- **Trade-off:** Bagging averages over many trees, which may dilute the
  2026 cohort signal if the majority of training data is from 2024-2025.
  GBT with early stopping adapts better to the non-stationary regime.

### Not Recommended

| Family | Reason |
|--------|--------|
| **Logistic Regression** | The channel×spend interaction and the non-linear relationship between recency and churn would require extensive manual feature engineering. No benefit over GBT at this scale. |
| **Deep Neural Networks** | N=1,500 with 10-15 raw features is far too small. Severe overfitting risk with negligible upside over GBT. |
| **Survival Analysis** | Theoretically appropriate for time-to-event, but the assignment scope is binary churn classification with a fixed label window. |

## 4. Feature Engineering

### Usable (full-history, non-circular)
- `total_spend`, `txn_count`, `avg_txn_value`, `max_txn_value`
- `distinct_categories`, `distinct_channels`
- `return_rate` (returns / total txns)
- `customer_tenure_days` (last_txn_date - signup_date)
- `signup_year` / `signup_quarter` (critical — captures time drift)
- `acquisition_channel` (ordinal or one-hot encoded)
- `region` (ordinal or one-hot encoded)
- Per-category engagement: transactions or spend per category
- `zero_activity_flag` (binary — distinguishes the 5.4% zero-activity subpopulation)

### Excluded (leakage-tainted — circular with label definition)
- **`days_since_last_txn`** — perfectly separates churned (≥90) from non-churned (≤89);
  correlation with churn = 0.74. Definitionally circular with the 90-day inactivity label.
- **`trailing_90d_spend`** — same 90-day window as the label.
- **`trailing_90d_txn_count`** — same.
- **`inactive_90d_flag`** — directly encodes the label.

### Caveated
- **`avg_days_between_txns`** — safe if computed over the customer's full history;
  unsafe if restricted to the trailing 90 days.

## 5. Validation Strategy

### 5.1 Time-Based Split (Mandatory)

- **Train:** signups before 2026-01-01 (2024-2025 cohorts: 1,183 customers)
- **Validation / Test:** signups from 2026-01-01 onward (317 customers)
- Rationale: random split would leak the time drift (9% → 28% churn) into both
  training and testing, overestimating real-world performance.

### 5.2 Evaluation Metrics

- **Primary:** F1-score (balances precision/recall on the 13.5% minority class)
- **Secondary:** ROC-AUC, Precision-Recall AUC
- **Report only:** Accuracy (86.5% baseline is misleading)

### 5.3 Baseline Benchmark

| Model | Expected F1 | Expected AUC | Notes |
|-------|-------------|--------------|-------|
| Dummy (always majority) | 0.0 | 0.50 | Baseline floor |
| Logistic Regression (non-leaked vars) | ~0.15-0.25 | ~0.65-0.70 | Baseline lower bound |
| **GBT (XGBoost/LightGBM)** | **>0.35** | **>0.75** | **Target** |

## 6. Risk Factors

| Risk | Mitigation |
|------|-----------|
| Leakage-circularity (90-day window) | Exclude all trailing-90-day features; document in model card |
| Non-stationarity (churn rate ↑ over time) | Time-based validation; include signup cohort as feature |
| Data source inconsistency (parquet vs DB) | Source all features from raw transactions DB |
| Small dataset (1,500 customers) | Cross-validation; early stopping; no DNN |
| Zero-activity subpopulation (5.4%, 100% churned) | Model separately or add zero_activity_flag — these customers have no transaction history to learn from |