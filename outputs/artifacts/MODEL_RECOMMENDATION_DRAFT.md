# Model-Family Recommendation (Stage 0 Draft)

> Generated 2026-07-31 from raw data sources. **Draft — not authoritative.**
> Learner must review against skill.md Domain Context standards and correct before build.

## 1. Problem Type

**Binary classification** — predict whether a customer will churn (churned=1) within the label window (appears to be 90 days based on the definitional leakage analysis).

## 2. Key Data Characteristics Driving the Recommendation

| Characteristic | Detail | Modeling Implication |
|---------------|--------|---------------------|
| **Class imbalance** | 13.5% churn rate (~6.4:1 ratio) | Need class-weight or sampling; avoid accuracy |
| **Non-linear relationships** | Churn likelihood increases sharply with recency; interactions between channel, region, and engagement | Tree-based models likely outperform linear models |
| **Non-stationarity** | Churn rate triples from 2024 (9%) to 2026 (28%) | Time-aware validation; cohort features |
| **Feature types** | Mix of numeric (spend, count), categorical (channel, region, category), and temporal (recency, tenure) | Model must handle heterogeneous types |
| **Leakage concern** | Days-since-last-txn is definitionally circular with churn label | Exclude 90-day window features from training |
| **Multi-source mismatch** | Parquet file incomplete for July 2026 | Use raw DB as feature source; validate on complete data |

## 3. Recommended Model Families (Ranked)

### 3.1 Primary Recommendation: Gradient Boosted Trees (XGBoost / LightGBM)

**Rationale:**
1. **Handles non-linear interactions** automatically — critical given that engagement metrics interact with channel and region.
2. **Native missing-value handling** — no imputation needed for any field (all sources are clean, but robustness is good).
3. **Feature importance** provides interpretability for stakeholders.
4. **Class-weight support** — `scale_pos_weight` in XGBoost handles the ~6.4:1 ratio without manual resampling.
5. **Handles mixed data types** without one-hot explosion (categorical features with 4-6 categories).

**Configuration considerations:**
- Use `scale_pos_weight ≈ 6.4` or `sum(negative)/sum(positive)`.
- Early stopping on validation set to avoid overfitting (N=1,500 is moderate).
- Time-based split (not random) given the non-stationarity.

### 3.2 Secondary Recommendation: Random Forest

**Rationale:**
1. More robust to outliers (transaction amount outliers don't bias tree splits as severely).
2. Easier to debug and explain.
3. Usually underperforms GBT on structured sparse data but can be competitive here given the modest dataset size.

**Trade-off:** May overfit the 2024-2025 regime and fail on 2026; bagging helps but GBT with early stopping is preferred for time-aware tuning.

### 3.3 Not Recommended: Logistic Regression

**Why:** The relationship between recency, engagement, and churn is unlikely to be linear-additive. The interaction between channel and spend (e.g., organic + low spend) would require manual feature engineering. GBT automatically captures these interactions.

### 3.4 Not Recommended: Deep Neural Networks

**Why:** N=1,500 with ~10-15 raw features is too small for DNNs. Risk of overfitting is high, and the gain over GBT is negligible at this scale.

## 4. Feature Engineering Suggestions

### 4.1 Aggregate Features (from transaction DB)
- `total_spend`, `txn_count`, `avg_txn_value`, `max_txn_value` (full history)
- `distinct_categories`, `distinct_channels`
- `return_rate` (returns / total txns)
- `customer_tenure_days` (last_txn - first_txn)

### 4.2 Recency Features (⚠️ leakage-aware)
- `days_since_last_txn` — **EXCLUDE** if churn label is defined by no activity in 90 days
- Alternative: `days_since_last_txn` but re-label churn with a held-out future window (only use if the label definition allows)
- `customer_lifetime_days` (last_txn_date - signup_date) — safer, non-circular

### 4.3 Time/Cohort Features
- `signup_year` (or `signup_quarter`) — critical given the strong time drift
- `txn_recency_ratio` (avg days between txns) — captures engagement intensity without circularity

### 4.4 Categorical Features
- `acquisition_channel` (one-hot or ordinal-encoded)
- `region` (one-hot or ordinal-encoded)
- Top categories used (binary flags or counts per category)

## 5. Validation Strategy

### 5.1 Time-Based Split (Mandatory)
- **Train:** Signups before 2026-01-01 (2024-2025 cohorts)
- **Validation / Test:** Signups from 2026 onward
- Rationale: Random split would leak the time-drift pattern and overestimate performance.

### 5.2 Evaluation Metrics
- **Primary:** F1-score (balances precision/recall on the minority class)
- **Secondary:** ROC-AUC, Precision-Recall AUC
- **Report but don't optimize:** Accuracy (misleading on imbalanced data)

## 6. Baseline Benchmark

Before building the primary model, establish a simple baseline:
- **Dummy classifier** (always predict majority class): F1 = 0.0, Accuracy = 86.5%
- **logistic regression** with all non-leaked features: provides a lower-bound AUC
- Target: GBT should achieve **F1 > 0.35** and **AUC > 0.75** to be considered an improvement over baseline.

## 7. Risk Factors

| Risk | Mitigation |
|------|-----------|
| Churn label definition may be circular with recency features | Confirm label definition; exclude days-since-last-txn |
| Data may not be stationary (churn rate ↑ over time) | Time-based validation; signup cohort features |
| Parquet vs DB inconsistency | Use DB for all feature computation |
| Small dataset (1,500 customers) | Cross-validation; simpler models; no DNN |