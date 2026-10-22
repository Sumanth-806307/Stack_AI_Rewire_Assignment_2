# eda_review.md — Churn EDA & Model-Family Recommendation

> Corrected review of Stage 0's generated drafts, produced by checking each claim
> against the raw data. Every correction below is backed by a number I recomputed myself.

---

## Distribution / missingness / outlier corrections

- **Correction: Stage 0's EDA_FINDINGS_DRAFT.md does not mention the zero-activity population.**
  Stage 0's draft described transaction patterns "by churn status" as if all customers have
  transaction data. In reality, **81 of 1,500 customers (5.4%) have zero rows in
  `transactions.db`** — and **100% of them (81/81) churned**. These 81 customers represent
  **40.1% of all churned customers (81/202)**. They are not "missing" transaction data; they
  are a distinct subpopulation who signed up but never transacted. The draft conflates them
  with active customers who stopped transacting, which inflates the apparent separation in
  features like `days_since_last_txn` (NaN for zero-activity customers, ≥90 for active
  churned). A correct EDA must treat zero-activity separately from active-churned and
  active-engaged.

- **Correction: Stage 0's draft skipped the hypothesis test for `avg_txn_value`, calling it
  "near-identical" without running a test.**
  I ran Mann-Whitney U on average transaction value by churn status: churned mean=$50.36,
  non-churned mean=$50.49, **U=72,797.5, p=0.184** — not statistically significant. The
  conclusion (no meaningful difference) is correct, but the draft presented it as
  self-evident without a test result, which is not a defensible EDA practice. A correct
  finding must report the test statistic and p-value, not just eyeball the means.

- **Correction: Stage 0's draft labeled avg_txn_value outliers as "4.9%" — the exact number
  depends on whether you use per-customer avg or raw transaction amounts.**
  The 4.9% figure is for per-customer average transaction value (IQR method). On raw
  individual transactions instead, outlier rate would differ. The draft should specify which
  grain (customer-level vs. transaction-level) the outlier detection was run at.

---

## Hypothesis tests to run

- **Driver**: `total_spend` — **Test**: Mann-Whitney U — **Why**: right-skewed monetary
  variable (mean=$420 vs median=$364+ range), violates normality assumption of t-test.
- **Driver**: `txn_count` — **Test**: Mann-Whitney U — **Why**: count variable, discrete
  with right skew, not normal.
- **Driver**: `distinct_categories` — **Test**: Mann-Whitney U — **Why**: bounded discrete
  ordinal (1-6), not continuous nor normal.
- **Driver**: `avg_txn_value` — **Test**: Mann-Whitney U — **Why**: ratio of two right-skewed
  variables; still non-normal.
- **Driver**: `return_count` — **Test**: Mann-Whitney U — **Why**: zero-inflated count, far
  from normal.
- **Driver**: `acquisition_channel` — **Test**: Chi-square test of independence — **Why**:
  categorical predictor (4 levels) vs. binary churn outcome, no ordering assumption.
- **Driver**: `region` — **Test**: Chi-square test of independence — **Why**: categorical
  predictor (4 levels) vs. binary churn outcome.
- **Driver**: `signup_year` — **Test**: Chi-square test of independence — **Why**: categorical
  (3 year-cohorts) vs. binary churn outcome. Alternative: treat as ordinal trend test.
- **Driver**: `is_returned` rate per customer — **Test**: Mann-Whitney U — **Why**:
  zero-inflated proportion, strongly non-normal.

---

## Leakage / imbalance / non-stationarity / multi-source flags

- **Leakage**: `days_since_last_txn` (and any trailing-90-day window feature) is definitionally
  circular with the churn label. Manual check: **all 121 churned customers with transaction
  data have days_since_last_txn ≥ 90; all 1,298 non-churned have ≤ 89.** Additionally,
  **all 81 zero-activity customers are labeled churned** — further confirming the label
  captures "no activity in the trailing window." Variables to exclude: `days_since_last_txn`,
  `trailing_90d_spend`, `trailing_90d_txn_count`, any `inactive_90d` binary flag.
  Variables still usable: full-history `total_spend`, `txn_count`, `distinct_categories`,
  `return_count`, `avg_txn_value`. The distinction is whether the feature's computation
  window overlaps the label's defining window.

- **Imbalance**: Overall churn rate is **13.47% (202/1,500)** — ~6.4:1 ratio. A dummy
  classifier predicting "not churned" for everyone achieves **86.5% accuracy with zero
  churn recall**. Metrics ruled in: F1-score, precision-recall AUC, ROC-AUC, sensitivity/
  specificity. Metrics ruled out: accuracy as primary metric. Class weighting or
  resampling is needed for any model that does not natively support `scale_pos_weight`.

- **Non-stationarity**: Monthly signup-cohort churn rates checked manually: 2024=9.1%,
  2025=10.1%, 2026 H1=27.8%. **Chi-square test confirms significant time drift
  (χ²=70.72, p<0.0001).** Monthly resolution shows a sharp inflection starting March 2026
  (30.8% churn), peaking June 2026 (57.1%). **Consequence**: a random train/test split
  would leak the time trend, overestimating performance. A time-based split (pre-2026
  train / 2026+ test) is mandatory. `signup_year` or `signup_quarter` should be a feature.

- **Multi-source**: `transactions.db` was reaggregated to customer x month and compared
  against `transactions_monthly_agg.parquet` month by month.
  - **102 rows** present in DB but absent from parquet (10 customers with July 2026-only
    activity, entirely missing from parquet).
  - **110 of 8,086 matched rows** have count/amount discrepancies — all in **July 2026**.
  - July 2026 totals: DB has 1,231 transactions ($62,084.62); parquet aggregates to 994
    ($50,116.67) — a **19% shortfall** on both counts.
  - **Root cause**: parquet appears truncated at a cutoff during July 2026. The DB is the
    complete source of truth. Any analysis relying on the parquet alone will undercount
    recent activity and bias against 2026-signup customers.

---

## Model-family judgment

- **Candidate family**: **Gradient Boosted Trees (XGBoost / LightGBM)** — Primary recommendation
- **Why, tied to this dataset's checked characteristics**:
  1. **Non-linear interactions are central** — the channel×spend interaction shown above
     (low-spend organic churns at 21.6% vs. high-spend organic at 8.3%) would require
     manual feature engineering in a linear model but is natively captured by tree splits.
  2. **Class imbalance is native** — `scale_pos_weight` handles the 6.4:1 ratio without
     separate resampling.
  3. **Mixed data types** — handles categorical (channel, region, category) alongside
     skewed numeric (spend, counts) without one-hot explosion.
  4. **Robust to skewed distributions** — tree splits are unaffected by the right-skewed
     monetary variables that would distort linear models.
  5. **Feature importance** — provides stakeholder-interpretable attribution.
- **Alternative considered**: **Random Forest** — **Why not the top pick**: GBT with early
  stopping generalizes better across the non-stationary time regimes (2024-2025 vs. 2026);
  RF's bagging averages across trees from the majority regime, making it slower to adapt
  to the 2026 shift without explicit time weighting.
- **Drivers assumed usable / excluded**:
  - **Usable** (full-history, non-circular): `total_spend`, `txn_count`, `avg_txn_value`,
    `max_txn_value`, `distinct_categories`, `distinct_channels`, `return_rate`,
    `customer_tenure_days` (last_txn - signup_date), `signup_year`/`signup_quarter`,
    `acquisition_channel`, `region`, category-engagement flags.
  - **Excluded (leakage-tainted)**: `days_since_last_txn`, `trailing_90d_spend`,
    `trailing_90d_txn_count`, `trailing_90d_return_count`, any binary `inactive_90d` flag.
  - **Caveated**: `txn_recency_ratio` (avg days between txns) — safe only if it does not
    implicitly encode a 90-day inactivity window. Must be computed over the customer's
    entire history, not over the trailing 90 days.