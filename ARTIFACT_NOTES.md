# Artifact Notes — Self-Critique Log

> Generated at build time. Documents what was reconsidered, what went smoothly,
> and what specific checks were verified before concluding.

---

## Things I Got Wrong or Reconsidered

### 1. Multi-Source Mismatch Count (Greater Than First Reported)

**Initial assumption (Stage 0):** The draft said "212 count mismatches" between DB
and parquet. On re-checking during the build:
- **102 rows** are in the DB but entirely absent from the parquet (left_only).
- **110** of the 8,086 matched rows have count/amount mismatches.
- Total effect = 212 individual rows with some form of discrepancy.
- The *magnitude* (212 total affected rows) was correct, but conflating left_only
  with within-match mismatches was imprecise. The build separates these clearly.

### 2. As-of-Date Edge Case

**Initial assumption:** `as_of_date = "2026-07-31"` with a strict `>` comparison
flagged 38 transactions as "after as-of." These are all on July 31, 2026 itself
(max timestamp = 2026-07-31 23:50:29). The fix was to use end-of-day as the cutoff
(`>= 2026-08-01`), which correctly passes the check. This is a minor implementation
detail, not a data quality issue.

### 3. Avg Transaction Value Effect Size

**Expected:** The EDA draft's claim that avg_txn_value is "near-identical" was correct
in conclusion but lacked a formal test. The build confirmed: Cohen's d = 0.01 — a
vanishingly small effect. P=0.18, not significant. This is now properly documented
with the test result.

### 4. Outlier Grain

**Reconsidered:** The outlier rate for transaction amounts differs depending on
whether you compute at the transaction level (6.67% IQR outliers) vs. at the
per-customer average level (~4.9%). The build uses raw transaction-level for
profiling and customer-level for hypothesis testing — both are documented with
their specific grain.

---

## Checks Performed Before Sign-Off

### Verification Pass
- ✅ `scripts/verify_findings.py` passes — every claim in EDA_FINDINGS.md reproduces
  from raw data.
- ✅ Mutation test: changing the churn rate from 13.47% to 12.00% in a scratch copy
  was caught by the verify script (tolerance=0.001). The test is sensitive enough.

### Leakage-Circularity Flag
- ✅ Re-verified: churned min DSL=90, non-churned max DSL=89. Perfect separation.
- ✅ Correlation with churn = 0.741 — the strongest predictor, and definitionally
  circular.
- ✅ 81 zero-activity customers all churned — label definition confirmed as
  capturing "no activity in observation window."
- ✅ MODEL_RECOMMENDATION.md explicitly excludes all leakage-tainted variables.
  No recommendation relies on them.

### Multi-Source Comparison
- ✅ Actual re-aggregation performed: transactions.db grouped to customer×month
  and compared against parquet month by month.
- ✅ Not assumed to agree — 110 count mismatches found, all in July 2026.
- ✅ 19.28% shortfall quantified.

### Zero-Activity vs Missing Distinction
- ✅ Confirmed: 81 customers = 5.4% with zero rows in DB. These are legitimate
  zero-activity customers, not missing data.
- ✅ 100% churned rate among them.
- ✅ Treated as a distinct subpopulation throughout findings and recommendation.

### Hypothesis Test Appropriateness
- ✅ Each test matched to its variable's shape (skewed → MWU, categorical → χ²).
- ✅ No t-test used (normality assumption violated by every continuous variable).
- ✅ Effect sizes reported alongside p-values.

---

## Remaining Caveats

1. The churn label's exact definition was not provided in a data dictionary — the
   "90-day inactivity" conclusion is inferred from the perfect separation threshold.
   If the label uses a different window length (e.g., 60 or 120 days), the exact
   leakage boundary shifts but the circularity concern remains.

2. All three data sources are zero-missingness — this is unusually clean for
   production data. Real-world churn EDA would spend more effort on null handling.

3. The notebook hasn't been manually rendered or visually inspected for plot
   quality (font sizes, legend overlap, color accessibility) in this build pass —
   this would normally be a visual check before submission.