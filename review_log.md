# Self-Review Log

> Step 3 review of the full build diff — loaders, profiling, hypothesis tests,
> time/leakage checks, notebook, scripts, findings docs, and tests.
> Each finding was either fixed or explicitly noted as left for a stated reason.

---

## 1. EDA_FINDINGS.md — Claim Reproducibility

| Check | Result |
|-------|--------|
| `scripts/verify_findings.py` passes | ✅ All 30+ claims verified |
| Mutation test (changed churn rate to 12%) | ✅ Caught by verify script (tolerance=0.001) |
| Every hypothesis test result matches raw data | ✅ Confirmed |
| Zero-activity counts match raw data | ✅ Confirmed |

**Finding:** None. All claims trace to reproducible computations.

---

## 2. MODEL_RECOMMENDATION.md — Leakage-Tainted Variables

| Check | Result |
|-------|--------|
| `days_since_last_txn` explicitly excluded | ✅ Yes — §4 "Excluded" section |
| `trailing_90d_*` features explicitly excluded | ✅ Yes |
| Recommendation reasoning doesn't rely on leaked features | ✅ Yes — uses full-history aggregates and categoricals only |
| Alternative families considered | ✅ RF, Logistic Regression, DNN all evaluated |

**Finding:** None. All leakage-tainted variables are explicitly excluded.

---

## 3. DATA_ISSUES.md — Issue Verification

| Issue | Severity | Verified Number | Fixed? |
|-------|----------|-----------------|--------|
| Label leakage (90-day circularity) | CRITICAL | corr=0.741, gap=1 day | ✅ |
| Multi-source parquet incomplete | HIGH | 19.28% shortfall, all July 2026 | ✅ |
| Time non-stationarity | HIGH | 9.1% → 27.8% churn (p<0.0001) | ✅ |
| Class imbalance | MEDIUM | 13.5% churn, 6.4:1 ratio | ✅ |
| Zero-activity subpopulation | MEDIUM | 81 customers, 100% churned | ✅ Was a blank in Stage 0 draft |
| Skewed amounts | LOW | 6.67% IQR outliers | ✅ |
| No missing values | LOW | 0% across all sources | ✅ |

**Finding:** The zero-activity subpopulation was a gap in Stage 0's draft that I identified during
the build and added as Issue 5 in DATA_ISSUES.md. This is correctly handled.

---

## 4. Hypothesis Tests — Appropriateness

| Driver | Test Used | Matched to Shape? | Result |
|--------|-----------|-------------------|--------|
| total_spend | Mann-Whitney U | ✅ (right-skewed) | Significant (p=0.0001) |
| txn_count | Mann-Whitney U | ✅ (discrete, skewed) | Significant (p=0.0002) |
| distinct_categories | Mann-Whitney U | ✅ (bounded ordinal) | Significant (p=0.0006) |
| avg_txn_value | Mann-Whitney U | ✅ (ratio, skewed) | Not significant (p=0.184) |
| return_count | Mann-Whitney U | ✅ (zero-inflated) | Not significant (p=0.147) |
| is_returned_rate | Mann-Whitney U | ✅ (zero-inflated proportion) | Not significant (p=0.800) |
| acquisition_channel | Chi-square | ✅ (categorical) | Not significant (p=0.120) |
| region | Chi-square | ✅ (categorical) | Not significant (p=0.289) |
| signup_year | Chi-square | ✅ (categorical/cohort) | Significant (p<0.0001) |

**Finding:** None. All tests are matched to their variable's distribution shape.
No t-tests used (normality assumption violated by every continuous variable).

---

## 5. Test Suite

| File | Tests | Status |
|------|-------|--------|
| `test_load.py` | 9 | ✅ Pass |
| `test_profile.py` | 7 | ✅ Pass |
| `test_hypotheses.py` | 5 | ✅ Pass |
| `test_time_and_leakage_checks.py` | 10 | ✅ Pass |
| `test_findings_reproduce.py` | 23 | ✅ Pass |
| **Total** | **55** | **✅ All pass** |

**Findings:**
- `test_findings_reproduce.py` uses `==` instead of `is` for numpy boolean comparison
  (was: `assert hyp["total_spend"]["significant"] is True` → `== True`). Fixed after first
  pytest run caught the `np.True_` identity mismatch.
- `test_hypotheses.py::test_run_all_hypothesis_tests_return_structure` expected 8 results
  but `run_all_hypothesis_tests()` returns 9 (includes `region` test). Fixed by updating
  the assertion to `== 9`.

---

## 6. As-of-Date Discipline

**Finding:** Initial check using `> as_of_date` flagged 38 transactions on July 31, 2026
as "after as-of." Updated the cutoff to `>= next_day` (end-of-day boundary), which correctly
passes. These transactions are legitimate July data.

**Status:** ✅ Resolved. No future data leaks detected (all txns ≤ 2026-07-31 23:50:29).

---

## 7. Multi-Source Comparison

**Finding:** The merge column naming was non-obvious — the `suffixes=("_db", "_pq")` applies
only to columns present in both DataFrames. `txn_month` gets `_db`/`_pq`, but parquet columns
like `txn_count` keep their original names (no collision). The initial implementation used
`txn_count_pq` which didn't exist. Fixed by using the correct column name.

**Status:** ✅ Resolved. Multi-source check now correctly compares `txn_count_db` vs `txn_count`.

---

## Items Left Unresolved

1. **Churn label definition is inferred, not confirmed.** The "90-day inactivity" definition
   is the strongest hypothesis from the data (perfect separation at threshold 90), but no
   data dictionary confirms it. If the actual label window differs (e.g., 60 or 120 days),
   the exact boundary changes but the circularity concern remains. This is a documentation
   note, not a code defect.

2. **Notebook has not been visually rendered.** The notebook cells are designed to run
   top to bottom and produce correct output, but visual checks (font sizes, plot quality,
   color accessibility) were not performed in this automated build pass. This would be a
   manual check before submission.

3. **No missing values in any source.** This is unusual for real data. Implementations
   should not assume this pattern holds in production, but nothing to fix for this
   assignment's scope.