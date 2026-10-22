"""Tests that every numeric claim in EDA_FINDINGS.md reproduces from raw data.

This is the highest-value test in this suite — it catches plausible-but-wrong
numbers in the findings document. Tolerances are tight (0.001-0.5 depending on
the claim), and any mismatch is a hard failure.

If EDA_FINDINGS.md changes, update the expected values in this test to match.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy import stats

from eda.load import load_customers, load_transactions_db, load_transactions_agg
from eda.profile import (
    distribution_stats,
    outlier_count_iqr,
    count_zero_activity_customers,
    churn_rate_by_group,
)
from eda.hypotheses import run_all_hypothesis_tests
from eda.time_checks import check_multi_source_agreement
from eda.leakage_checks import check_days_since_last_txn_separation


# ---------------------------------------------------------------------------
# Fixtures (load once per session)
# ---------------------------------------------------------------------------
CUSTOMERS, _ = load_customers()
TRANSACTIONS, _ = load_transactions_db()
AGG, _ = load_transactions_agg()


# ---------------------------------------------------------------------------
# 1. Data Overview — source row counts
# ---------------------------------------------------------------------------
class TestDataOverview:
    def test_total_customers(self):
        assert len(CUSTOMERS) == 1500

    def test_total_transactions(self):
        assert len(TRANSACTIONS) == 11742

    def test_total_agg_rows(self):
        assert len(AGG) == 8086


# ---------------------------------------------------------------------------
# 2. Churn rate & imbalance
# ---------------------------------------------------------------------------
class TestChurnRate:
    def test_overall_churn_rate(self):
        rate = CUSTOMERS["churned"].mean()
        assert abs(rate - 0.1347) < 0.001, f"Got {rate}"

    def test_churned_count(self):
        assert CUSTOMERS["churned"].sum() == 202

    def test_non_churned_count(self):
        assert (1 - CUSTOMERS["churned"]).sum() == 1298


# ---------------------------------------------------------------------------
# 3. Zero-activity customers
# ---------------------------------------------------------------------------
class TestZeroActivity:
    def test_zero_activity_count(self):
        za = count_zero_activity_customers(CUSTOMERS, TRANSACTIONS)
        assert za["n_zero_activity"] == 81, f"Got {za['n_zero_activity']}"

    def test_zero_activity_all_churned(self):
        za = count_zero_activity_customers(CUSTOMERS, TRANSACTIONS)
        assert za["n_churned"] == 81
        assert za["pct_churned"] == 100.0

    def test_zero_activity_share_of_churn(self):
        za = count_zero_activity_customers(CUSTOMERS, TRANSACTIONS)
        assert abs(za["pct_of_all_churned"] - 40.1) < 0.5


# ---------------------------------------------------------------------------
# 4. Churn by acquisition channel
# ---------------------------------------------------------------------------
class TestChurnByChannel:
    def test_organic_churn_rate(self):
        chan = churn_rate_by_group(CUSTOMERS, "acquisition_channel")
        rates = dict(zip(chan["acquisition_channel"], chan["churn_rate"]))
        assert abs(rates.get("organic", 0) - 0.1527) < 0.01


# ---------------------------------------------------------------------------
# 5. Churn by signup year
# ---------------------------------------------------------------------------
class TestChurnByYear:
    def test_signup_year_rates(self):
        df = CUSTOMERS.copy()
        df["signup_year"] = pd.to_datetime(df["signup_date"]).dt.year
        yearly = df.groupby("signup_year")["churned"].mean()
        assert abs(yearly.loc[2024] - 0.0912) < 0.01
        assert abs(yearly.loc[2026] - 0.2776) < 0.01


# ---------------------------------------------------------------------------
# 6. Hypothesis test results
# ---------------------------------------------------------------------------
class TestHypothesisResults:
    def test_total_spend_significant(self):
        results = run_all_hypothesis_tests(CUSTOMERS, TRANSACTIONS)
        hyp = {h["driver"]: h for h in results}
        assert hyp["total_spend"]["significant"] == True
        assert hyp["total_spend"]["direction"] == "lower"

    def test_avg_txn_not_significant(self):
        results = run_all_hypothesis_tests(CUSTOMERS, TRANSACTIONS)
        hyp = {h["driver"]: h for h in results}
        assert hyp["avg_txn_value"]["significant"] == False

    def test_return_count_not_significant(self):
        results = run_all_hypothesis_tests(CUSTOMERS, TRANSACTIONS)
        hyp = {h["driver"]: h for h in results}
        assert hyp["return_count"]["significant"] == False

    def test_acquisition_channel_not_significant(self):
        results = run_all_hypothesis_tests(CUSTOMERS, TRANSACTIONS)
        hyp = {h["driver"]: h for h in results}
        assert hyp["acquisition_channel"]["significant"] == False

    def test_signup_year_highly_significant(self):
        results = run_all_hypothesis_tests(CUSTOMERS, TRANSACTIONS)
        hyp = {h["driver"]: h for h in results}
        assert hyp["signup_year"]["significant"] == True


# ---------------------------------------------------------------------------
# 7. Multi-source agreement
# ---------------------------------------------------------------------------
class TestMultiSourceAgreement:
    def test_left_only_count(self):
        msa = check_multi_source_agreement(TRANSACTIONS, AGG)
        assert msa["n_left_only"] == 102

    def test_right_only_zero(self):
        msa = check_multi_source_agreement(TRANSACTIONS, AGG)
        assert msa["n_right_only"] == 0

    def test_all_mismatches_in_july_2026(self):
        msa = check_multi_source_agreement(TRANSACTIONS, AGG)
        assert msa["mismatched_months"] == ["2026-07"]

    def test_july_shortfall(self):
        msa = check_multi_source_agreement(TRANSACTIONS, AGG)
        assert abs(msa["july_pct_shortfall"] - 19.28) < 0.5


# ---------------------------------------------------------------------------
# 8. Leakage — days_since_last_txn separation
# ---------------------------------------------------------------------------
class TestLeakage:
    def test_churned_min_dsl(self):
        leak = check_days_since_last_txn_separation(CUSTOMERS, TRANSACTIONS)
        assert leak["churned_min_days_since_last_txn"] == 90

    def test_non_churned_max_dsl(self):
        leak = check_days_since_last_txn_separation(CUSTOMERS, TRANSACTIONS)
        assert leak["non_churned_max_days_since_last_txn"] == 89

    def test_perfect_separation(self):
        leak = check_days_since_last_txn_separation(CUSTOMERS, TRANSACTIONS)
        assert leak["perfect_separation"] is True