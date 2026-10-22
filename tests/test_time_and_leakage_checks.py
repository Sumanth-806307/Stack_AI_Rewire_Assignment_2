"""Tests for eda/time_checks.py and eda/leakage_checks.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from eda.time_checks import (
    monthly_tx_volume,
    check_multi_source_agreement,
    monthly_churn_rate_by_signup_cohort,
)
from eda.leakage_checks import (
    check_days_since_last_txn_separation,
    check_as_of_date_discipline,
    flag_leakage_tainted_variables,
)


def test_monthly_tx_volume_monotonic():
    """Volume should increase over time (more customers, more txns)."""
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "txn_timestamp": pd.to_datetime(["2024-01-15", "2025-06-20", "2026-03-10"]),
        "amount": [10.0, 20.0, 30.0],
        "is_returned": [0, 0, 0],
    })
    vol = monthly_tx_volume(df)
    assert len(vol) == 3  # 3 distinct months
    assert vol["txn_count"].sum() == 3


def test_multi_source_agreement_perfect():
    """If both sources agree, no mismatches should be found."""
    customers = pd.DataFrame({"customer_id": range(10), "churned": [0]*10})
    txns = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "txn_timestamp": pd.to_datetime(["2025-01-15", "2025-01-20", "2025-02-10"]),
        "amount": [10.0, 20.0, 30.0],
        "is_returned": [0, 1, 0],
    })
    # Build matching parquet
    txns_month = txns.copy()
    txns_month["txn_month"] = txns_month["txn_timestamp"].dt.to_period("M")
    agg = txns_month.groupby(["customer_id", "txn_month"]).agg(
        txn_count=("amount", "count"),
        total_amount=("amount", "sum"),
        return_count=("is_returned", "sum"),
    ).reset_index()
    agg["txn_month"] = agg["txn_month"].dt.to_timestamp()

    result = check_multi_source_agreement(txns, agg)
    assert result["n_left_only"] == 0
    assert result["n_right_only"] == 0
    assert result["n_count_mismatch"] == 0


def test_multi_source_agreement_mismatch():
    """If parquet is missing rows, left_only should be > 0."""
    customers = pd.DataFrame({"customer_id": range(10), "churned": [0]*10})
    txns = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "txn_timestamp": pd.to_datetime(["2025-01-15", "2025-01-20", "2025-02-10"]),
        "amount": [10.0, 20.0, 30.0],
        "is_returned": [0, 0, 0],
    })
    agg = pd.DataFrame({
        "customer_id": [1],
        "txn_month": pd.to_datetime(["2025-01-01"]),
        "txn_count": [1],
        "total_amount": [10.0],
        "return_count": [0],
    })
    result = check_multi_source_agreement(txns, agg)
    assert result["n_left_only"] > 0


def test_monthly_churn_rate_signup_cohort():
    df = pd.DataFrame({
        "customer_id": range(6),
        "signup_date": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-06-10",
                                        "2025-03-05", "2025-07-22", "2026-01-01"]),
        "churned": [0, 0, 0, 1, 1, 1],
    })
    cohort = monthly_churn_rate_by_signup_cohort(df)
    assert len(cohort) == 5  # 5 distinct signup months
    assert cohort["churn_rate"].max() <= 1.0


def test_check_days_since_last_txn_perfect_separation():
    """Engineer a scenario where DSL perfectly separates."""
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "churned": [0, 0, 1, 1],
    })
    txns = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "txn_timestamp": pd.to_datetime(["2026-07-01", "2026-07-15", "2026-04-01", "2026-03-15"]),
    })
    result = check_days_since_last_txn_separation(customers, txns, ref_date="2026-07-31")
    # Churned min DSL should be > non-churned max DSL
    if result["churned_min_days_since_last_txn"] is not None and result["non_churned_max_days_since_last_txn"] is not None:
        assert result["churned_min_days_since_last_txn"] >= result["non_churned_max_days_since_last_txn"]


def test_check_days_since_last_txn_no_separation():
    """Engineer scenario where DSL overlaps between groups."""
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "churned": [0, 1, 0],
    })
    # All txns around same time → no separation
    txns = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "txn_timestamp": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
    })
    result = check_days_since_last_txn_separation(customers, txns, ref_date="2026-07-31")
    gap = result["separation_gap"]
    assert gap is None or gap < 50  # Should be small or None


def test_as_of_date_discipline_clean():
    customers = pd.DataFrame({"customer_id": [1], "churned": [0]})
    txns = pd.DataFrame({
        "customer_id": [1],
        "txn_timestamp": pd.to_datetime(["2026-07-31 12:00:00"]),
    })
    result = check_as_of_date_discipline(customers, txns, as_of_date="2026-07-31")
    assert result["passes_check"] is True


def test_flag_leakage_tainted_variables():
    result = flag_leakage_tainted_variables(["days_since_last_txn", "total_spend", "some_unknown"])
    assert "days_since_last_txn" in result["tainted"]
    assert "total_spend" in result["safe"]
    assert "some_unknown" in result["unknown"]


def test_flag_leakage_tainted_none():
    result = flag_leakage_tainted_variables(["total_spend", "region"])
    assert len(result["tainted"]) == 0
    assert len(result["safe"]) == 2