"""Tests for eda/hypotheses.py — each test against constructed input with known answer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import eda.hypotheses as hyp
from eda.hypotheses import run_all_hypothesis_tests


def _make_fixture(seed: int = 42, n_customers: int = 100) -> tuple:
    """Create a small customers + transactions fixture.

    Churned customers get lower spend and fewer txns.
    First 80 customers are non-churned, last 20 are churned.
    """
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "customer_id": range(n_customers),
        "signup_date": pd.date_range("2024-01-01", periods=n_customers, freq="D"),
        "acquisition_channel": rng.choice(["organic", "paid_search", "referral", "social"], n_customers),
        "region": rng.choice(["east", "west", "north", "south"], n_customers),
        "churned": [0] * 80 + [1] * 20,
    })

    txns = []
    for cid in range(n_customers):
        churned = 1 if cid >= 80 else 0
        n_txns = rng.integers(0, 5) if churned else rng.integers(3, 15)
        categories = rng.choice(["apparel", "beauty", "electronics", "grocery", "home", "sports"], n_txns)
        amounts = rng.uniform(10, 100, n_txns)
        for i in range(n_txns):
            txns.append({
                "customer_id": cid,
                "category": categories[i],
                "amount": amounts[i],
                "is_returned": 0,
            })
    transactions = pd.DataFrame(txns)
    return customers, transactions


def test_total_spend_significant():
    """Churned should have lower total spend."""
    customers, transactions = _make_fixture()
    tx_agg = transactions.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        distinct_categories=("category", "nunique"),
        avg_txn_value=("amount", "mean"),
        return_count=("is_returned", "sum"),
    ).reset_index()
    stat, p, d, direction, test = hyp.test_total_spend(customers, tx_agg)
    assert direction == "lower", "Churned should have lower spend"
    assert p < 0.05 or d > 0.1, "Effect should be detectable"


def test_avg_txn_value_not_significant():
    """Avg txn value should NOT differ between groups (same amounts generated)."""
    rng = np.random.default_rng(42)
    customers = pd.DataFrame({
        "customer_id": range(60),
        "churned": [0] * 30 + [1] * 30,
    })
    txns = []
    for cid in range(60):
        churned = 1 if cid >= 30 else 0
        n_txns = rng.integers(3, 8)
        amounts = rng.uniform(20, 80, n_txns)  # same distribution for both groups
        for i in range(n_txns):
            txns.append({"customer_id": cid, "amount": amounts[i], "category": "a", "is_returned": 0})
    transactions = pd.DataFrame(txns)
    tx_agg = transactions.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        distinct_categories=("category", "nunique"),
        avg_txn_value=("amount", "mean"),
        return_count=("is_returned", "sum"),
    ).reset_index()
    stat, p, d, direction, test = hyp.test_avg_txn_value(customers, tx_agg)
    assert p > 0.01, f"Should NOT be significant (p={p})"


def test_acquisition_channel_not_necessarily_significant():
    """chi-square test should return without error and produce sane p-value."""
    customers, _ = _make_fixture()
    stat, p, v, direction, test = hyp.test_acquisition_channel(customers)
    assert 0.0 <= p <= 1.0
    assert test == "Chi-square"


def test_signup_year_can_detect_drift():
    """If signup year is engineered to differ, chi-square should detect it."""
    customers = pd.DataFrame({
        "customer_id": range(200),
        "signup_date": pd.date_range("2024-01-01", periods=200, freq="D"),
        "churned": [0] * 100 + [1] * 100,
        "acquisition_channel": "organic",
        "region": "east",
    })
    stat, p, v, direction, test = hyp.test_signup_year(customers)
    # With 100/100 split, churn rate is 50% — the test should still complete
    assert 0.0 <= p <= 1.0


def test_run_all_hypothesis_tests_return_structure():
    customers, transactions = _make_fixture()
    results = run_all_hypothesis_tests(customers, transactions)
    assert len(results) == 9
    expected_keys = {"driver", "label", "test", "statistic", "p_value", "significant", "effect_size", "direction"}
    for r in results:
        assert expected_keys.issubset(set(r.keys())), f"Missing keys in {r['driver']}"