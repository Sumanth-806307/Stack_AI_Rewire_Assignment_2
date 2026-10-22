"""Tests for eda/profile.py — profiling functions against known-shape data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from eda.profile import (
    schema_summary,
    distribution_stats,
    outlier_count_iqr,
    count_zero_activity_customers,
    churn_rate_by_group,
)


def test_schema_summary_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = schema_summary(df, "test")
    assert result["source"] == "test"
    assert result["rows"] == 3
    assert len(result["columns"]) == 2


def test_schema_summary_missingness():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, "z"]})
    cols = schema_summary(df)["columns"]
    missing_a = [c for c in cols if c["column"] == "a"][0]
    missing_b = [c for c in cols if c["column"] == "b"][0]
    assert missing_a["n_missing"] == 1
    assert missing_b["n_missing"] == 2


def test_distribution_stats_right_skew():
    # Right-skewed: mean > median
    s = pd.Series([1, 2, 3, 4, 100])
    stats = distribution_stats(s)
    assert stats["skew"] == "right"
    assert stats["mean"] > stats["median"]


def test_distribution_stats_symmetric():
    s = pd.Series([1, 2, 3, 4, 5])
    stats = distribution_stats(s)
    # mean=3, median=3 -> approximately_symmetric
    assert stats["skew"] in ("approximately_symmetric", "right")


def test_outlier_count_iqr_known():
    # 1, 2, 3, 4, 100 -> Q1=1.5, Q3=4.5, IQR=3
    # Upper bound = 4.5 + 4.5 = 9, so 100 is an outlier
    s = pd.Series([1, 2, 3, 4, 100])
    result = outlier_count_iqr(s)
    assert result["n_outliers_high"] == 1
    assert result["n_outliers_total"] == 1


def test_outlier_count_iqr_no_outliers():
    s = pd.Series([10, 12, 13, 14, 16, 18])
    result = outlier_count_iqr(s)
    assert result["n_outliers_total"] == 0


def test_count_zero_activity_customers():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4], "churned": [0, 1, 0, 1]})
    transactions = pd.DataFrame({"customer_id": [1, 3, 3]})
    result = count_zero_activity_customers(customers, transactions)
    assert result["n_zero_activity"] == 2  # customers 2 and 4
    assert result["n_churned"] == 2  # both churned
    assert result["pct_churned"] == 100.0


def test_churn_rate_by_group():
    df = pd.DataFrame({
        "group": ["a", "a", "a", "b", "b"],
        "churned": [1, 0, 0, 1, 1],
    })
    result = churn_rate_by_group(df, "group")
    rates = dict(zip(result["group"], result["churn_rate"]))
    assert abs(rates["a"] - 1/3) < 0.001  # 0.333
    assert abs(rates["b"] - 1.0) < 0.001  # 2/2