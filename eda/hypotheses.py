"""Hypothesis tests for each candidate churn driver.

One pure function per driver, each returning (statistic, p_value, effect_size, direction, test_name).
Tests are matched to each variable's actual distribution shape:
  - Categorical vs binary churn  → chi-square test of independence
  - Continuous/numeric, skewed   → Mann-Whitney U
  - Continuous/numeric, normal   → independent t-test (if assumption holds)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def _cohens_d(series_a: pd.Series, series_b: pd.Series) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(series_a), len(series_b)
    s1, s2 = series_a.var(ddof=1), series_b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float(abs(series_a.mean() - series_b.mean()) / pooled)


def test_total_spend(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: total_spend (right-skewed monetary) vs churn."""
    merged = df.merge(tx_agg, on="customer_id", how="left")
    churned = merged.loc[merged["churned"] == 1, "total_spend"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "total_spend"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_txn_count(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: txn_count (discrete count, skewed) vs churn."""
    merged = df.merge(tx_agg, on="customer_id", how="left")
    churned = merged.loc[merged["churned"] == 1, "txn_count"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "txn_count"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_distinct_categories(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: distinct_categories (bounded ordinal) vs churn."""
    merged = df.merge(tx_agg, on="customer_id", how="left")
    churned = merged.loc[merged["churned"] == 1, "distinct_categories"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "distinct_categories"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_avg_txn_value(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: avg_txn_value (ratio of skewed vars) vs churn."""
    merged = df.merge(tx_agg, on="customer_id", how="left")
    churned = merged.loc[merged["churned"] == 1, "avg_txn_value"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "avg_txn_value"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_return_count(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: return_count (zero-inflated count) vs churn."""
    merged = df.merge(tx_agg, on="customer_id", how="left")
    churned = merged.loc[merged["churned"] == 1, "return_count"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "return_count"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_is_returned_rate(
    df: pd.DataFrame, tx_agg: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Mann-Whitney U: is_returned rate per customer (zero-inflated proportion) vs churn.

    is_returned_rate = return_count / txn_count, representing the proportion
    of a customer's transactions that are returns. Zero-inflated and strongly
    non-normal, so Mann-Whitney U is the appropriate test.
    """
    merged = df.merge(tx_agg, on="customer_id", how="left")
    # Compute return rate; customers with zero transactions get NaN — treat as 0 rate
    merged["is_returned_rate"] = np.where(
        merged["txn_count"] > 0,
        merged["return_count"] / merged["txn_count"],
        0.0
    )
    churned = merged.loc[merged["churned"] == 1, "is_returned_rate"].dropna()
    not_churned = merged.loc[merged["churned"] == 0, "is_returned_rate"].dropna()
    stat, p = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")
    d = _cohens_d(churned, not_churned)
    direction = "lower" if churned.mean() < not_churned.mean() else "higher"
    return stat, p, d, direction, "Mann-Whitney U"


def test_acquisition_channel(
    df: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Chi-square: acquisition_channel (categorical, 4 levels) vs churn."""
    ct = pd.crosstab(df["acquisition_channel"], df["churned"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    # Cramer's V as effect size
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    direction = f"highest_rate: {ct[1].idxmax() if 1 in ct else 'N/A'}"
    return chi2, p, float(v), direction, "Chi-square"


def test_region(
    df: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Chi-square: region (categorical, 4 levels) vs churn."""
    ct = pd.crosstab(df["region"], df["churned"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    direction = f"highest_rate: {ct[1].idxmax() if 1 in ct else 'N/A'}"
    return chi2, p, float(v), direction, "Chi-square"


def test_signup_year(
    df: pd.DataFrame
) -> Tuple[float, float, float, str, str]:
    """Chi-square: signup_year (3 cohort levels) vs churn."""
    dfc = df.copy()
    dfc["signup_year"] = pd.to_datetime(dfc["signup_date"]).dt.year
    ct = pd.crosstab(dfc["signup_year"], dfc["churned"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    denom = min(k - 1, r - 1)
    v = np.sqrt(phi2 / denom) if denom > 0 else 0.0
    direction = f"highest_rate: cohort {ct[1].idxmax() if 1 in ct else 'N/A'}"
    return chi2, p, float(v), direction, "Chi-square"


def run_all_hypothesis_tests(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
) -> list:
    """Run every hypothesis test and return a list of result dicts.

    This is the main entrypoint for scripts/run_hypothesis_tests.py.
    """
    # Pre-compute transaction-level aggregates
    tx_agg = df_transactions.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        distinct_categories=("category", "nunique"),
        avg_txn_value=("amount", "mean"),
        return_count=("is_returned", "sum"),
    ).reset_index()
    # is_returned_rate computed at test time (needs txn_count to avoid div by zero)

    df_cp = df_customers.copy()
    df_cp["signup_year"] = pd.to_datetime(df_cp["signup_date"]).dt.year

    tests = [
        ("total_spend", "Total spend (monetary, skewed)", test_total_spend(df_cp, tx_agg)),
        ("txn_count", "Transaction count (discrete, skewed)", test_txn_count(df_cp, tx_agg)),
        ("distinct_categories", "Distinct categories (bounded ordinal)", test_distinct_categories(df_cp, tx_agg)),
        ("avg_txn_value", "Avg transaction value (ratio, skewed)", test_avg_txn_value(df_cp, tx_agg)),
        ("return_count", "Return count (zero-inflated)", test_return_count(df_cp, tx_agg)),
        ("is_returned_rate", "Return rate per customer (zero-inflated proportion)", test_is_returned_rate(df_cp, tx_agg)),
        ("acquisition_channel", "Acquisition channel (categorical, 4 levels)", test_acquisition_channel(df_cp)),
        ("region", "Region (categorical, 4 levels)", test_region(df_cp)),
        ("signup_year", "Signup year (categorical, 3 cohorts)", test_signup_year(df_cp)),
    ]

    results = []
    for key, label, (stat, p, effect, direction, test_name) in tests:
        results.append({
            "driver": key,
            "label": label,
            "test": test_name,
            "statistic": round(stat, 4),
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "effect_size": round(effect, 4),
            "direction": direction,
        })

    return results