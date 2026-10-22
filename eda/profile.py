"""Profiling functions: schema, missingness, distributions, outliers, zero-activity.

All functions are pure — they take DataFrames and return structured results.
"""

import pandas as pd
import numpy as np


def schema_summary(df: pd.DataFrame, name: str = "") -> dict:
    """Return column names, dtypes, null counts, and null percentages."""
    summary = []
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / len(df) * 100, 2),
            "n_unique": int(df[col].nunique()) if df[col].dtype in ("object", "category") else None,
        })
    return {"source": name, "rows": len(df), "columns": summary}


def distribution_stats(s: pd.Series) -> dict:
    """Mean, median, std, skew direction, min, max for a numeric series."""
    stats = {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "p75": round(float(s.quantile(0.75)), 2),
    }
    # Determine skew direction
    if stats["mean"] > stats["median"] * 1.05:
        stats["skew"] = "right"
    elif stats["mean"] < stats["median"] * 0.95:
        stats["skew"] = "left"
    else:
        stats["skew"] = "approximately_symmetric"
    return stats


def outlier_count_iqr(s: pd.Series) -> dict:
    """Count outliers using the IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_low = int((s < lower).sum())
    n_high = int((s > upper).sum())
    return {
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "n_outliers_low": n_low,
        "n_outliers_high": n_high,
        "n_outliers_total": n_low + n_high,
        "pct_outliers": round((n_low + n_high) / s.count() * 100, 2),
    }


def count_zero_activity_customers(df_customers: pd.DataFrame, df_transactions: pd.DataFrame) -> dict:
    """Count customers with zero transactions.

    Customers in df_customers who have no matching customer_id in df_transactions.
    Also reports how many of those zero-activity customers churned.
    """
    tx_customer_ids = set(df_transactions["customer_id"].unique())
    all_customer_ids = set(df_customers["customer_id"])
    zero_active = all_customer_ids - tx_customer_ids
    n_zero = len(zero_active)
    zero_df = df_customers[df_customers["customer_id"].isin(zero_active)]
    n_churned = int(zero_df["churned"].sum())
    return {
        "n_zero_activity": n_zero,
        "pct_of_customers": round(n_zero / len(df_customers) * 100, 2),
        "n_churned": n_churned,
        "pct_churned": round(n_churned / n_zero * 100, 2) if n_zero > 0 else 0.0,
        "pct_of_all_churned": round(n_churned / df_customers["churned"].sum() * 100, 2),
    }


def churn_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Churn count, total, and rate grouped by a categorical column."""
    result = df.groupby(group_col)["churned"].agg(
        n_customers="count",
        n_churned="sum",
    ).reset_index()
    result["churn_rate"] = (result["n_churned"] / result["n_customers"]).round(4)
    result = result.sort_values("churn_rate", ascending=False)
    return result