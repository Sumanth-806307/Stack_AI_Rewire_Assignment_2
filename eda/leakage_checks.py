"""Leakage checks: label-circularity detection and as-of-date discipline.

Functions here identify candidate drivers whose computation window overlaps
the churn label's trailing-90-day definition, and verify that no future data
leaked into profiling or hypothesis-test computations.
"""

import pandas as pd
import numpy as np
from typing import List


LEAKAGE_TAINTED_VARS = [
    "days_since_last_txn",
    "trailing_90d_spend",
    "trailing_90d_txn_count",
    "trailing_90d_return_count",
    "inactive_90d_flag",
]

LEAKAGE_SAFE_VARS = [
    "total_spend",
    "txn_count",
    "avg_txn_value",
    "max_txn_value",
    "distinct_categories",
    "distinct_channels",
    "return_count",
    "return_rate",
    "customer_tenure_days",
    "signup_year",
    "signup_quarter",
    "acquisition_channel",
    "region",
]


def check_days_since_last_txn_separation(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
    ref_date: str = "2026-07-31",
) -> dict:
    """Check if days_since_last_txn perfectly separates churned vs non-churned.

    This is the primary leakage-circularity test: if the churn label is defined
    as 'no transactions in the trailing 90 days', then a feature encoding the
    same 90-day window will be perfectly (but circularly) predictive.

    Returns a dict with:
    - churned_min_dsl: min days_since_last_txn among churned customers
    - non_churned_max_dsl: max days_since_last_txn among non-churned customers
    - separation_gap: churned_min - non_churned_max (negative = overlap)
    - perfect_separation: True if gap >= 0
    - correlation_with_label: Pearson r
    - n_zero_activity_and_churned: zero-activity customers who are churned
    """
    ref = pd.Timestamp(ref_date)

    # Compute last transaction per customer
    last_tx = df_transactions.groupby("customer_id")["txn_timestamp"].max().reset_index()
    last_tx["last_txn"] = pd.to_datetime(last_tx["txn_timestamp"])
    last_tx["days_since_last_txn"] = (ref - last_tx["last_txn"]).dt.days

    merged = df_customers.merge(last_tx, on="customer_id", how="left")

    # Zero-activity customers have NaN days_since_last_txn
    has_tx = merged["days_since_last_txn"].notna()
    churned = merged["churned"] == 1

    churned_min = merged.loc[churned & has_tx, "days_since_last_txn"].min()
    non_churned_max = merged.loc[~churned & has_tx, "days_since_last_txn"].max()
    churned_min_val = int(churned_min) if pd.notna(churned_min) else None
    non_churned_max_val = int(non_churned_max) if pd.notna(non_churned_max) else None

    gap = None
    if churned_min_val is not None and non_churned_max_val is not None:
        gap = churned_min_val - non_churned_max_val

    corr = merged["days_since_last_txn"].corr(merged["churned"])

    # Count zero-activity churned customers
    zero_active = merged[merged["days_since_last_txn"].isna()]
    zero_churned = int(zero_active["churned"].sum())

    return {
        "churned_min_days_since_last_txn": churned_min_val,
        "non_churned_max_days_since_last_txn": non_churned_max_val,
        "separation_gap": gap,
        "perfect_separation": gap is not None and gap >= 0,
        "correlation_with_churn": round(float(corr), 4),
        "n_zero_activity_and_churned": zero_churned,
    }


def check_as_of_date_discipline(
    df_customers: pd.DataFrame,
    df_transactions: pd.DataFrame,
    as_of_date: str = "2026-07-31",
) -> dict:
    """Verify no transactions after the as-of date are used.

    If any transaction has a timestamp after the dataset's as-of date,
    it would be a data leak (future data used to predict the present).

    Uses end-of-day 2026-07-31 as cutoff (23:59:59), since all data
    is timestamped within July 31, 2026.
    """
    as_of_end = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)  # 2026-08-01 00:00:00
    max_txn_date = df_transactions["txn_timestamp"].max()
    n_after = int((df_transactions["txn_timestamp"] >= as_of_end).sum())

    return {
        "as_of_date": as_of_date,
        "as_of_cutoff": str(as_of_end),
        "max_txn_timestamp": str(max_txn_date),
        "n_transactions_after_cutoff": n_after,
        "passes_check": n_after == 0,
    }


def flag_leakage_tainted_variables(
    eda_defined_vars: List[str],
) -> dict:
    """Check which of the user's candidate drivers are leakage-tainted.

    Args:
        eda_defined_vars: list of variable names considered as candidate drivers

    Returns:
        dict with 'tainted', 'safe', and 'unknown' lists
    """
    tainted = [v for v in eda_defined_vars if v in LEAKAGE_TAINTED_VARS]
    safe = [v for v in eda_defined_vars if v in LEAKAGE_SAFE_VARS]
    unknown = [v for v in eda_defined_vars if v not in LEAKAGE_TAINTED_VARS and v not in LEAKAGE_SAFE_VARS]
    return {
        "tainted": tainted,
        "safe": safe,
        "unknown": unknown,
    }