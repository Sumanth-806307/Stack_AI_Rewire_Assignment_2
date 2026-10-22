"""Time-trend and multi-source agreement checks.

- Monthly transaction-volume trend from the row-level transactions table.
- Multi-source agreement: reaggregate transactions.db to customer x month
  and compare month by month against transactions_monthly_agg.parquet.
"""

import pandas as pd
import numpy as np


def monthly_tx_volume(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly transaction count and total amount from raw transactions.

    Returns a DataFrame indexed by calendar month with txn_count and total_amount.
    """
    df = df_transactions.copy()
    df["txn_month"] = df["txn_timestamp"].dt.to_period("M")
    monthly = df.groupby("txn_month").agg(
        txn_count=("amount", "count"),
        total_amount=("amount", "sum"),
        n_customers=("customer_id", "nunique"),
    ).reset_index()
    monthly["txn_month_str"] = monthly["txn_month"].astype(str)
    monthly = monthly.sort_values("txn_month")
    return monthly


def check_multi_source_agreement(
    df_transactions: pd.DataFrame,
    df_agg: pd.DataFrame,
) -> dict:
    """Reaggregate transactions.db to customer x month and compare to parquet.

    Returns a dict with:
    - total_rows_db: rows in the DB re-aggregation
    - total_rows_pq: rows in the parquet
    - n_left_only: rows only in DB (no parquet match)
    - n_right_only: rows only in parquet (no DB match)
    - n_count_mismatch: matched rows where txn_count differs
    - n_amount_mismatch: matched rows where total_amount differs
    - n_return_mismatch: matched rows where return_count differs
    - mismatched_months: which months have differences
    - db_july_total: July 2026 total amount from DB
    - pq_july_total: July 2026 total amount from parquet
    - july_pct_shortfall: percentage shortfall in parquet vs DB for July 2026
    """
    # Reaggregate DB to customer x month
    df = df_transactions.copy()
    df["txn_month"] = df["txn_timestamp"].dt.to_period("M")
    db_agg = df.groupby(["customer_id", "txn_month"]).agg(
        txn_count_db=("amount", "count"),
        total_amount_db=("amount", "sum"),
        return_count_db=("is_returned", "sum"),
    ).reset_index()
    db_agg["txn_month_str"] = db_agg["txn_month"].astype(str)

    # Parquet prep
    pq = df_agg.copy()
    pq["txn_month"] = pq["txn_month"].dt.to_period("M")
    pq["txn_month_str"] = pq["txn_month"].astype(str)

    # Merge
    # Note: the merge suffixes add _db to columns that exist in both dataframes.
    # Parquet-side columns keep their original names (no collision except txn_month -> txn_month_pq).
    merged = db_agg.merge(
        pq,
        left_on=["customer_id", "txn_month_str"],
        right_on=["customer_id", "txn_month_str"],
        how="outer",
        indicator=True,
        suffixes=("_db", "_pq"),
    )

    left_only = merged[merged["_merge"] == "left_only"]
    right_only = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    # Count mismatches within 'both' rows
    # Columns from parquet side keep original names (txn_count, total_amount, return_count)
    # because the only column collision is txn_month (suffixed _db / _pq).
    count_mismatch = both[both["txn_count_db"] != both["txn_count"]]
    amt_mismatch = both[both["total_amount_db"].sub(both["total_amount"]).abs() > 0.01]
    return_mismatch = both[both["return_count_db"] != both["return_count"]]

    # Identify which months have any disagreement
    mismatched_months = sorted(
        set(count_mismatch["txn_month_str"].unique())
        | set(amt_mismatch["txn_month_str"].unique())
        | set(return_mismatch["txn_month_str"].unique())
        | set(left_only["txn_month_str"].unique())
    )

    # July 2026 specifics
    db_july = db_agg[db_agg["txn_month_str"] == "2026-07"]
    pq_july = pq[pq["txn_month_str"] == "2026-07"]

    db_july_total = float(db_july["total_amount_db"].sum())
    pq_july_total = float(pq_july["total_amount"].sum())
    july_pct_shortfall = round(
        (db_july_total - pq_july_total) / db_july_total * 100, 2
    ) if db_july_total > 0 else 0.0

    return {
        "total_rows_db": len(db_agg),
        "total_rows_pq": len(pq),
        "n_left_only": len(left_only),
        "n_right_only": len(right_only),
        "n_count_mismatch": len(count_mismatch),
        "n_amount_mismatch": len(amt_mismatch),
        "n_return_mismatch": len(return_mismatch),
        "mismatched_months": mismatched_months,
        "db_july_total": db_july_total,
        "pq_july_total": pq_july_total,
        "july_pct_shortfall": july_pct_shortfall,
    }


def monthly_churn_rate_by_signup_cohort(df_customers: pd.DataFrame) -> pd.DataFrame:
    """Compute churn rate by signup month (cohort)."""
    df = df_customers.copy()
    df["signup_month"] = pd.to_datetime(df["signup_date"]).dt.to_period("M")
    cohort = df.groupby("signup_month")["churned"].agg(
        n_customers="count",
        n_churned="sum",
    ).reset_index()
    cohort["churn_rate"] = (cohort["n_churned"] / cohort["n_customers"]).round(4)
    cohort["signup_month_str"] = cohort["signup_month"].astype(str)
    cohort = cohort.sort_values("signup_month")
    return cohort