"""Loaders for all three raw data sources.

One loader function per source — pure I/O, no analysis logic.
Each returns a tuple of (data, metadata) where metadata captures
row count, column names, and dtypes for downstream use.
"""

import pandas as pd
import sqlite3
from typing import Tuple, Any


def load_customers(path: str = "data/customers.csv") -> Tuple[pd.DataFrame, dict]:
    """Load the customers CSV. Returns (DataFrame, metadata dict)."""
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    meta = {
        "source": path,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    return df, meta


def load_transactions_db(path: str = "data/transactions.db") -> Tuple[pd.DataFrame, dict]:
    """Load transactions from SQLite. Returns (DataFrame, metadata dict)."""
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    meta = {
        "source": path,
        "table": "transactions",
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    return df, meta


def load_transactions_agg(path: str = "data/transactions_monthly_agg.parquet") -> Tuple[pd.DataFrame, dict]:
    """Load pre-aggregated monthly parquet. Returns (DataFrame, metadata dict)."""
    df = pd.read_parquet(path)
    meta = {
        "source": path,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    return df, meta


def load_all() -> dict:
    """Convenience: load all three sources into a dict keyed by short name."""
    return {
        "customers": load_customers(),
        "transactions": load_transactions_db(),
        "agg": load_transactions_agg(),
    }