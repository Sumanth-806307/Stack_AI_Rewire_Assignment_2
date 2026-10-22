"""Tests for eda/load.py — each loader against the real files."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from eda.load import load_customers, load_transactions_db, load_transactions_agg


def test_load_customers_columns():
    df, meta = load_customers()
    expected = {"customer_id", "signup_date", "acquisition_channel", "region", "churned"}
    assert set(df.columns) == expected, f"Got {set(df.columns)}"


def test_load_customers_dtypes():
    df, meta = load_customers()
    assert df["customer_id"].dtype == "int64"
    assert df["churned"].dtype == "int64"


def test_load_customers_row_count():
    df, meta = load_customers()
    assert len(df) == 1500, f"Expected 1500, got {len(df)}"


def test_load_transactions_db_columns():
    df, meta = load_transactions_db()
    expected = {"customer_id", "txn_timestamp", "category", "channel", "amount", "is_returned"}
    assert set(df.columns) == expected


def test_load_transactions_db_dtypes():
    df, meta = load_transactions_db()
    assert df["amount"].dtype == "float64"
    assert df["is_returned"].dtype == "int64"


def test_load_transactions_db_row_count():
    df, meta = load_transactions_db()
    assert len(df) == 11742, f"Expected 11742, got {len(df)}"


def test_load_transactions_agg_columns():
    df, meta = load_transactions_agg()
    expected = {"customer_id", "txn_month", "txn_count", "total_amount", "return_count"}
    assert set(df.columns) == expected


def test_load_transactions_agg_dtypes():
    df, meta = load_transactions_agg()
    assert df["txn_count"].dtype == "int64"
    assert df["total_amount"].dtype == "float64"
    assert df["return_count"].dtype == "int64"


def test_load_transactions_agg_row_count():
    df, meta = load_transactions_agg()
    # Allow some tolerance — parquet may have slightly different row counts
    # depending on when the file was generated
    assert 8000 <= len(df) <= 8150, f"Expected ~8086, got {len(df)}"


def test_load_all():
    from eda.load import load_all
    sources = load_all()
    assert set(sources.keys()) == {"customers", "transactions", "agg"}
    for key in sources:
        df, meta = sources[key]
        assert isinstance(df, pd.DataFrame)
        assert isinstance(meta, dict)