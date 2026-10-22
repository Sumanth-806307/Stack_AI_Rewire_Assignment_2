#!/usr/bin/env python3
"""Run profiling across all three data sources."""

import sys
import os


def main() -> None:
    """Load all data sources, compute profiling summaries, and print results."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from eda.load import load_all
    except ImportError as e:
        print(f"ERROR: Failed to import EDA module — {e}", file=sys.stderr)
        print("HINT: Ensure you are running from the project root or that dependencies", file=sys.stderr)
        print("      (pandas, numpy) are installed in your environment.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.profile import (
            schema_summary,
            distribution_stats,
            outlier_count_iqr,
            count_zero_activity_customers,
            churn_rate_by_group,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import profiling module — {e}", file=sys.stderr)
        print("HINT: Verify eda/profile.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load all data sources
    # ------------------------------------------------------------------
    try:
        sources = load_all()
    except FileNotFoundError as e:
        print(f"ERROR: A required data file was not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/customers.csv, data/transactions.db, and", file=sys.stderr)
        print("      data/transactions_monthly_agg.parquet exist.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load data sources — {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Profile customers
    # ------------------------------------------------------------------
    try:
        df_cust = sources["customers"][0]
    except KeyError:
        print("ERROR: 'customers' key missing from loaded sources.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("PROFILE: CUSTOMERS (customers.csv)")
    print("=" * 70)

    try:
        schema = schema_summary(df_cust, "customers.csv")
        print(f"Rows: {schema['rows']}")
        for c in schema["columns"]:
            print(f"  {c['column']:25s}  {c['dtype']:12s}  missing={c['n_missing']} ({c['pct_missing']}%)")
        print()
    except Exception as e:
        print(f"ERROR: Failed to compute customer schema summary — {e}", file=sys.stderr)

    # Numeric distributions
    for col in ["customer_id", "churned"]:
        if col in df_cust.columns:
            try:
                s = df_cust[col]
                print(f"Distribution: {col}")
                print(f"  {distribution_stats(s)}")
                if col == "churned":
                    print(f"  Churn count: {s.sum()}, rate: {s.mean():.4f}")
                print()
            except Exception as e:
                print(f"ERROR: Failed to compute distribution for '{col}' — {e}", file=sys.stderr)

    # Categorical distributions
    for col in ["acquisition_channel", "region"]:
        if col in df_cust.columns:
            try:
                print(f"Value counts: {col}")
                print(df_cust[col].value_counts().to_string())
                print()
            except Exception as e:
                print(f"ERROR: Failed to compute value counts for '{col}' — {e}", file=sys.stderr)

    # Churn by groups
    for col in ["acquisition_channel", "region"]:
        try:
            result = churn_rate_by_group(df_cust, col)
            print(f"Churn by {col}:")
            print(result.to_string(index=False))
            print()
        except Exception as e:
            print(f"ERROR: Failed to compute churn rate for '{col}' — {e}", file=sys.stderr)

    # Zero-activity
    try:
        df_tx = sources["transactions"][0]
    except KeyError:
        print("ERROR: 'transactions' key missing from loaded sources.", file=sys.stderr)
        sys.exit(1)

    try:
        za = count_zero_activity_customers(df_cust, df_tx)
        print(f"Zero-activity customers: {za['n_zero_activity']} ({za['pct_of_customers']}%)")
        print(f"  Churned: {za['n_churned']} ({za['pct_churned']}% of zero-activity)")
        print(f"  {za['pct_of_all_churned']}% of all churned customers")
        print()
    except KeyError as e:
        print(f"ERROR: Missing expected key in zero-activity result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Zero-activity analysis failed — {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Profile transactions (DB)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PROFILE: TRANSACTIONS (transactions.db)")
    print("=" * 70)

    try:
        schema_tx = schema_summary(df_tx, "transactions.db")
        print(f"Rows: {schema_tx['rows']}")
        for c in schema_tx["columns"]:
            print(f"  {c['column']:20s}  {c['dtype']:12s}  missing={c['n_missing']} ({c['pct_missing']}%)")
        print()
    except Exception as e:
        print(f"ERROR: Failed to compute transaction schema summary — {e}", file=sys.stderr)

    # Transaction amount distribution
    try:
        print("Distribution: amount")
        print(f"  {distribution_stats(df_tx['amount'])}")
        print(f"  Outliers (IQR): {outlier_count_iqr(df_tx['amount'])}")
        print()
    except KeyError:
        print("ERROR: Column 'amount' not found in transactions data.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Amount distribution analysis failed — {e}", file=sys.stderr)

    # Category & channel distributions
    for col in ["category", "channel"]:
        if col in df_tx.columns:
            try:
                print(f"Distribution: {col}")
                print(df_tx[col].value_counts().to_string())
                print()
            except Exception as e:
                print(f"ERROR: Failed to compute value counts for '{col}' — {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Profile transactions agg (parquet)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PROFILE: TRANSACTIONS AGG (parquet)")
    print("=" * 70)

    try:
        df_pq = sources["agg"][0]
    except KeyError:
        print("ERROR: 'agg' key missing from loaded sources.", file=sys.stderr)
        sys.exit(1)

    try:
        schema_pq = schema_summary(df_pq, "transactions_monthly_agg.parquet")
        print(f"Rows: {schema_pq['rows']}")
        for c in schema_pq["columns"]:
            print(f"  {c['column']:20s}  {c['dtype']:12s}  missing={c['n_missing']} ({c['pct_missing']}%)")
        print()
    except Exception as e:
        print(f"ERROR: Failed to compute agg schema summary — {e}", file=sys.stderr)

    for col in ["txn_count", "total_amount", "return_count"]:
        if col in df_pq.columns:
            try:
                print(f"Distribution: {col}")
                print(f"  {distribution_stats(df_pq[col])}")
                print(f"  Outliers (IQR): {outlier_count_iqr(df_pq[col])}")
                print()
            except Exception as e:
                print(f"ERROR: Failed to compute distribution for '{col}' — {e}", file=sys.stderr)

    print("Profile complete.")


if __name__ == "__main__":
    main()