#!/usr/bin/env python3
"""Run time-trend, multi-source agreement, and leakage checks."""

import sys
import os


def main() -> None:
    """Load all data sources and run time-trend, multi-source, and leakage checks."""
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from eda.load import load_customers, load_transactions_db, load_transactions_agg
    except ImportError as e:
        print(f"ERROR: Failed to import EDA module — {e}", file=sys.stderr)
        print("HINT: Ensure you are running from the project root or that dependencies", file=sys.stderr)
        print("      (pandas, numpy) are installed in your environment.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.time_checks import (
            monthly_tx_volume,
            check_multi_source_agreement,
            monthly_churn_rate_by_signup_cohort,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import time_checks module — {e}", file=sys.stderr)
        print("HINT: Verify eda/time_checks.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.leakage_checks import (
            check_days_since_last_txn_separation,
            check_as_of_date_discipline,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import leakage_checks module — {e}", file=sys.stderr)
        print("HINT: Verify eda/leakage_checks.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    try:
        df_cust, _ = load_customers()
        print("Loaded customers.csv")
    except FileNotFoundError as e:
        print(f"ERROR: Customer data file not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/customers.csv exists relative to the project root.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load customer data — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_tx, _ = load_transactions_db()
        print("Loaded transactions.db")
    except FileNotFoundError as e:
        print(f"ERROR: Transaction database not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/transactions.db exists relative to the project root.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load transaction data — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_pq, _ = load_transactions_agg()
        print("Loaded transactions_monthly_agg.parquet")
    except FileNotFoundError as e:
        print(f"ERROR: Parquet aggregate file not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/transactions_monthly_agg.parquet exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load parquet aggregate data — {e}", file=sys.stderr)
        sys.exit(1)

    print()

    # ------------------------------------------------------------------
    # TIME TREND: MONTHLY TX VOLUME
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TIME TREND: MONTHLY TX VOLUME")
    print("=" * 70)

    try:
        volume = monthly_tx_volume(df_tx)
    except KeyError as e:
        print(f"ERROR: Missing expected column for monthly volume — {e}", file=sys.stderr)
        print("HINT: Confirm transactions data has 'txn_timestamp' and 'amount' columns.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Monthly volume computation failed — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Date range: {volume['txn_month'].min()} to {volume['txn_month'].max()}")
        print("\nFirst 6 months:")
        print(volume.head(6).to_string(index=False))
        print("\nLast 6 months:")
        print(volume.tail(6).to_string(index=False))
    except KeyError as e:
        print(f"ERROR: Unexpected volume result format — missing column {e}", file=sys.stderr)

    # Check for growth trend
    try:
        first_half = volume.head(len(volume) // 2)["txn_count"].sum()
        second_half = volume.tail(len(volume) // 2)["txn_count"].sum()
        print(f"\nFirst-half txns: {first_half}")
        print(f"Second-half txns: {second_half}")
        if first_half == 0:
            print("WARNING: First-half txn count is zero; growth ratio is undefined.")
        else:
            print(f"Growth ratio: {second_half / first_half:.2f}x")
    except Exception as e:
        print(f"ERROR: Growth-trend computation failed — {e}", file=sys.stderr)

    print()

    # ------------------------------------------------------------------
    # TIME TREND: CHURN RATE BY SIGNUP COHORT
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TIME TREND: CHURN RATE BY SIGNUP COHORT")
    print("=" * 70)

    try:
        cohort = monthly_churn_rate_by_signup_cohort(df_cust)
        print("\nAll cohorts (first 10 and last 10):")
        print(cohort.head(10).to_string(index=False))
        print("...")
        print(cohort.tail(10).to_string(index=False))
    except KeyError as e:
        print(f"ERROR: Missing expected column for cohort analysis — {e}", file=sys.stderr)
        print("HINT: Confirm customers data has 'signup_date' and 'churned' columns.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Cohort analysis failed — {e}", file=sys.stderr)

    # Summarize by year
    try:
        df_cust["signup_year"] = pd.to_datetime(df_cust["signup_date"]).dt.year
        yearly = df_cust.groupby("signup_year")["churned"].agg(n="count", churned="sum")
        yearly["rate"] = (yearly["churned"] / yearly["n"] * 100).round(1)
        print("\nBy signup year:")
        print(yearly.to_string())
    except KeyError as e:
        print(f"ERROR: Missing required column for yearly summary — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Yearly summary computation failed — {e}", file=sys.stderr)

    print()

    # ------------------------------------------------------------------
    # MULTI-SOURCE AGREEMENT
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MULTI-SOURCE AGREEMENT (DB vs PARQUET)")
    print("=" * 70)

    try:
        msa = check_multi_source_agreement(df_tx, df_pq)
    except Exception as e:
        print(f"ERROR: Multi-source agreement check failed — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"DB re-aggregated rows: {msa['total_rows_db']}")
        print(f"Parquet rows: {msa['total_rows_pq']}")
        print(f"Rows only in DB (left_only): {msa['n_left_only']}")
        print(f"Rows only in parquet (right_only): {msa['n_right_only']}")
        print(f"Matched rows with count mismatch: {msa['n_count_mismatch']}")
        print(f"Matched rows with amount mismatch: {msa['n_amount_mismatch']}")
        print(f"Matched rows with return mismatch: {msa['n_return_mismatch']}")
        print(f"Mismatched months: {msa['mismatched_months']}")
        print(f"July 2026 DB total: ${msa['db_july_total']:.2f}")
        print(f"July 2026 Parquet total: ${msa['pq_july_total']:.2f}")
        print(f"July 2026 shortfall: {msa['july_pct_shortfall']}%")
    except KeyError as e:
        print(f"ERROR: Missing expected key in multi-source agreement result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Displaying multi-source agreement results failed — {e}", file=sys.stderr)

    print()

    # ------------------------------------------------------------------
    # LEAKAGE: DAYS-SINCE-LAST-TXN
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LEAKAGE: DAYS-SINCE-LAST-TXN SEPARATION")
    print("=" * 70)

    try:
        leak = check_days_since_last_txn_separation(df_cust, df_tx)
        print(f"Churned min days_since_last_txn: {leak['churned_min_days_since_last_txn']}")
        print(f"Non-churned max days_since_last_txn: {leak['non_churned_max_days_since_last_txn']}")
        print(f"Separation gap: {leak['separation_gap']}")
        print(f"Perfect separation: {leak['perfect_separation']}")
        print(f"Correlation with churn: {leak['correlation_with_churn']}")
        print(f"Zero-activity & churned: {leak['n_zero_activity_and_churned']}")
    except KeyError as e:
        print(f"ERROR: Missing expected key in leakage result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Leakage check (days-since-last-txn) failed — {e}", file=sys.stderr)

    print()

    # ------------------------------------------------------------------
    # LEAKAGE: AS-OF-DATE
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LEAKAGE: AS-OF-DATE DISCIPLINE")
    print("=" * 70)

    try:
        aod = check_as_of_date_discipline(df_cust, df_tx)
        print(f"As-of date: {aod['as_of_date']}")
        print(f"Max txn timestamp: {aod['max_txn_timestamp']}")
        print(f"Txns after as-of: {aod['n_transactions_after_cutoff']}")
        print(f"Check passes: {aod['passes_check']}")
    except KeyError as e:
        print(f"ERROR: Missing expected key in as-of-date result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: As-of-date discipline check failed — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()