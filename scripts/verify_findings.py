#!/usr/bin/env python3
"""Verify every numeric claim in EDA_FINDINGS.md reproduces from raw data.

Reads the markdown file to extract expected values, re-computes them from raw
data, and reports any mismatches.

Exit code 0 = all OK. Exit code 1 = at least one claim doesn't reproduce
or an error occurred.
"""

import sys
import os
import re


def parse_expected_from_md(path: str) -> dict:
    """Parse EDA_FINDINGS.md and return a flat dict of label → expected value.

    Scans for numeric patterns that appear after known section markers.
    """
    with open(path, "r") as f:
        text = f.read()

    expected: dict = {}

    # Helper: match a number (int or float, with optional commas)
    def num(s: str) -> float:
        return float(s.replace(",", ""))

    # -- Row counts (table rows like "| customers.csv | CSV | 1,500 | ...") --
    m = re.search(r"\|\s*`customers\.csv`\s*\|.*?\|\s*([\d,]+)\s*\|", text)
    if m:
        expected["Total customers (csv)"] = num(m.group(1))
    m = re.search(r"\|\s*`transactions\.db`\s*\|.*?\|\s*([\d,]+)\s*\|", text)
    if m:
        expected["Total transactions (db)"] = num(m.group(1))
    m = re.search(
        r"\|\s*`transactions_monthly_agg\.parquet`\s*\|.*?\|\s*([\d,]+)\s*\|", text
    )
    if m:
        expected["Total agg rows (parquet)"] = num(m.group(1))

    # -- Churn rate & imbalance --
    # "13.5%" ... "202 churned / 1,500 total"
    m = re.search(r"Overall churn rate:\s*([\d.]+)%", text)
    if m:
        expected["Overall churn rate (pct)"] = float(m.group(1))
    m = re.search(r"(\d+)\s+churned\s*/\s*[\d,]+\s+total", text)
    if m:
        expected["Churned count"] = int(m.group(1))
    m = re.search(r"Non-churned:\s*([\d,]+)\s*\(", text)
    if m:
        expected["Not churned count"] = num(m.group(1))

    # -- Zero-activity --
    # "**81 customers (5.4%**"
    m = re.search(r"\*\*(\d+)\s+customers\s*\(([\d.]+)%\)", text)
    if m:
        expected["Zero-activity count"] = int(m.group(1))
        expected["Zero-activity % of customers"] = float(m.group(2))
    # "**All 81 (100%) are labeled churned**"
    m = re.search(r"All\s+(\d+)\s+\(([\d.]+)%\)\s+are\s+labeled\s+churned", text)
    if m:
        expected["Zero-activity churned count"] = int(m.group(1))
        expected["Zero-activity churn rate"] = float(m.group(2))
    # "representing **40.1%** of all churned customers"
    m = re.search(r"representing\s+\*\*([\d.]+)%\*\*\s+of\s+all\s+churned", text)
    if m:
        expected["Zero-activity % of all churned"] = float(m.group(1))

    # -- Channel churn rates from table --
    # "| organic | 609 | 93 | 15.3% |"
    for channel in ["organic", "social", "referral", "paid_search"]:
        pat = rf"\|\s*{channel}\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)%\s*\|"
        m = re.search(pat, text)
        if m:
            expected[f"{channel} churn rate (pct)"] = float(m.group(1))

    # -- Region churn rates from table --
    for region in ["east", "south", "west", "north"]:
        pat = rf"\|\s*{region}\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)%\s*\|"
        m = re.search(pat, text)
        if m:
            expected[f"{region} churn rate (pct)"] = float(m.group(1))

    # -- Signup-year churn rates from table --
    # "| 2024 | 570 | 52 | **9.1%** |"
    for year in ["2024", "2025", "2026"]:
        pat = rf"\|\s*{year}\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\*?\*?([\d.]+)%\*?\*?\s*\|"
        m = re.search(pat, text)
        if m:
            expected[f"{year} churn rate (pct)"] = float(m.group(1))

    # -- Transaction amount distribution --
    # "| Mean | $49.72 |"
    m = re.search(r"\|\s*Mean\s*\|\s*\$?([\d,]+\.?\d*)", text)
    if m:
        expected["Tx amount mean"] = float(m.group(1))
    m = re.search(r"\|\s*Median\s*\|\s*\$?([\d,]+\.?\d*)", text)
    if m:
        expected["Tx amount median"] = float(m.group(1))
    m = re.search(r"\|\s*Max\s*\|\s*\$?([\d,]+\.?\d*)", text)
    if m:
        expected["Tx amount max"] = num(m.group(1))
    # "| Skew | **Right**"
    if re.search(r"\|\s*Skew\s*\|\s*\*?\*?Right\*?\*?", text):
        expected["Tx amount skew direction"] = "right"
    # "| IQR outliers (high side) | 783 (6.67%) |"
    m = re.search(
        r"\|\s*IQR outliers.*?\|\s*([\d,]+)\s*\(([\d.]+)%\)", text
    )
    if m:
        expected["Tx amount IQR outlier count"] = num(m.group(1))

    # -- Multi-source agreement --
    # "| Rows only in DB (left_only) | **102**"
    m = re.search(
        r"\|\s*Rows only in DB \(left_only\)\s*\|\s*\*?\*?(\d+)\*?\*?", text
    )
    if m:
        expected["MSA left_only rows"] = int(m.group(1))
    # "| Rows only in parquet (right_only) | 0 |"
    m = re.search(
        r"\|\s*Rows only in parquet \(right_only\)\s*\|\s*(\d+)", text
    )
    if m:
        expected["MSA right_only rows"] = int(m.group(1))
    # "| Matched rows with count mismatch | **110** |"
    m = re.search(
        r"\|\s*Matched rows with count mismatch\s*\|\s*\*?\*?(\d+)\*?\*?", text
    )
    if m:
        expected["MSA count mismatches"] = int(m.group(1))
    # "| Mismatched months | **Only July 2026** |"
    if re.search(r"Mismatched months.*?July 2026", text):
        expected["MSA all mismatches in July 2026"] = True
    # "| July 2026 shortfall | **19.28%** |"
    m = re.search(
        r"\|\s*July 2026 shortfall\s*\|\s*\*?\*?([\d.]+)%\*?\*?", text
    )
    if m:
        expected["MSA July shortfall %"] = float(m.group(1))

    # -- Leakage --
    # "days_since_last_txn ≥ 90"
    m = re.search(r"days_since_last_txn\s*≥\s*(\d+)", text)
    if m:
        expected["Leakage churned min DSL"] = int(m.group(1))
    # "≤ 89"
    m = re.search(r"≤\s*(\d+)", text)
    if m:
        expected["Leakage non-churned max DSL"] = int(m.group(1))

    return expected


def main() -> None:
    """Load raw data, parse expected values from EDA_FINDINGS.md, compare."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, ".."))

    import pandas as pd
    import numpy as np

    # ------------------------------------------------------------------
    # Imports with helpful error messages
    # ------------------------------------------------------------------
    try:
        from eda.load import load_customers, load_transactions_db, load_transactions_agg
    except ImportError as e:
        print(f"ERROR: Failed to import EDA module — {e}", file=sys.stderr)
        print("HINT: Ensure you are running from the project root or that dependencies", file=sys.stderr)
        print("      (pandas, numpy, scipy) are installed.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.profile import (
            distribution_stats,
            outlier_count_iqr,
            count_zero_activity_customers,
            churn_rate_by_group,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import profiling module — {e}", file=sys.stderr)
        print("HINT: Verify eda/profile.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.hypotheses import run_all_hypothesis_tests
    except ImportError as e:
        print(f"ERROR: Failed to import hypothesis-test module — {e}", file=sys.stderr)
        print("HINT: Verify eda/hypotheses.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.time_checks import check_multi_source_agreement
    except ImportError as e:
        print(f"ERROR: Failed to import time_checks module — {e}", file=sys.stderr)
        print("HINT: Verify eda/time_checks.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.leakage_checks import check_days_since_last_txn_separation
    except ImportError as e:
        print(f"ERROR: Failed to import leakage_checks module — {e}", file=sys.stderr)
        print("HINT: Verify eda/leakage_checks.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    try:
        df_cust, _ = load_customers()
        print(f"Loaded {len(df_cust)} customers.")
    except FileNotFoundError as e:
        print(f"ERROR: Customer data file not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/customers.csv exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load customer data — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_tx, _ = load_transactions_db()
        print(f"Loaded {len(df_tx)} transactions.")
    except FileNotFoundError as e:
        print(f"ERROR: Transaction database not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/transactions.db exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load transaction data — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_pq, _ = load_transactions_agg()
        print(f"Loaded {len(df_pq)} agg rows.")
    except FileNotFoundError as e:
        print(f"ERROR: Parquet aggregate file not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/transactions_monthly_agg.parquet exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load parquet aggregate data — {e}", file=sys.stderr)
        sys.exit(1)

    print()

    # ------------------------------------------------------------------
    # Parse expected values from EDA_FINDINGS.md
    # ------------------------------------------------------------------
    md_path = os.path.join(script_dir, "..", "EDA_FINDINGS.md")
    try:
        expected = parse_expected_from_md(md_path)
    except FileNotFoundError:
        print(f"ERROR: EDA_FINDINGS.md not found at {md_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to parse EDA_FINDINGS.md — {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(expected)} expected values from EDA_FINDINGS.md.")
    print()

    # ------------------------------------------------------------------
    # Compute values from raw data and compare
    # ------------------------------------------------------------------
    errors: list[str] = []

    def check(label: str, computed, expected_val, tolerance: float = 0.02):
        """Assert computed ≈ expected_val, or record an error."""
        try:
            if isinstance(expected_val, (int, float, np.floating, np.integer)):
                if abs(float(computed) - float(expected_val)) > tolerance:
                    errors.append(f"{label}: expected {computed}, got {expected_val}")
            elif isinstance(expected_val, str):
                if str(computed).lower() != expected_val.lower():
                    errors.append(f"{label}: expected '{computed}', got '{expected_val}'")
            elif expected_val is True or expected_val is False:
                if bool(computed) != expected_val:
                    errors.append(f"{label}: expected {computed}, got {expected_val}")
            else:
                if computed != expected_val:
                    errors.append(f"{label}: expected {computed}, got {expected_val}")
        except (TypeError, ValueError) as e:
            errors.append(f"{label}: comparison failed — {e} (computed={computed!r}, expected={expected_val!r})")

    # --- 1. Data Overview ---
    if "Total customers (csv)" in expected:
        check("Total customers (csv)", len(df_cust), expected["Total customers (csv)"])
    if "Total transactions (db)" in expected:
        check("Total transactions (db)", len(df_tx), expected["Total transactions (db)"])
    if "Total agg rows (parquet)" in expected:
        check("Total agg rows (parquet)", len(df_pq), expected["Total agg rows (parquet)"])

    # --- 2. Churn rate & imbalance ---
    try:
        churn_rate = df_cust["churned"].mean()
        churned_sum = df_cust["churned"].sum()
        not_churned_sum = (1 - df_cust["churned"]).sum()
        if "Overall churn rate (pct)" in expected:
            check("Overall churn rate", churn_rate * 100, expected["Overall churn rate (pct)"], tolerance=0.1)
        if "Churned count" in expected:
            check("Churned count", int(churned_sum), expected["Churned count"])
        if "Not churned count" in expected:
            check("Not churned count", int(not_churned_sum), expected["Not churned count"])
    except KeyError:
        print("ERROR: Column 'churned' not found in customers data.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Churn rate computation failed — {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Zero-activity customers ---
    try:
        za = count_zero_activity_customers(df_cust, df_tx)
        if "Zero-activity count" in expected:
            check("Zero-activity count", za["n_zero_activity"], expected["Zero-activity count"])
        if "Zero-activity % of customers" in expected:
            check("Zero-activity % of customers", za["pct_of_customers"], expected["Zero-activity % of customers"], tolerance=0.1)
        if "Zero-activity churned count" in expected:
            check("Zero-activity churned count", za["n_churned"], expected["Zero-activity churned count"])
        if "Zero-activity churn rate" in expected:
            check("Zero-activity churn rate", za["pct_churned"], expected["Zero-activity churn rate"], tolerance=0.1)
        if "Zero-activity % of all churned" in expected:
            check("Zero-activity % of all churned", za["pct_of_all_churned"], expected["Zero-activity % of all churned"], tolerance=0.5)
    except KeyError as e:
        print(f"ERROR: Missing expected key in zero-activity result — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Zero-activity computation failed — {e}", file=sys.stderr)
        sys.exit(1)

    # --- 4. Churn by acquisition channel ---
    try:
        chan = churn_rate_by_group(df_cust, "acquisition_channel")
        chan_dict = dict(zip(chan["acquisition_channel"], chan["churn_rate"]))
        for channel in ["organic", "social", "referral", "paid_search"]:
            key = f"{channel} churn rate (pct)"
            if key in expected:
                computed_pct = chan_dict.get(channel, 0) * 100
                check(f"{channel} churn rate", computed_pct, expected[key], tolerance=0.1)
    except Exception as e:
        print(f"ERROR: Churn-by-channel computation failed — {e}", file=sys.stderr)

    # --- 5. Churn by region ---
    try:
        reg = churn_rate_by_group(df_cust, "region")
        reg_dict = dict(zip(reg["region"], reg["churn_rate"]))
        for region in ["east", "south", "west", "north"]:
            key = f"{region} churn rate (pct)"
            if key in expected:
                computed_pct = reg_dict.get(region, 0) * 100
                check(f"{region} churn rate", computed_pct, expected[key], tolerance=0.1)
    except Exception as e:
        print(f"ERROR: Churn-by-region computation failed — {e}", file=sys.stderr)

    # --- 6. Churn by signup year ---
    try:
        df_cust_cp = df_cust.copy()
        df_cust_cp["signup_year"] = pd.to_datetime(df_cust_cp["signup_date"]).dt.year
        yearly = df_cust_cp.groupby("signup_year")["churned"].agg(["count", "sum"])
        yearly["rate"] = yearly["sum"] / yearly["count"]
        for year_str in ["2024", "2025", "2026"]:
            key = f"{year_str} churn rate (pct)"
            if key in expected and int(year_str) in yearly.index:
                computed_pct = yearly.loc[int(year_str), "rate"] * 100
                check(f"{year_str} churn rate", computed_pct, expected[key], tolerance=0.1)
    except KeyError as e:
        print(f"ERROR: Missing column or year in signup-year analysis — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Signup-year analysis failed — {e}", file=sys.stderr)

    # --- 7. Transaction amount distribution ---
    try:
        amt_dist = distribution_stats(df_tx["amount"])
        if "Tx amount mean" in expected:
            check("Tx amount mean", amt_dist["mean"], expected["Tx amount mean"], tolerance=0.1)
        if "Tx amount median" in expected:
            check("Tx amount median", amt_dist["median"], expected["Tx amount median"], tolerance=0.1)
        if "Tx amount max" in expected:
            check("Tx amount max", amt_dist["max"], expected["Tx amount max"], tolerance=0.1)
        if "Tx amount skew direction" in expected:
            check("Tx amount skew direction", amt_dist["skew"], expected["Tx amount skew direction"])
    except KeyError:
        print("ERROR: Column 'amount' not found in transactions data.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Amount distribution analysis failed — {e}", file=sys.stderr)

    try:
        amt_out = outlier_count_iqr(df_tx["amount"])
        if "Tx amount IQR outlier count" in expected:
            check("Tx amount IQR outlier count", amt_out["n_outliers_total"], expected["Tx amount IQR outlier count"], tolerance=1)
    except KeyError as e:
        print(f"ERROR: Missing expected key in outlier result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Outlier detection failed — {e}", file=sys.stderr)

    # --- 8. Multi-source agreement ---
    try:
        msa = check_multi_source_agreement(df_tx, df_pq)
        if "MSA left_only rows" in expected:
            check("MSA left_only rows", msa["n_left_only"], expected["MSA left_only rows"])
        if "MSA right_only rows" in expected:
            check("MSA right_only rows", msa["n_right_only"], expected["MSA right_only rows"])
        if "MSA count mismatches" in expected:
            check("MSA count mismatches", msa["n_count_mismatch"], expected["MSA count mismatches"], tolerance=1)
        if "MSA all mismatches in July 2026" in expected:
            passes = msa["mismatched_months"] == ["2026-07"]
            check("MSA all mismatches in July 2026", passes, expected["MSA all mismatches in July 2026"])
        if "MSA July shortfall %" in expected:
            check("MSA July shortfall %", msa["july_pct_shortfall"], expected["MSA July shortfall %"], tolerance=0.5)
    except KeyError as e:
        print(f"ERROR: Missing expected key in multi-source agreement result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Multi-source agreement verification failed — {e}", file=sys.stderr)

    # --- 9. Leakage: days-since-last-txn ---
    try:
        leak = check_days_since_last_txn_separation(df_cust, df_tx)
        if "Leakage churned min DSL" in expected:
            check("Leakage churned min DSL", leak["churned_min_days_since_last_txn"], expected["Leakage churned min DSL"])
        if "Leakage non-churned max DSL" in expected:
            check("Leakage non-churned max DSL", leak["non_churned_max_days_since_last_txn"], expected["Leakage non-churned max DSL"])
    except KeyError as e:
        print(f"ERROR: Missing expected key in leakage result — {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Leakage verification failed — {e}", file=sys.stderr)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if errors:
        print(f"\nFAIL: {len(errors)} verification error(s) — EDA_FINDINGS.md is out of sync with raw data:")
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)
    else:
        print("✅ ALL CLAIMS VERIFIED — every number in EDA_FINDINGS.md reproduces from raw data.")
        sys.exit(0)


if __name__ == "__main__":
    main()