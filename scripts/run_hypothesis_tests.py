#!/usr/bin/env python3
"""Run all hypothesis tests and print a results table."""

import sys
import os


def main() -> None:
    """Load data, run hypothesis tests, and persist results in multiple formats."""
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from eda.load import load_customers, load_transactions_db
    except ImportError as e:
        print(f"ERROR: Failed to import EDA module — {e}", file=sys.stderr)
        print("HINT: Ensure you are running from the project root or that dependencies", file=sys.stderr)
        print("      (pandas, numpy, scipy) are installed in your environment.", file=sys.stderr)
        sys.exit(1)

    try:
        from eda.hypotheses import run_all_hypothesis_tests
    except ImportError as e:
        print(f"ERROR: Failed to import hypothesis-test module — {e}", file=sys.stderr)
        print("HINT: Verify eda/hypotheses.py exists and has no syntax errors.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    try:
        df_cust, cust_meta = load_customers()
        print(f"Loaded customers: {cust_meta['rows']} rows from {cust_meta['source']}")
    except FileNotFoundError as e:
        print(f"ERROR: Customer data file not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/customers.csv exists relative to the project root.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load customer data — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_tx, tx_meta = load_transactions_db()
        print(f"Loaded transactions: {tx_meta['rows']} rows from {tx_meta['source']}")
    except FileNotFoundError as e:
        print(f"ERROR: Transaction database not found — {e}", file=sys.stderr)
        print("HINT: Ensure data/transactions.db exists relative to the project root.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load transaction data — {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run hypothesis tests
    # ------------------------------------------------------------------
    try:
        results = run_all_hypothesis_tests(df_cust, df_tx)
    except KeyError as e:
        print(f"ERROR: Missing expected column in data — {e}", file=sys.stderr)
        print("HINT: Confirm the data files have the required schema.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Hypothesis test execution failed — {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("WARNING: No hypothesis test results were returned. Nothing to display.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Print formatted table
    # ------------------------------------------------------------------
    print("=" * 100)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 100)
    print(f"{'Driver':<25s} {'Test':<20s} {'Statistic':<12s} {'p-value':<10s} {'Signif?':<8s} {'Effect':<10s} {'Direction'}")
    print("-" * 100)
    for r in results:
        sig = "YES *" if r["significant"] else "no"
        print(f"{r['label']:<25s} {r['test']:<20s} {r['statistic']:<12.4f} {r['p_value']:<10.6f} {sig:<8s} {r['effect_size']:<10.4f} {r['direction']}")

    print()
    sig_count = sum(1 for r in results if r["significant"])
    total = len(results)
    print(f"Significant at α=0.05: {sig_count}/{total}")
    print("Non-significant: ", [r["driver"] for r in results if not r["significant"]])

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    def numpy_converter(obj):
        """Convert NumPy types to standard Python types for JSON serialization."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    import json
    import csv

    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError as e:
        print(f"ERROR: Cannot create output directory '{output_dir}' — {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: File-system error creating output directory — {e}", file=sys.stderr)
        sys.exit(1)

    # 1) JSON — machine-readable
    json_path = os.path.join(output_dir, "hypothesis_results.json")
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=numpy_converter)
        print(f"\nJSON → {json_path}")
    except (TypeError, ValueError) as e:
        print(f"ERROR: Failed to serialize results to JSON — {e}", file=sys.stderr)
    except OSError as e:
        print(f"ERROR: Failed to write JSON file — {e}", file=sys.stderr)

    # 2) CSV — spreadsheet-friendly
    csv_path = os.path.join(output_dir, "hypothesis_results.csv")
    try:
        fieldnames = ["driver", "label", "test", "statistic", "p_value", "significant", "effect_size", "direction"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV  → {csv_path}")
    except OSError as e:
        print(f"ERROR: Failed to write CSV file — {e}", file=sys.stderr)

    # 3) Markdown table — readable for reports
    md_path = os.path.join(output_dir, "hypothesis_results.md")
    try:
        with open(md_path, "w") as f:
            f.write("# Hypothesis Test Results\n\n")
            f.write(f"**Significant at α=0.05:** {sig_count}/{total}\n\n")
            f.write("| Driver | Test | Statistic | p-value | Significant? | Effect Size | Direction |\n")
            f.write("|--------|------|-----------|---------|--------------|-------------|-----------|\n")
            for r in results:
                sig = "Yes" if r["significant"] else "No"
                f.write(f"| {r['label']} | {r['test']} | {r['statistic']} | {r['p_value']} | {sig} | {r['effect_size']} | {r['direction']} |\n")
            f.write("\n### Non-significant drivers\n")
            ns = [r["driver"] for r in results if not r["significant"]]
            for d in ns:
                f.write(f"- {d}\n")
        print(f"MD   → {md_path}")
    except OSError as e:
        print(f"ERROR: Failed to write Markdown file — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()