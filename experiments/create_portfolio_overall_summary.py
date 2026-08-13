"""
Create an overall summary combining results from all portfolio experiments.

This script combines portfolio experiment results from detailed result files:
- portfolio_yahoo_test.csv (4 rows, K=3 implicitly)
- portfolio_yahoo_20stocks.csv (2 rows, K=5 implicitly)
- portfolio_orlibrary_full_results.csv (OR-Library full-instance results)

For each experiment setting (dataset + K), creates 5 categories:
1. No-Sparsity Baseline (method == "no_sparsity")
2. Gurobi Cardinality (method == "cardinality")
3. Genetic Algorithm (method == "genetic_algorithm")
4. Simulated Annealing (method == "simulated_annealing")
5. Best Sparse (lowest variance among cardinality, genetic_algorithm, simulated_annealing)

Output: A comprehensive comparison table across all portfolio experiments.
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def safe_get(row, col_name, default=None):
    """Safely get a value from a pandas Series."""
    try:
        if col_name in row.index:
            val = row[col_name]
            if pd.isna(val):
                return default
            return val
        return default
    except:
        return default


def process_orlibrary_results(results_dir):
    """
    Process OR-Library results: group by dataset and K, extract 5 categories each.
    
    Returns: List of dicts (one per category per setting)
    """
    summary_rows = []
    preferred_path = os.path.join(results_dir, "portfolio_orlibrary_full_results.csv")
    legacy_path = os.path.join(results_dir, "portfolio_orlibrary_results.csv")
    orlibrary_path = preferred_path if os.path.exists(preferred_path) else legacy_path
    
    if not os.path.exists(orlibrary_path):
        return summary_rows
    
    df = pd.read_csv(orlibrary_path)
    print(f"  ✓ Loaded OR-Library from {os.path.basename(orlibrary_path)}: {len(df)} rows")
    
    # Group by dataset and K
    for (dataset, k), group_df in df.groupby(["dataset", "K"]):
        group_df = group_df.reset_index(drop=True)
        
        # Extract 5 categories
        summary_rows.extend(_extract_categories(
            group_df,
            experiment="OR-Library",
            dataset=dataset,
            k_value=k
        ))
    
    return summary_rows


def process_yahoo_test(results_dir):
    """
    Process Yahoo Test results: all rows are K=3, dataset=Yahoo_5stocks.
    
    Returns: List of dicts (one per category)
    """
    summary_rows = []
    yahoo_test_path = os.path.join(results_dir, "portfolio_yahoo_test.csv")
    
    if not os.path.exists(yahoo_test_path):
        return summary_rows
    
    df = pd.read_csv(yahoo_test_path)
    print(f"  ✓ Loaded Yahoo Test: {len(df)} rows")
    
    summary_rows.extend(_extract_categories(
        df,
        experiment="Yahoo_Test",
        dataset="Yahoo_5stocks",
        k_value=3
    ))
    
    return summary_rows


def process_yahoo_20stocks(results_dir):
    """
    Process Yahoo 20 Stocks results: all rows are K=5, dataset=Yahoo_20stocks.
    
    Also loads heuristic runs if available (genetic_algorithm, simulated_annealing).
    
    Returns: List of dicts (one per category)
    """
    summary_rows = []
    yahoo_20_path = os.path.join(results_dir, "portfolio_yahoo_20stocks.csv")
    
    if not os.path.exists(yahoo_20_path):
        return summary_rows
    
    df = pd.read_csv(yahoo_20_path)
    print(f"  ✓ Loaded Yahoo 20 Stocks: {len(df)} rows")
    
    # Try to load heuristic runs for GA and SA
    heuristic_path = os.path.join(results_dir, "portfolio_yahoo_20stocks_heuristic_runs.csv")
    
    if os.path.exists(heuristic_path):
        heuristic_df = pd.read_csv(heuristic_path)
        print(f"  ✓ Loaded Yahoo 20 Stocks heuristic runs: {len(heuristic_df)} rows")
        
        # Extract best run for each heuristic method
        for method in ["genetic_algorithm", "simulated_annealing"]:
            method_df = heuristic_df[heuristic_df["method"] == method]
            if len(method_df) > 0:
                # Find best (lowest variance) run
                best_idx = method_df["variance"].idxmin()
                best_row = method_df.loc[best_idx]
                
                # Create a row with same structure as main df
                heur_row = {
                    "method": method,
                    "variance": best_row["variance"],
                    "risk": best_row["risk"],
                    "return": np.nan,  # Not available in heuristic file
                    "number_of_selected_assets": len(json.loads(best_row["selected_indices"].replace("'", '"'))) if isinstance(best_row["selected_indices"], str) else 0,
                    "selected_indices": best_row["selected_indices"],
                    "selected_tickers": best_row["selected_tickers"],
                    "solve_time": best_row.get("runtime", np.nan),
                    "mip_gap": np.nan,
                }
                
                # Add to dataframe
                df = pd.concat([df, pd.DataFrame([heur_row])], ignore_index=True)
                print(f"    ✓ Added best {method} run (var={best_row['variance']:.6f})")
    else:
        print(f"  ⚠ Warning: Heuristic runs file not found at {heuristic_path}")
        print(f"    Yahoo_20stocks will only contain: no_sparsity, cardinality, best_sparse")
    
    summary_rows.extend(_extract_categories(
        df,
        experiment="Yahoo_20stocks",
        dataset="Yahoo_20stocks",
        k_value=5
    ))
    
    return summary_rows


def _extract_categories(group_df, experiment, dataset, k_value):
    """
    Extract 5 categories from a group of results (all same experiment/dataset/K).
    
    Categories:
    1. No-Sparsity Baseline
    2. Gurobi Cardinality
    3. Genetic Algorithm
    4. Simulated Annealing
    5. Best Sparse (lowest variance among cardinality, GA, SA)
    
    Returns: List of row dicts
    """
    summary_rows = []
    
    # Helper to build output row
    def build_row(category_name, method_name, row_data):
        # Handle selected_indices/selected_assets mapping
        selected_indices = safe_get(row_data, "selected_indices", None)
        selected_assets = safe_get(row_data, "selected_assets", None)
        
        # If selected_indices is missing but selected_assets exists, copy it
        if selected_indices is None and selected_assets is not None:
            selected_indices = selected_assets
        
        return {
            "experiment": experiment,
            "dataset": dataset,
            "K": k_value,
            "category": category_name,
            "method": method_name,
            "variance": safe_get(row_data, "variance"),
            "risk": safe_get(row_data, "risk"),
            "return": safe_get(row_data, "return"),
            "number_of_selected_assets": safe_get(row_data, "number_of_selected_assets"),
            "selected_indices": selected_indices,
            "selected_tickers": safe_get(row_data, "selected_tickers"),
            "selected_assets": selected_assets,
            "solve_time": safe_get(row_data, "solve_time"),
            "mip_gap": safe_get(row_data, "mip_gap"),
            "status": safe_get(row_data, "status"),
            "original_number_of_assets": safe_get(row_data, "original_number_of_assets"),
            "number_of_assets_used": safe_get(row_data, "number_of_assets_used"),
            "reduced_instance": safe_get(row_data, "reduced_instance"),
            "reduction_rule": safe_get(row_data, "reduction_rule"),
        }
    
    # 1. No-Sparsity Baseline
    no_sparsity = group_df[group_df["method"] == "no_sparsity"]
    if len(no_sparsity) > 0:
        idx = no_sparsity["variance"].idxmin()
        summary_rows.append(build_row("No-Sparsity Baseline", "no_sparsity", group_df.loc[idx]))
    
    # 2. Gurobi Cardinality
    cardinality = group_df[group_df["method"] == "cardinality"]
    if len(cardinality) > 0:
        idx = cardinality["variance"].idxmin()
        summary_rows.append(build_row("Gurobi Cardinality", "cardinality", group_df.loc[idx]))
    
    # 3. Genetic Algorithm
    ga = group_df[group_df["method"] == "genetic_algorithm"]
    if len(ga) > 0:
        idx = ga["variance"].idxmin()
        summary_rows.append(build_row("Genetic Algorithm", "genetic_algorithm", group_df.loc[idx]))
    
    # 4. Simulated Annealing
    sa = group_df[group_df["method"] == "simulated_annealing"]
    if len(sa) > 0:
        idx = sa["variance"].idxmin()
        summary_rows.append(build_row("Simulated Annealing", "simulated_annealing", group_df.loc[idx]))
    
    # 5. Best Sparse (lowest variance among cardinality, GA, SA)
    sparse_methods = ["cardinality", "genetic_algorithm", "simulated_annealing"]
    sparse_df = group_df[group_df["method"].isin(sparse_methods)]
    if len(sparse_df) > 0:
        idx = sparse_df["variance"].idxmin()
        best_method = group_df.loc[idx, "method"]
        summary_rows.append(build_row("Best Sparse", best_method, group_df.loc[idx]))
    
    return summary_rows


def validate_sparse_cardinality(df_summary):
    """Report cardinality violations for sparse methods/categories only."""
    required_cols = {"K", "number_of_selected_assets", "method", "category"}
    if not required_cols.issubset(df_summary.columns):
        print("\nSparse-method cardinality check skipped: required columns are missing")
        return

    sparse_methods = {"cardinality", "genetic_algorithm", "simulated_annealing"}
    sparse_categories = {
        "Gurobi Cardinality",
        "Genetic Algorithm",
        "Simulated Annealing",
        "Best Sparse",
    }
    sparse_mask = (
        df_summary["method"].isin(sparse_methods)
        | df_summary["category"].isin(sparse_categories)
    )
    violations = df_summary[
        sparse_mask
        & (df_summary["number_of_selected_assets"] > df_summary["K"])
    ]

    print("\nSparse-method cardinality check:")
    print(f"  Evaluated rows: {int(sparse_mask.sum())}")
    print(f"  Violations: {len(violations)}")
    if len(violations) > 0:
        cols = ["experiment", "dataset", "K", "category", "method", "number_of_selected_assets"]
        print(violations[cols].to_string(index=False))


def main():
    """Create combined portfolio summary."""
    
    print("=" * 110)
    print("CREATING OVERALL PORTFOLIO SUMMARY")
    print("=" * 110)
    
    # Define paths
    results_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tables'
    )
    
    if not os.path.exists(results_dir):
        print(f"\n❌ ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load and extract from portfolio files
    print("\nLoading portfolio result files and extracting categories...")
    
    combined_rows = []
    
    # Process each dataset
    combined_rows.extend(process_orlibrary_results(results_dir))
    combined_rows.extend(process_yahoo_test(results_dir))
    combined_rows.extend(process_yahoo_20stocks(results_dir))
    
    if len(combined_rows) == 0:
        print("\n❌ ERROR: No portfolio result files found or no categories extracted")
        sys.exit(1)
    
    df_combined = pd.DataFrame(combined_rows)
    print(f"✓ Extracted {len(df_combined)} categories total")
    
    # Define final column order
    column_order = [
        "experiment",
        "dataset",
        "K",
        "category",
        "method",
        "variance",
        "risk",
        "return",
        "number_of_selected_assets",
        "selected_indices",
        "selected_tickers",
        "selected_assets",
        "solve_time",
        "mip_gap",
        "status",
        "original_number_of_assets",
        "number_of_assets_used",
        "reduced_instance",
        "reduction_rule",
    ]
    
    # Keep only columns that exist in the data
    available_columns = [col for col in column_order if col in df_combined.columns]
    df_combined = df_combined[available_columns]
    
    # Sort by experiment, dataset, K, category
    sort_cols = [col for col in ["experiment", "dataset", "K", "category"] if col in df_combined.columns]
    if sort_cols:
        df_combined = df_combined.sort_values(sort_cols).reset_index(drop=True)

    validate_sparse_cardinality(df_combined)
    
    # Save to file
    output_path = os.path.join(results_dir, 'portfolio_overall_summary.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Summary saved to: {output_path}")
    
    # Print the summary table
    print("\n" + "=" * 110)
    print("OVERALL PORTFOLIO SUMMARY")
    print("=" * 110)
    print()
    
    # Display organized by experiment/dataset
    for experiment in sorted(df_combined["experiment"].unique()):
        exp_df = df_combined[df_combined["experiment"] == experiment]
        print(f"\n{experiment}:")
        
        for dataset in sorted(exp_df["dataset"].unique()):
            df_dataset = exp_df[exp_df["dataset"] == dataset]
            
            # Show K values
            k_vals = sorted(df_dataset["K"].dropna().unique())
            k_str = f" (K={k_vals})" if len(k_vals) > 0 else ""
            print(f"  {dataset}{k_str}:")
            
            # Select key columns for display
            display_cols = ["K", "category", "method", "variance", "risk", "number_of_selected_assets"]
            available_cols = [col for col in display_cols if col in df_dataset.columns]
            
            # Format for readability
            for idx, row in df_dataset.iterrows():
                cat = row["category"]
                meth = row["method"]
                var = row["variance"]
                risk = row["risk"]
                n_assets = row["number_of_selected_assets"]
                print(f"    {cat:25s} | {meth:20s} | var={var:.6f} | risk={risk:.6f} | assets={n_assets}")
    
    # Print summary statistics
    print("\n" + "=" * 110)
    print("SUMMARY STATISTICS")
    print("=" * 110)
    
    for experiment in sorted(df_combined["experiment"].unique()):
        exp_df = df_combined[df_combined["experiment"] == experiment]
        print(f"\n{experiment}:")
        print(f"  Total categories: {len(exp_df)}")
        
        # Count by category type
        for cat_type in ["No-Sparsity Baseline", "Gurobi Cardinality", "Genetic Algorithm", "Simulated Annealing", "Best Sparse"]:
            count = len(exp_df[exp_df["category"] == cat_type])
            if count > 0:
                print(f"    {cat_type}: {count}")
        
        # Best overall for this experiment (lowest variance)
        if "variance" in exp_df.columns:
            best_idx = exp_df["variance"].idxmin()
            best_row = df_combined.loc[best_idx]
            print(f"  Best Overall: {best_row['category']} - "
                  f"{best_row['method']} "
                  f"(variance={best_row['variance']:.6f}, dataset={best_row['dataset']})")
    
    print("\n" + "=" * 110)
    print("✓ OVERALL PORTFOLIO SUMMARY CREATED SUCCESSFULLY")
    print("=" * 110)
    print(f"\nOutput file: {output_path}")
    print(f"Total rows: {len(df_combined)}")


if __name__ == "__main__":
    main()
