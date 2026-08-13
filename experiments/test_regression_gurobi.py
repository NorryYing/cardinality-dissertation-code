"""
Test script for Gurobi-based best subset selection regression.

This script loads the diabetes dataset and runs best subset selection
with hard cardinality constraints for multiple K values, then saves results to a CSV file.

Run from the project root:
    python experiments/test_regression_gurobi.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from regression_data import load_diabetes_data
from regression_gurobi import solve_best_subset_gurobi


def main():
    """Load data, run Gurobi solver, and save results."""
    
    print("Loading diabetes dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_diabetes_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()
    
    # Collect results
    results = []
    
    # Best subset selection with different K values
    K_values = [3, 5, 7]
    print("Running best subset selection with Gurobi (hard cardinality constraints)...")
    for K in K_values:
        print(f"  Solving for K={K}...")
        result = solve_best_subset_gurobi(
            X_train, y_train, X_test, y_test, 
            K=K, 
            time_limit=60, 
            big_m=100
        )
        results.append(result)
        
        if result["status"] == "optimal":
            print(f"    K={K}: train_mse={result['train_mse']:.6f}, test_mse={result['test_mse']:.6f}, features={result['number_of_selected_features']} (OPTIMAL)")
        elif result["status"] == "time_limit":
            print(f"    K={K}: train_mse={result['train_mse']:.6f}, test_mse={result['test_mse']:.6f}, features={result['number_of_selected_features']} (TIME LIMIT)")
        else:
            print(f"    K={K}: {result['status']}")
    print()
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Reorder and select columns for display
    display_columns = [
        "method",
        "K",
        "train_mse",
        "test_mse",
        "number_of_selected_features",
        "selected_features",
        "solve_time",
        "mip_gap",
        "status",
    ]
    df_display = df_results[display_columns].copy()
    
    # Print results table
    print("=" * 140)
    print("RESULTS TABLE")
    print("=" * 140)
    print(df_display.to_string(index=False))
    print()
    
    # Save full results to CSV
    output_dir = project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "regression_gurobi_test.csv"
    
    # Convert lists and arrays to strings for CSV storage
    df_save = df_results.copy()
    df_save["selected_features"] = df_save["selected_features"].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    df_save["coefficients"] = df_save["coefficients"].apply(
        lambda x: str(x.tolist()) if hasattr(x, 'tolist') else x
    )
    
    df_save.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()
    
    # Summary statistics
    print("=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(f"Total runs: {len(results)}")
    print(f"Successful (optimal or suboptimal): {sum(1 for r in results if 'optimal' in r['status'] or 'suboptimal' in r['status'])}")
    print(f"Time limit reached: {sum(1 for r in results if 'time_limit' in r['status'])}")
    print(f"Failed: {sum(1 for r in results if 'error' in r['status'].lower() or 'infeasible' in r['status'])}")
    print()
    
    # Feature selection comparison
    print("Feature Selection Summary (Hard Cardinality Constraint):")
    for result in results:
        if result["number_of_selected_features"] is not None:
            print(f"  K={result['K']}: {result['number_of_selected_features']} features selected, test_mse={result['test_mse']:.6f}, status={result['status']}")
        else:
            print(f"  K={result['K']}: Failed - {result['status']}")


if __name__ == "__main__":
    main()
