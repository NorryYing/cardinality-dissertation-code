"""
Test script for baseline regression methods: OLS and LASSO.

This script loads the diabetes dataset and runs both OLS and LASSO regression
with multiple alpha values, then saves results to a CSV file.

Run from the project root:
    python experiments/test_regression_baselines.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from regression_data import load_diabetes_data
from regression_lasso import solve_ols_regression, solve_lasso_regression


def main():
    """Load data, run regression methods, and save results."""
    
    print("Loading diabetes dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_diabetes_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()
    
    # Collect results
    results = []
    
    # OLS baseline
    print("Running OLS regression...")
    ols_result = solve_ols_regression(X_train, y_train, X_test, y_test)
    ols_result["alpha"] = None  # OLS doesn't have alpha parameter
    results.append(ols_result)
    print(f"  OLS: train_mse={ols_result['train_mse']:.6f}, test_mse={ols_result['test_mse']:.6f}")
    print()
    
    # LASSO with different alpha values
    alpha_values = [0.001, 0.01, 0.1, 1.0]
    print("Running LASSO regression with multiple alpha values...")
    for alpha in alpha_values:
        lasso_result = solve_lasso_regression(X_train, y_train, X_test, y_test, alpha=alpha)
        results.append(lasso_result)
        print(f"  LASSO (alpha={alpha}): train_mse={lasso_result['train_mse']:.6f}, test_mse={lasso_result['test_mse']:.6f}, features={lasso_result['number_of_selected_features']}")
    print()
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Reorder and select columns for display
    display_columns = [
        "method",
        "alpha",
        "train_mse",
        "test_mse",
        "number_of_selected_features",
        "selected_features",
        "solve_time",
        "status",
    ]
    df_display = df_results[display_columns].copy()
    
    # Print results table
    print("=" * 120)
    print("RESULTS TABLE")
    print("=" * 120)
    print(df_display.to_string(index=False))
    print()
    
    # Save full results to CSV
    output_dir = project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "regression_baseline_test.csv"
    
    # Convert lists to strings for CSV storage
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
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] != 'success')}")
    print()
    
    # Feature selection comparison
    print("Feature Selection Summary:")
    for result in results:
        method_str = result["method"]
        if result["alpha"] is not None:
            method_str = f"{method_str} (α={result['alpha']})"
        print(f"  {method_str}: {result['number_of_selected_features']} features selected")


if __name__ == "__main__":
    main()
