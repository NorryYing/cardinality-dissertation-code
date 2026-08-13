"""
Test script for Iterative Hard Thresholding (IHT) regression.

This script loads the diabetes dataset and runs IHT with hard cardinality constraints
for multiple K values, then saves results to a CSV file.

Run from the project root:
    python experiments/test_regression_iht.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from regression_data import load_diabetes_data
from regression_iht import solve_iht_regression


def main():
    """Load data, run IHT solver, and save results."""
    
    print("Loading diabetes dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_diabetes_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print()
    
    # Collect results
    results = []
    
    # IHT with different K values
    K_values = [3, 5, 7]
    print("Running Iterative Hard Thresholding with direct hard cardinality constraints...")
    print("(Using multiple restarts and restricted OLS refitting for stability)")
    for K in K_values:
        print(f"  Solving for K={K}...")
        result = solve_iht_regression(
            X_train, y_train, X_test, y_test,
            K=K,
            learning_rate=None,  # Auto-compute as 0.5/L
            max_iter=1000,
            tol=1e-6,
            n_restarts=10,  # Multiple restarts for better solutions
            refit_support=True  # Refit OLS on selected support
        )
        results.append(result)
        
        print(f"    K={K}: train_mse={result['train_mse']:.6f}, test_mse={result['test_mse']:.6f}, features={result['number_of_selected_features']}, iterations={result['iterations']}, restarts={result['n_restarts']}, status={result['status']}")
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
        "iterations",
        "n_restarts",
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
    output_file = output_dir / "regression_iht_test.csv"
    
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
    print(f"Converged: {sum(1 for r in results if 'converged' in r['status'])}")
    print(f"Max iterations reached: {sum(1 for r in results if 'max_iter_reached' in r['status'])}")
    print(f"Failed: {sum(1 for r in results if 'error' in r['status'].lower())}")
    print()
    
    # Feature selection and performance comparison
    print("Feature Selection and Performance Summary (Direct Hard Thresholding with Restarts):")
    for result in results:
        if result["number_of_selected_features"] is not None:
            print(f"  K={result['K']}: {result['number_of_selected_features']} features selected, test_mse={result['test_mse']:.6f}, iterations={result['iterations']}, restarts={result['n_restarts']}, status={result['status']}")
        else:
            print(f"  K={result['K']}: Failed - {result['status']}")


if __name__ == "__main__":
    main()
