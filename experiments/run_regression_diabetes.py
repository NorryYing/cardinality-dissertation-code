"""
Main experiment script: Sparse linear regression on diabetes dataset.

This script compares four method groups for feature selection:
1. OLS: No-sparsity baseline
2. LASSO: L1 regularization baseline
3. Gurobi: Direct hard cardinality (optimal for small problems)
4. IHT: Iterative hard thresholding (fast heuristic)

Run from the project root:
    python experiments/run_regression_diabetes.py
"""

import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import pandas as pd
import numpy as np
from regression_data import load_diabetes_data
from regression_lasso import solve_ols_regression, solve_lasso_regression
from regression_gurobi import solve_best_subset_gurobi
from regression_iht import solve_iht_regression


def main():
    """Run full regression comparison experiment."""
    
    print("\n" + "=" * 100)
    print("SPARSE LINEAR REGRESSION EXPERIMENT: DIABETES DATASET")
    print("=" * 100)
    print()
    
    # ========================================================================
    # Load data
    # ========================================================================
    print("Loading diabetes dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_diabetes_data(
        test_size=0.2,
        random_state=42,
        standardize=True
    )
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {feature_names}")
    print()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    K_values = [3, 5, 7]
    lasso_alphas = [0.001, 0.01, 0.1, 1.0]
    
    # ========================================================================
    # Collect all results
    # ========================================================================
    all_results = []
    
    # ========================================================================
    # 1. OLS Baseline (no sparsity)
    # ========================================================================
    print("Running OLS baseline...")
    ols_result = solve_ols_regression(X_train, y_train, X_test, y_test)
    ols_result["K"] = None
    ols_result["alpha"] = None
    ols_result["mip_gap"] = None
    ols_result["iterations"] = None
    ols_result["n_restarts"] = None
    all_results.append(ols_result)
    print(f"  OLS: test_mse={ols_result['test_mse']:.6f}")
    print()
    
    # ========================================================================
    # 2. LASSO Baseline (L1 regularization)
    # ========================================================================
    print("Running LASSO with multiple alpha values...")
    for alpha in lasso_alphas:
        lasso_result = solve_lasso_regression(X_train, y_train, X_test, y_test, alpha=alpha)
        lasso_result["K"] = None
        lasso_result["mip_gap"] = None
        lasso_result["iterations"] = None
        lasso_result["n_restarts"] = None
        all_results.append(lasso_result)
        print(f"  LASSO (α={alpha}): test_mse={lasso_result['test_mse']:.6f}, features={lasso_result['number_of_selected_features']}")
    print()
    
    # ========================================================================
    # 3. Gurobi Best Subset Selection (hard cardinality - optimal)
    # ========================================================================
    print("Running Gurobi best subset selection with hard cardinality constraints...")
    for K in K_values:
        gurobi_result = solve_best_subset_gurobi(
            X_train, y_train, X_test, y_test,
            K=K,
            time_limit=60,
            big_m=1000
        )
        gurobi_result["alpha"] = None
        gurobi_result["iterations"] = None
        gurobi_result["n_restarts"] = None
        all_results.append(gurobi_result)
        print(f"  Gurobi (K={K}): test_mse={gurobi_result['test_mse']:.6f}, features={gurobi_result['number_of_selected_features']}, status={gurobi_result['status']}")
    print()
    
    # ========================================================================
    # 4. Iterative Hard Thresholding (fast heuristic)
    # ========================================================================
    print("Running Iterative Hard Thresholding with multiple restarts...")
    for K in K_values:
        iht_result = solve_iht_regression(
            X_train, y_train, X_test, y_test,
            K=K,
            learning_rate=None,  # Auto-compute as 0.5/L
            max_iter=1000,
            tol=1e-6,
            n_restarts=10,
            refit_support=True
        )
        iht_result["alpha"] = None
        iht_result["mip_gap"] = None
        all_results.append(iht_result)
        print(f"  IHT (K={K}): test_mse={iht_result['test_mse']:.6f}, features={iht_result['number_of_selected_features']}, status={iht_result['status']}")
    print()
    
    # ========================================================================
    # Create full results DataFrame
    # ========================================================================
    df_results = pd.DataFrame(all_results)
    
    # Add selected_feature_names by mapping indices to feature names
    def get_feature_names(selected_indices, feature_names_list):
        """Convert feature indices to feature names."""
        if selected_indices is None or (isinstance(selected_indices, float) and np.isnan(selected_indices)):
            return None
        if isinstance(selected_indices, list):
            return [feature_names_list[i] for i in selected_indices]
        return None
    
    df_results["selected_feature_names"] = df_results["selected_features"].apply(
        lambda x: get_feature_names(x, feature_names)
    )
    
    # Format K as integer where applicable (remove .0)
    df_results["K"] = df_results["K"].apply(lambda x: int(x) if x is not None and not np.isnan(x) else x)
    df_results["alpha"] = df_results["alpha"].apply(lambda x: x if x is None or np.isnan(x) else x)
    
    # Reorder columns for clarity
    column_order = [
        "method",
        "K",
        "alpha",
        "train_mse",
        "test_mse",
        "number_of_selected_features",
        "selected_features",
        "selected_feature_names",
        "coefficients",
        "intercept",
        "solve_time",
        "mip_gap",
        "iterations",
        "n_restarts",
        "status",
    ]
    df_results = df_results[column_order]
    
    # ========================================================================
    # Save full results
    # ========================================================================
    output_dir = project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "regression_diabetes_results.csv"
    
    # Convert lists and arrays to clean JSON strings for CSV storage
    df_save = df_results.copy()
    df_save["selected_features"] = df_save["selected_features"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    df_save["selected_feature_names"] = df_save["selected_feature_names"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    df_save["coefficients"] = df_save["coefficients"].apply(
        lambda x: json.dumps(list(x)) if hasattr(x, '__iter__') and not isinstance(x, str) else x
    )
    
    df_save.to_csv(output_file, index=False)
    print(f"Full results saved to: {output_file}")
    
    # ========================================================================
    # Display results table
    # ========================================================================
    print()
    print("=" * 140)
    print("FULL RESULTS TABLE")
    print("=" * 140)
    
    # Create display version with formatted columns
    df_display = df_results.copy()
    df_display["test_mse"] = df_display["test_mse"].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    df_display["train_mse"] = df_display["train_mse"].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    df_display["solve_time"] = df_display["solve_time"].apply(lambda x: f"{x:.6f}" if x is not None else "N/A")
    
    print(df_display.to_string(index=False))
    print()
    
    # ========================================================================
    # Create summary
    # ========================================================================
    summary_data = []
    total_number_of_features = len(feature_names)

    # Best overall by test MSE
    valid_results = df_results[df_results["test_mse"].notna()].sort_values("test_mse")
    if not valid_results.empty:
        best_overall = valid_results.iloc[0]
        summary_data.append({
            "category": "Best Overall",
            "method": best_overall["method"],
            "K": best_overall["K"],
            "alpha": best_overall["alpha"],
            "test_mse": best_overall["test_mse"],
            "features": best_overall["number_of_selected_features"],
            "selected_feature_names": best_overall["selected_feature_names"],
            "status": best_overall["status"],
        })
    
    # Best sparse method (where features < total features)
    sparse_results = valid_results[
        valid_results["number_of_selected_features"] < total_number_of_features
    ].sort_values("test_mse")
    if not sparse_results.empty:
        best_sparse = sparse_results.iloc[0]
        summary_data.append({
            "category": "Best Sparse",
            "method": best_sparse["method"],
            "K": best_sparse["K"],
            "alpha": best_sparse["alpha"],
            "test_mse": best_sparse["test_mse"],
            "features": best_sparse["number_of_selected_features"],
            "selected_feature_names": best_sparse["selected_feature_names"],
            "status": best_sparse["status"],
        })
    
    # Best hard cardinality method (Gurobi or IHT)
    hard_card_results = valid_results[
        (valid_results["method"] == "Best Subset (Gurobi)") | (valid_results["method"] == "IHT")
    ].sort_values("test_mse")
    if not hard_card_results.empty:
        best_hard_card = hard_card_results.iloc[0]
        summary_data.append({
            "category": "Best Hard Cardinality",
            "method": best_hard_card["method"],
            "K": best_hard_card["K"],
            "alpha": best_hard_card["alpha"],
            "test_mse": best_hard_card["test_mse"],
            "features": best_hard_card["number_of_selected_features"],
            "selected_feature_names": best_hard_card["selected_feature_names"],
            "status": best_hard_card["status"],
        })
    
    # Best Gurobi result
    gurobi_results = valid_results[valid_results["method"] == "Best Subset (Gurobi)"].sort_values("test_mse")
    if not gurobi_results.empty:
        best_gurobi = gurobi_results.iloc[0]
        summary_data.append({
            "category": "Best Gurobi",
            "method": best_gurobi["method"],
            "K": best_gurobi["K"],
            "alpha": best_gurobi["alpha"],
            "test_mse": best_gurobi["test_mse"],
            "features": best_gurobi["number_of_selected_features"],
            "selected_feature_names": best_gurobi["selected_feature_names"],
            "status": best_gurobi["status"],
        })
    
    # Best IHT result
    iht_results = valid_results[valid_results["method"] == "IHT"].sort_values("test_mse")
    if not iht_results.empty:
        best_iht = iht_results.iloc[0]
        summary_data.append({
            "category": "Best IHT",
            "method": best_iht["method"],
            "K": best_iht["K"],
            "alpha": best_iht["alpha"],
            "test_mse": best_iht["test_mse"],
            "features": best_iht["number_of_selected_features"],
            "selected_feature_names": best_iht["selected_feature_names"],
            "status": best_iht["status"],
        })
    
    # Best LASSO result
    lasso_results = valid_results[valid_results["method"] == "LASSO"].sort_values("test_mse")
    if not lasso_results.empty:
        best_lasso = lasso_results.iloc[0]
        summary_data.append({
            "category": "Best LASSO",
            "method": best_lasso["method"],
            "K": best_lasso["K"],
            "alpha": best_lasso["alpha"],
            "test_mse": best_lasso["test_mse"],
            "features": best_lasso["number_of_selected_features"],
            "selected_feature_names": best_lasso["selected_feature_names"],
            "status": best_lasso["status"],
        })
    
    # OLS result
    ols_results = valid_results[valid_results["method"] == "OLS"]
    if not ols_results.empty:
        ols_res = ols_results.iloc[0]
        summary_data.append({
            "category": "OLS Baseline",
            "method": ols_res["method"],
            "K": ols_res["K"],
            "alpha": ols_res["alpha"],
            "test_mse": ols_res["test_mse"],
            "features": ols_res["number_of_selected_features"],
            "selected_feature_names": ols_res["selected_feature_names"],
            "status": ols_res["status"],
        })
    
    # Save summary
    df_summary = pd.DataFrame(summary_data)
    summary_file = output_dir / "regression_diabetes_summary.csv"
    
    # Convert feature names list to clean JSON string for CSV
    df_summary_save = df_summary.copy()
    df_summary_save["selected_feature_names"] = df_summary_save["selected_feature_names"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    
    # Format K as integers where applicable
    df_summary_save["K"] = df_summary_save["K"].apply(
        lambda x: int(x) if x is not None and not np.isnan(x) else x
    )
    
    df_summary_save.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    # ========================================================================
    # Display summary
    # ========================================================================
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print()
    
    # ========================================================================
    # Key statistics
    # ========================================================================
    print("=" * 100)
    print("KEY STATISTICS")
    print("=" * 100)
    print(f"Total experiments run: {len(df_results)}")
    print(f"Successful: {(df_results['test_mse'].notna()).sum()}")
    print(f"Failed: {(df_results['test_mse'].isna()).sum()}")
    print()
    
    print("Results by method:")
    for method in df_results["method"].unique():
        method_results = df_results[df_results["method"] == method]
        successful = method_results[method_results["test_mse"].notna()]
        if not successful.empty:
            best_mse = successful["test_mse"].min()
            avg_mse = successful["test_mse"].mean()
            print(f"  {method}: best_test_mse={best_mse:.6f}, avg_test_mse={avg_mse:.6f}, runs={len(method_results)}")
    print()


if __name__ == "__main__":
    main()
