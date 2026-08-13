"""Sparse linear regression experiment on PMLB 197_cpu_act dataset.

This script compares four sparse regression methods on the 197_cpu_act dataset:
1. OLS baseline (no sparsity)
2. LASSO L1 relaxation (multiple alpha values)
3. Best subset selection via Gurobi (multiple cardinality constraints)
4. Iterative Hard Thresholding / IHT (multiple cardinality constraints)

The 197_cpu_act dataset is a medium-size regression benchmark from PMLB with
6553 training observations and 21 features.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from time import time

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from regression_data import load_pmlb_regression_data
from regression_lasso import solve_ols_regression, solve_lasso_regression
from regression_gurobi import solve_best_subset_gurobi
from regression_iht import solve_iht_regression


# ============================================================================
# Configuration
# ============================================================================
DATASET_NAME = "197_cpu_act"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sparse methods configuration
K_VALUES = [5, 10, 15]
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
GUROBI_TIME_LIMIT = 60
GUROBI_BIG_M = 1000
IHT_N_RESTARTS = 10
IHT_REFIT_SUPPORT = True

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "tables"


# ============================================================================
# Helper Functions
# ============================================================================
def get_feature_names(selected_indices, feature_names_list):
    """Map feature indices to feature names.

    Parameters
    ----------
    selected_indices : list or float
        List of feature indices, or NaN/None if no features selected.
    feature_names_list : list
        List of all feature names.

    Returns
    -------
    list or None
        List of feature names corresponding to indices, or None if input invalid.
    """
    if selected_indices is None or (isinstance(selected_indices, float) and np.isnan(selected_indices)):
        return None
    if isinstance(selected_indices, list):
        return [feature_names_list[i] for i in selected_indices]
    return None


def main():
    """Run the main experiment."""
    print("=" * 80)
    print("SPARSE LINEAR REGRESSION EXPERIMENT: PMLB 197_cpu_act DATASET")
    print("=" * 80)
    print()

    # ========================================================================
    # Load data
    # ========================================================================
    print("Loading PMLB 197_cpu_act dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_pmlb_regression_data(
        dataset_name=DATASET_NAME,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        standardize=True,
    )
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {feature_names}")
    print()

    # ========================================================================
    # Run experiments
    # ========================================================================
    results = []

    # OLS baseline
    print("Running OLS baseline...")
    result_ols = solve_ols_regression(X_train, y_train, X_test, y_test)
    result_ols["dataset"] = DATASET_NAME
    result_ols["K"] = None
    result_ols["alpha"] = None
    result_ols["mip_gap"] = None
    result_ols["iterations"] = None
    result_ols["n_restarts"] = None
    result_ols["selected_feature_names"] = get_feature_names(
        result_ols["selected_features"], feature_names
    )
    print(f"  OLS: test_mse={result_ols['test_mse']:.6f}")
    results.append(result_ols)
    print()

    # LASSO with multiple alpha values
    print("Running LASSO with multiple alpha values...")
    for alpha in LASSO_ALPHAS:
        result_lasso = solve_lasso_regression(X_train, y_train, X_test, y_test, alpha=alpha)
        result_lasso["dataset"] = DATASET_NAME
        result_lasso["K"] = None
        result_lasso["mip_gap"] = None
        result_lasso["iterations"] = None
        result_lasso["n_restarts"] = None
        result_lasso["selected_feature_names"] = get_feature_names(
            result_lasso["selected_features"], feature_names
        )
        print(f"  LASSO (α={alpha}): test_mse={result_lasso['test_mse']:.6f}, features={result_lasso['number_of_selected_features']}")
        results.append(result_lasso)
    print()

    # Gurobi best subset selection
    print("Running Gurobi best subset selection with hard cardinality constraints...")
    for K in K_VALUES:
        result_gurobi = solve_best_subset_gurobi(
            X_train,
            y_train,
            X_test,
            y_test,
            K=K,
            time_limit=GUROBI_TIME_LIMIT,
            big_m=GUROBI_BIG_M,
        )
        result_gurobi["dataset"] = DATASET_NAME
        result_gurobi["alpha"] = None
        result_gurobi["iterations"] = None
        result_gurobi["n_restarts"] = None
        result_gurobi["selected_feature_names"] = get_feature_names(
            result_gurobi["selected_features"], feature_names
        )
        print(f"  Gurobi (K={K}): test_mse={result_gurobi['test_mse']:.6f}, features={result_gurobi['number_of_selected_features']}, status={result_gurobi['status']}")
        results.append(result_gurobi)
    print()

    # Iterative Hard Thresholding
    print("Running Iterative Hard Thresholding with multiple restarts...")
    for K in K_VALUES:
        result_iht = solve_iht_regression(
            X_train,
            y_train,
            X_test,
            y_test,
            K=K,
            max_iter=1000,
            n_restarts=IHT_N_RESTARTS,
            refit_support=IHT_REFIT_SUPPORT,
        )
        result_iht["dataset"] = DATASET_NAME
        result_iht["alpha"] = None
        result_iht["mip_gap"] = None
        result_iht["selected_feature_names"] = get_feature_names(
            result_iht["selected_features"], feature_names
        )
        print(f"  IHT (K={K}): test_mse={result_iht['test_mse']:.6f}, features={result_iht['number_of_selected_features']}, status={result_iht['status']}")
        results.append(result_iht)
    print()

    # ========================================================================
    # Save full results
    # ========================================================================
    df_results = pd.DataFrame(results)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reorder columns for clarity
    column_order = [
        "dataset",
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

    # Convert lists to clean JSON strings for CSV
    df_results_save = df_results.copy()
    df_results_save["selected_features"] = df_results_save["selected_features"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    df_results_save["selected_feature_names"] = df_results_save["selected_feature_names"].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else x
    )
    df_results_save["coefficients"] = df_results_save["coefficients"].apply(
        lambda x: json.dumps(list(x)) if hasattr(x, '__iter__') and not isinstance(x, str) else x
    )

    # Format K as integers where applicable
    df_results_save["K"] = df_results_save["K"].apply(
        lambda x: int(x) if x is not None and not np.isnan(x) else x
    )

    results_file = OUTPUT_DIR / f"regression_pmlb_{DATASET_NAME}_results.csv"
    df_results_save.to_csv(results_file, index=False)
    print(f"Full results saved to: {results_file}")
    print()

    # ========================================================================
    # Print full results table
    # ========================================================================
    print("=" * 80)
    print("FULL RESULTS TABLE")
    print("=" * 80)
    print(df_results.to_string())
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
    summary_file = OUTPUT_DIR / f"regression_pmlb_{DATASET_NAME}_summary.csv"

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
    print()

    # ========================================================================
    # Print summary table
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: KEY RESULTS BY CATEGORY")
    print("=" * 80)
    print(df_summary.to_string())
    print()

    print("=" * 80)
    print("✓ PMLB regression experiment completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
