"""
Main experiment script for Communities and Crime regression dataset.
Compares OLS, LASSO, Gurobi best subset selection, and IHT methods.

Feature reduction: Top 30 features based on absolute correlation with y_train
to avoid exceeding the size-limited Gurobi license.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.regression_data import load_communities_crime_data
from src.regression_lasso import solve_ols_regression, solve_lasso_regression
from src.regression_gurobi import solve_best_subset_gurobi
from src.regression_iht import solve_iht_regression

# Experiment configuration
DATASET_NAME = "Communities_Crime"
K_VALUES = [5, 10, 15]
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0]
MAX_FEATURES = 30
GUROBI_TIME_LIMIT = 60
IHT_N_RESTARTS = 10
IHT_REFIT_SUPPORT = True

# Results storage
results_rows = []

def select_features_by_correlation(X_train, y_train, feature_names, max_features=30):
    """
    Select top max_features based on absolute correlation with y_train.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        feature_names: List of feature names
        max_features: Number of features to select
    
    Returns:
        selected_indices: Indices of selected features
        selected_feature_names: Names of selected features
    """
    # Compute absolute correlation with y_train
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    
    # Get top max_features indices
    selected_indices = np.argsort(correlations)[-max_features:][::-1]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"\nFeature Reduction Summary:")
    print(f"  Original number of features: {len(feature_names)}")
    print(f"  Selected number of features: {max_features}")
    print(f"  Top 5 selected features: {selected_feature_names[:5]}")
    print(f"  Top 5 correlations: {correlations[selected_indices[:5]]}")
    
    return selected_indices, selected_feature_names

def run_experiments():
    """Run all regression experiments."""
    
    print("=" * 80)
    print("REGRESSION EXPERIMENTS: COMMUNITIES AND CRIME DATASET")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_names = load_communities_crime_data(
        test_size=0.2, 
        random_state=42, 
        standardize=True
    )
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Original number of features: {len(feature_names)}")
    
    # Feature reduction based on correlation
    print(f"\nReducing features to top {MAX_FEATURES} based on correlation...")
    selected_indices, selected_feature_names = select_features_by_correlation(
        X_train, y_train, feature_names, max_features=MAX_FEATURES
    )
    
    # Apply feature reduction
    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced = X_test[:, selected_indices]
    print(f"After feature reduction: X_train={X_train_reduced.shape}, X_test={X_test_reduced.shape}")
    
    original_num_features = len(feature_names)
    num_features_used = MAX_FEATURES
    feature_reduction_rule = f"Top {MAX_FEATURES} features by absolute correlation with y_train"
    
    # ===== RUN EXPERIMENTS =====
    
    # 1. OLS baseline
    print("\n" + "=" * 80)
    print("1. Running OLS baseline...")
    print("=" * 80)
    result = solve_ols_regression(X_train_reduced, y_train, X_test_reduced, y_test)
    result["dataset"] = DATASET_NAME
    result["K"] = None
    result["alpha"] = None
    result["mip_gap"] = None
    result["iterations"] = None
    result["n_restarts"] = None
    result["original_number_of_features"] = original_num_features
    result["number_of_features_used"] = num_features_used
    result["feature_reduction_rule"] = feature_reduction_rule
    # Convert selected_features indices to feature names
    result["selected_feature_names"] = [selected_feature_names[i] for i in result["selected_features"]]
    result["selected_features"] = json.dumps(result["selected_features"])
    result["selected_feature_names"] = json.dumps(result["selected_feature_names"])
    result["coefficients"] = json.dumps(result["coefficients"].tolist())
    results_rows.append(result)
    print(f"OLS: train_mse={result['train_mse']:.4f}, test_mse={result['test_mse']:.4f}")
    
    # 2. LASSO with different alpha values
    print("\n" + "=" * 80)
    print("2. Running LASSO with different alpha values...")
    print("=" * 80)
    for alpha in LASSO_ALPHAS:
        print(f"\nLASSO (alpha={alpha})...")
        result = solve_lasso_regression(X_train_reduced, y_train, X_test_reduced, y_test, alpha=alpha)
        result["dataset"] = DATASET_NAME
        result["K"] = None
        result["mip_gap"] = None
        result["iterations"] = None
        result["n_restarts"] = None
        result["original_number_of_features"] = original_num_features
        result["number_of_features_used"] = num_features_used
        result["feature_reduction_rule"] = feature_reduction_rule
        # Convert selected_features indices to feature names
        result["selected_feature_names"] = [selected_feature_names[i] for i in result["selected_features"]]
        result["selected_features"] = json.dumps(result["selected_features"])
        result["selected_feature_names"] = json.dumps(result["selected_feature_names"])
        result["coefficients"] = json.dumps(result["coefficients"].tolist())
        results_rows.append(result)
        print(f"  train_mse={result['train_mse']:.4f}, test_mse={result['test_mse']:.4f}, "
              f"num_features={result['number_of_selected_features']}")
    
    # 3. Gurobi best subset selection
    print("\n" + "=" * 80)
    print("3. Running Gurobi best subset selection...")
    print("=" * 80)
    for K in K_VALUES:
        print(f"\nGurobi (K={K})...")
        result = solve_best_subset_gurobi(X_train_reduced, y_train, X_test_reduced, y_test, K, 
                                         time_limit=GUROBI_TIME_LIMIT, big_m=1000)
        result["dataset"] = DATASET_NAME
        result["alpha"] = None
        result["iterations"] = None
        result["n_restarts"] = None
        result["original_number_of_features"] = original_num_features
        result["number_of_features_used"] = num_features_used
        result["feature_reduction_rule"] = feature_reduction_rule
        # Convert selected_features indices to feature names
        result["selected_feature_names"] = [selected_feature_names[i] for i in result["selected_features"]]
        result["selected_features"] = json.dumps(result["selected_features"])
        result["selected_feature_names"] = json.dumps(result["selected_feature_names"])
        result["coefficients"] = json.dumps(result["coefficients"].tolist())
        results_rows.append(result)
        print(f"  train_mse={result['train_mse']:.4f}, test_mse={result['test_mse']:.4f}, "
              f"num_features={result['number_of_selected_features']}, mip_gap={result['mip_gap']:.6f}")
    
    # 4. Iterative Hard Thresholding
    print("\n" + "=" * 80)
    print("4. Running Iterative Hard Thresholding...")
    print("=" * 80)
    for K in K_VALUES:
        print(f"\nIHT (K={K})...")
        result = solve_iht_regression(X_train_reduced, y_train, X_test_reduced, y_test, K,
                                      n_restarts=IHT_N_RESTARTS, refit_support=IHT_REFIT_SUPPORT)
        result["dataset"] = DATASET_NAME
        result["alpha"] = None
        result["mip_gap"] = None
        result["original_number_of_features"] = original_num_features
        result["number_of_features_used"] = num_features_used
        result["feature_reduction_rule"] = feature_reduction_rule
        # Convert selected_features indices to feature names
        result["selected_feature_names"] = [selected_feature_names[i] for i in result["selected_features"]]
        result["selected_features"] = json.dumps(result["selected_features"])
        result["selected_feature_names"] = json.dumps(result["selected_feature_names"])
        result["coefficients"] = json.dumps(result["coefficients"].tolist())
        results_rows.append(result)
        print(f"  train_mse={result['train_mse']:.4f}, test_mse={result['test_mse']:.4f}, "
              f"num_features={result['number_of_selected_features']}, iterations={result['iterations']}")
    
    # ===== SAVE RESULTS =====
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_rows)
    
    # Ensure proper column order
    column_order = [
        "dataset", "method", "K", "alpha", "train_mse", "test_mse",
        "number_of_selected_features", "selected_features", "selected_feature_names",
        "coefficients", "intercept", "solve_time", "mip_gap", "iterations", "n_restarts",
        "status", "original_number_of_features", "number_of_features_used", "feature_reduction_rule"
    ]
    results_df = results_df[column_order]
    
    # Convert K and alpha to proper types
    results_df["K"] = results_df["K"].apply(lambda x: int(x) if x is not None and not np.isnan(float(x) if isinstance(x, (int, float, np.number)) else 0) else x)
    results_df["alpha"] = results_df["alpha"].apply(lambda x: x if x is None else float(x))
    
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'regression_communities_results.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    print(f"  Total experiments: {len(results_df)}")
    
    # ===== CREATE SUMMARY =====
    print("\n" + "=" * 80)
    print("CREATING SUMMARY")
    print("=" * 80)
    
    summary_data = {}
    
    # Best Overall (lowest test_mse)
    best_overall_idx = results_df["test_mse"].idxmin()
    best_overall = results_df.loc[best_overall_idx]
    summary_data["Best Overall"] = {
        "method": best_overall["method"],
        "K": best_overall["K"],
        "alpha": best_overall["alpha"],
        "test_mse": best_overall["test_mse"],
        "number_of_selected_features": best_overall["number_of_selected_features"]
    }
    print(f"\nBest Overall: {best_overall['method']} "
          f"(test_mse={best_overall['test_mse']:.4f}, features={best_overall['number_of_selected_features']})")
    
    # Best Sparse (lowest test_mse where features < total features used)
    valid_sparse = results_df[results_df["number_of_selected_features"] < results_df["number_of_features_used"]]
    if len(valid_sparse) > 0:
        best_sparse_idx = valid_sparse["test_mse"].idxmin()
        best_sparse = results_df.loc[best_sparse_idx]
        summary_data["Best Sparse"] = {
            "method": best_sparse["method"],
            "K": best_sparse["K"],
            "alpha": best_sparse["alpha"],
            "test_mse": best_sparse["test_mse"],
            "number_of_selected_features": best_sparse["number_of_selected_features"]
        }
        print(f"Best Sparse: {best_sparse['method']} "
              f"(test_mse={best_sparse['test_mse']:.4f}, features={best_sparse['number_of_selected_features']})")
    else:
        summary_data["Best Sparse"] = {"method": "N/A", "test_mse": np.nan}
        print("Best Sparse: N/A (no sparse solutions)")
    
    # Best Hard Cardinality (Gurobi or IHT only)
    hard_cardinality = results_df[results_df["method"].isin(["Best Subset (Gurobi)", "IHT"])]
    if len(hard_cardinality) > 0:
        best_hard_idx = hard_cardinality["test_mse"].idxmin()
        best_hard = results_df.loc[best_hard_idx]
        summary_data["Best Hard Cardinality"] = {
            "method": best_hard["method"],
            "K": best_hard["K"],
            "alpha": best_hard["alpha"],
            "test_mse": best_hard["test_mse"],
            "number_of_selected_features": best_hard["number_of_selected_features"]
        }
        print(f"Best Hard Cardinality: {best_hard['method']} "
              f"(test_mse={best_hard['test_mse']:.4f}, features={best_hard['number_of_selected_features']})")
    else:
        summary_data["Best Hard Cardinality"] = {"method": "N/A", "test_mse": np.nan}
    
    # Best Gurobi
    gurobi_results = results_df[results_df["method"] == "Best Subset (Gurobi)"]
    if len(gurobi_results) > 0:
        best_gurobi_idx = gurobi_results["test_mse"].idxmin()
        best_gurobi = results_df.loc[best_gurobi_idx]
        summary_data["Best Gurobi"] = {
            "method": "Best Subset (Gurobi)",
            "K": best_gurobi["K"],
            "test_mse": best_gurobi["test_mse"],
            "number_of_selected_features": best_gurobi["number_of_selected_features"]
        }
        print(f"Best Gurobi: K={best_gurobi['K']} "
              f"(test_mse={best_gurobi['test_mse']:.4f}, features={best_gurobi['number_of_selected_features']})")
    
    # Best IHT
    iht_results = results_df[results_df["method"] == "IHT"]
    if len(iht_results) > 0:
        best_iht_idx = iht_results["test_mse"].idxmin()
        best_iht = results_df.loc[best_iht_idx]
        summary_data["Best IHT"] = {
            "method": "IHT",
            "K": best_iht["K"],
            "test_mse": best_iht["test_mse"],
            "number_of_selected_features": best_iht["number_of_selected_features"]
        }
        print(f"Best IHT: K={best_iht['K']} "
              f"(test_mse={best_iht['test_mse']:.4f}, features={best_iht['number_of_selected_features']})")
    
    # Best LASSO
    lasso_results = results_df[results_df["method"] == "LASSO"]
    if len(lasso_results) > 0:
        best_lasso_idx = lasso_results["test_mse"].idxmin()
        best_lasso = results_df.loc[best_lasso_idx]
        summary_data["Best LASSO"] = {
            "method": "LASSO",
            "alpha": best_lasso["alpha"],
            "test_mse": best_lasso["test_mse"],
            "number_of_selected_features": best_lasso["number_of_selected_features"]
        }
        print(f"Best LASSO: alpha={best_lasso['alpha']} "
              f"(test_mse={best_lasso['test_mse']:.4f}, features={best_lasso['number_of_selected_features']})")
    
    # OLS Baseline
    ols_results = results_df[results_df["method"] == "OLS"]
    if len(ols_results) > 0:
        ols = ols_results.iloc[0]
        summary_data["OLS Baseline"] = {
            "method": "OLS",
            "test_mse": ols["test_mse"],
            "number_of_selected_features": ols["number_of_selected_features"]
        }
        print(f"OLS Baseline: test_mse={ols['test_mse']:.4f}, features={ols['number_of_selected_features']}")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data).T
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'regression_communities_summary.csv')
    summary_df.to_csv(summary_path)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_experiments()
