"""
Create an overall summary combining results from all three regression datasets.

This script rebuilds the summary from the detailed result files:
1. Diabetes dataset
2. PMLB 197_cpu_act dataset
3. Communities and Crime dataset

It extracts 7 categories per dataset:
- Best Overall
- Best Sparse
- Best Hard Cardinality
- Best Gurobi
- Best IHT
- Best LASSO
- OLS Baseline

Output: A comprehensive comparison table across all datasets.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def extract_dataset_summary(results_df, dataset_name):
    """
    Extract summary categories from a detailed results dataframe.
    
    Args:
        results_df: DataFrame with regression results
        dataset_name: Name of the dataset
    
    Returns:
        List of tuples (category, row_data) for the 7 summary categories
    """
    summary_rows = []
    
    # Determine total number of features from OLS row
    ols_row = results_df[results_df["method"] == "OLS"]
    if len(ols_row) > 0:
        total_num_features = ols_row.iloc[0]["number_of_selected_features"]
    else:
        total_num_features = None
    
    # 1. Best Overall (lowest test_mse among all)
    best_overall_idx = results_df["test_mse"].idxmin()
    best_overall_row = results_df.loc[best_overall_idx]
    summary_rows.append(("Best Overall", best_overall_row))
    
    # 2. Best Sparse (lowest test_mse where selected_features < total_features)
    if total_num_features is not None:
        sparse_candidates = results_df[results_df["number_of_selected_features"] < total_num_features]
        if len(sparse_candidates) > 0:
            best_sparse_idx = sparse_candidates["test_mse"].idxmin()
            best_sparse_row = results_df.loc[best_sparse_idx]
            summary_rows.append(("Best Sparse", best_sparse_row))
    
    # 3. Best Hard Cardinality (lowest test_mse among Gurobi and IHT only)
    hard_card_methods = ["Best Subset (Gurobi)", "IHT"]
    hard_card_candidates = results_df[results_df["method"].isin(hard_card_methods)]
    if len(hard_card_candidates) > 0:
        best_hard_idx = hard_card_candidates["test_mse"].idxmin()
        best_hard_row = results_df.loc[best_hard_idx]
        summary_rows.append(("Best Hard Cardinality", best_hard_row))
    
    # 4. Best Gurobi (lowest test_mse among Best Subset (Gurobi))
    gurobi_results = results_df[results_df["method"] == "Best Subset (Gurobi)"]
    if len(gurobi_results) > 0:
        best_gurobi_idx = gurobi_results["test_mse"].idxmin()
        best_gurobi_row = results_df.loc[best_gurobi_idx]
        summary_rows.append(("Best Gurobi", best_gurobi_row))
    
    # 5. Best IHT (lowest test_mse among IHT)
    iht_results = results_df[results_df["method"] == "IHT"]
    if len(iht_results) > 0:
        best_iht_idx = iht_results["test_mse"].idxmin()
        best_iht_row = results_df.loc[best_iht_idx]
        summary_rows.append(("Best IHT", best_iht_row))
    
    # 6. Best LASSO (lowest test_mse among LASSO)
    lasso_results = results_df[results_df["method"] == "LASSO"]
    if len(lasso_results) > 0:
        best_lasso_idx = lasso_results["test_mse"].idxmin()
        best_lasso_row = results_df.loc[best_lasso_idx]
        summary_rows.append(("Best LASSO", best_lasso_row))
    
    # 7. OLS Baseline (the OLS row)
    if len(ols_row) > 0:
        ols_data_row = ols_row.iloc[0]
        summary_rows.append(("OLS Baseline", ols_data_row))
    
    return summary_rows


def main():
    """Create combined regression summary from detailed results."""
    
    print("=" * 100)
    print("CREATING OVERALL REGRESSION SUMMARY FROM DETAILED RESULTS")
    print("=" * 100)
    
    # Define paths to detailed result files
    diabetes_results_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tables', 'regression_diabetes_results.csv'
    )
    pmlb_results_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tables', 'regression_pmlb_197_cpu_act_results.csv'
    )
    communities_results_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tables', 'regression_communities_results.csv'
    )
    
    # Check if all files exist
    files_to_check = [
        (diabetes_results_path, "Diabetes"),
        (pmlb_results_path, "PMLB 197_cpu_act"),
        (communities_results_path, "Communities & Crime"),
    ]
    
    missing_files = []
    for path, name in files_to_check:
        if not os.path.exists(path):
            missing_files.append(f"  - {name}: {path}")
    
    if missing_files:
        print(f"\n❌ ERROR: Missing result files:")
        for msg in missing_files:
            print(msg)
        sys.exit(1)
    
    # Load detailed results
    print("\nLoading detailed result files...")
    
    print("  ✓ Loading Diabetes results...")
    df_diabetes_results = pd.read_csv(diabetes_results_path)
    print(f"    Loaded {len(df_diabetes_results)} experiments")
    
    print("  ✓ Loading PMLB 197_cpu_act results...")
    df_pmlb_results = pd.read_csv(pmlb_results_path)
    print(f"    Loaded {len(df_pmlb_results)} experiments")
    
    print("  ✓ Loading Communities & Crime results...")
    df_communities_results = pd.read_csv(communities_results_path)
    print(f"    Loaded {len(df_communities_results)} experiments")
    
    # Extract summaries for each dataset
    print("\nExtracting summary categories...")
    
    print("  ✓ Extracting Diabetes summaries...")
    diabetes_summary = extract_dataset_summary(df_diabetes_results, "Diabetes")
    print(f"    Extracted {len(diabetes_summary)} categories")
    
    print("  ✓ Extracting PMLB 197_cpu_act summaries...")
    pmlb_summary = extract_dataset_summary(df_pmlb_results, "197_cpu_act")
    print(f"    Extracted {len(pmlb_summary)} categories")
    
    print("  ✓ Extracting Communities & Crime summaries...")
    communities_summary = extract_dataset_summary(df_communities_results, "Communities_Crime")
    print(f"    Extracted {len(communities_summary)} categories")
    
    # Build combined summary dataframe
    print("\nBuilding combined summary...")
    combined_rows = []
    
    for dataset_name, dataset_summary in [
        ("Diabetes", diabetes_summary),
        ("197_cpu_act", pmlb_summary),
        ("Communities_Crime", communities_summary),
    ]:
        for category, row_data in dataset_summary:
            combined_rows.append({
                "dataset": dataset_name,
                "category": category,
                "method": row_data.get("method", ""),
                "K": int(row_data["K"]) if pd.notna(row_data.get("K")) else None,
                "alpha": row_data.get("alpha", None),
                "test_mse": row_data["test_mse"],
                "number_of_selected_features": int(row_data["number_of_selected_features"]) if pd.notna(row_data.get("number_of_selected_features")) else None,
                "selected_feature_names": row_data.get("selected_feature_names", None),
                "status": row_data.get("status", None),
            })
    
    df_combined = pd.DataFrame(combined_rows)
    print(f"  ✓ Combined table: {len(df_combined)} rows")
    
    # Select and order columns
    column_order = [
        "dataset",
        "category",
        "method",
        "K",
        "alpha",
        "test_mse",
        "number_of_selected_features",
        "selected_feature_names",
        "status",
    ]
    available_columns = [col for col in column_order if col in df_combined.columns]
    df_combined = df_combined[available_columns]
    
    # Sort by dataset and category
    df_combined = df_combined.sort_values(["dataset", "category"]).reset_index(drop=True)
    
    # Save to file
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'tables', 'regression_overall_summary.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Summary saved to: {output_path}")
    
    # Print the combined table
    print("\n" + "=" * 100)
    print("OVERALL REGRESSION SUMMARY")
    print("=" * 100)
    print()
    
    # Display in a more readable format
    for dataset in sorted(df_combined["dataset"].unique()):
        df_dataset = df_combined[df_combined["dataset"] == dataset]
        print(f"\n{dataset}:")
        print("─" * 100)
        
        # Select key columns for display
        display_cols = ["category", "method", "K", "alpha", "test_mse", "number_of_selected_features", "status"]
        available_cols = [col for col in display_cols if col in df_dataset.columns]
        
        print(df_dataset[available_cols].to_string(index=False))
    
    # Print statistics by dataset
    print("\n" + "=" * 100)
    print("SUMMARY BY DATASET")
    print("=" * 100)
    
    for dataset in sorted(df_combined["dataset"].unique()):
        df_dataset = df_combined[df_combined["dataset"] == dataset]
        print(f"\n{dataset}:")
        print(f"  Total categories: {len(df_dataset)}")
        
        # Best overall for this dataset
        best_overall_idx = df_dataset["test_mse"].idxmin()
        best_overall = df_combined.loc[best_overall_idx]
        print(f"  Best Overall: {best_overall['category']} - "
              f"{best_overall['method']} "
              f"(test_mse={best_overall['test_mse']:.6f})")
    
    print("\n" + "=" * 100)
    print("✓ OVERALL SUMMARY CREATED SUCCESSFULLY")
    print("=" * 100)


if __name__ == "__main__":
    main()
