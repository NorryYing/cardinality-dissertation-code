"""Create dissertation-ready regression figures from the overall summary table.

Input:
- results/tables/regression_overall_summary.csv

Outputs:
- results/figures/regression_test_mse_by_dataset_method.png
- results/figures/regression_selected_features_by_dataset_method.png
- results/figures/regression_mse_vs_sparsity.png

Notes:
- Uses only pandas and matplotlib (no seaborn).
- Collapses duplicate summary rows to one row per dataset-method.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "tables" / "regression_overall_summary.csv"
FIGURES_DIR = ROOT / "results" / "figures"

FIG_MSE = FIGURES_DIR / "regression_test_mse_by_dataset_method.png"
FIG_FEATURES = FIGURES_DIR / "regression_selected_features_by_dataset_method.png"
FIG_SCATTER = FIGURES_DIR / "regression_mse_vs_sparsity.png"

METHOD_LABELS = {
    "OLS": "OLS",
    "LASSO": "LASSO",
    "BEST SUBSET (GUROBI)": "Best Subset (Gurobi)",
    "BEST SUBSET": "Best Subset (Gurobi)",
    "GUROBI": "Best Subset (Gurobi)",
    "IHT": "IHT",
}
METHOD_ORDER = ["OLS", "LASSO", "Best Subset (Gurobi)", "IHT"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized_to_actual = {col.strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_to_actual:
            return normalized_to_actual[key]
    raise KeyError(f"Could not find any of columns: {candidates}")


def _canonical_method(value: object) -> str | None:
    text = str(value).strip().upper()
    return METHOD_LABELS.get(text)


def _load_and_prepare() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    dataset_col = _find_column(df, ["dataset"])
    method_col = _find_column(df, ["method"])
    mse_col = _find_column(df, ["test_mse", "mse", "test mse"])
    nfeat_col = _find_column(df, ["number_of_selected_features", "num_selected_features", "selected_features"])

    cleaned = pd.DataFrame(
        {
            "dataset": df[dataset_col],
            "method_raw": df[method_col],
            "test_mse": pd.to_numeric(df[mse_col], errors="coerce"),
            "number_of_selected_features": pd.to_numeric(df[nfeat_col], errors="coerce"),
        }
    )

    cleaned["method"] = cleaned["method_raw"].map(_canonical_method)
    cleaned = cleaned.dropna(subset=["dataset", "method", "test_mse", "number_of_selected_features"])

    # Remove duplicate summary categories by keeping one representative row per
    # dataset-method, preferring lower test MSE and then fewer selected features.
    cleaned = cleaned.sort_values(
        ["dataset", "method", "test_mse", "number_of_selected_features"],
        ascending=[True, True, True, True],
    )
    cleaned = cleaned.drop_duplicates(subset=["dataset", "method"], keep="first")

    cleaned["dataset"] = cleaned["dataset"].astype(str)
    cleaned["method"] = pd.Categorical(cleaned["method"], categories=METHOD_ORDER, ordered=True)
    cleaned = cleaned.sort_values(["dataset", "method"]).reset_index(drop=True)
    return cleaned


def _plot_grouped_bar(df: pd.DataFrame, y_col: str, ylabel: str, title: str, output_path: Path) -> None:
    datasets = sorted(df["dataset"].unique().tolist())
    methods_present = [m for m in METHOD_ORDER if m in df["method"].astype(str).unique().tolist()]

    x_positions = list(range(len(datasets)))
    width = 0.18
    offsets = [
        (-1.5 * width),
        (-0.5 * width),
        (0.5 * width),
        (1.5 * width),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    for idx, method in enumerate(methods_present):
        method_df = df[df["method"].astype(str) == method].set_index("dataset")
        values = [method_df[y_col].get(dataset, float("nan")) for dataset in datasets]
        bars_x = [x + offsets[idx] for x in x_positions]
        ax.bar(bars_x, values, width=width, label=method)

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_mse_vs_sparsity(df: pd.DataFrame) -> None:
    datasets = sorted(df["dataset"].unique().tolist())
    methods_present = [m for m in METHOD_ORDER if m in df["method"].astype(str).unique().tolist()]

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {dataset: markers[i % len(markers)] for i, dataset in enumerate(datasets)}
    color_map = {
        "OLS": "tab:blue",
        "LASSO": "tab:orange",
        "Best Subset (Gurobi)": "tab:green",
        "IHT": "tab:red",
    }

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Plot points by (dataset, method) and annotate with short labels.
    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        method = str(row["method"])
        x = row["number_of_selected_features"]
        y = row["test_mse"]

        ax.scatter(
            x,
            y,
            marker=marker_map[dataset],
            color=color_map.get(method, "gray"),
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.annotate(
            f"{dataset}-{method.split()[0]}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    ax.set_title("Regression: Test MSE vs. Sparsity")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Test MSE")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Build compact legends: one for methods (color), one for datasets (marker).
    method_handles = []
    for method in methods_present:
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(method, "gray"),
            markeredgecolor="black",
            markersize=7,
            label=method,
        )
        method_handles.append(handle)

    dataset_handles = []
    for dataset in datasets:
        handle = plt.Line2D(
            [0],
            [0],
            marker=marker_map[dataset],
            color="black",
            linestyle="None",
            markersize=7,
            label=dataset,
        )
        dataset_handles.append(handle)

    legend1 = ax.legend(handles=method_handles, title="Method", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, title="Dataset", loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG_SCATTER, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_and_prepare()

    if df.empty:
        raise ValueError("No usable rows found after filtering and deduplication")

    _plot_grouped_bar(
        df,
        y_col="test_mse",
        ylabel="Test MSE",
        title="Regression: Test MSE by Dataset and Method",
        output_path=FIG_MSE,
    )
    _plot_grouped_bar(
        df,
        y_col="number_of_selected_features",
        ylabel="Number of Selected Features",
        title="Regression: Selected Features by Dataset and Method",
        output_path=FIG_FEATURES,
    )
    _plot_mse_vs_sparsity(df)

    print(f"Saved figure: {FIG_MSE}")
    print(f"Saved figure: {FIG_FEATURES}")
    print(f"Saved figure: {FIG_SCATTER}")
    print(f"Rows used for plotting (after deduplication): {len(df)}")


if __name__ == "__main__":
    main()
