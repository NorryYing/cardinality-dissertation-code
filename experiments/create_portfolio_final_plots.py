"""Create dissertation figures and a compact table for portfolio results.

Inputs
------
- results/tables/portfolio_orlibrary_full_summary.csv
- results/tables/portfolio_overall_summary.csv

Outputs
-------
- results/figures/portfolio_orlibrary_variance_by_k_method.png
- results/figures/portfolio_orlibrary_selected_assets_by_k_method.png
- results/figures/portfolio_orlibrary_runtime_heatmap.png (if solve_time is available)
- results/tables/portfolio_orlibrary_compact_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

FULL_RESULTS_PATH = TABLES_DIR / "portfolio_orlibrary_full_summary.csv"
OVERALL_SUMMARY_PATH = TABLES_DIR / "portfolio_overall_summary.csv"
COMPACT_SUMMARY_PATH = TABLES_DIR / "portfolio_orlibrary_compact_summary.csv"

VARIANCE_FIG_PATH = FIGURES_DIR / "portfolio_orlibrary_variance_by_k_method.png"
SELECTED_ASSETS_FIG_PATH = FIGURES_DIR / "portfolio_orlibrary_selected_assets_by_k_method.png"
RUNTIME_FIG_PATH = FIGURES_DIR / "portfolio_orlibrary_runtime_heatmap.png"

METHOD_ORDER = ["no_sparsity", "cardinality", "genetic_algorithm", "simulated_annealing"]
CATEGORY_ORDER = [
    "No-Sparsity Baseline",
    "Gurobi Cardinality",
    "Genetic Algorithm",
    "Simulated Annealing",
    "Best Sparse",
]

TITLE_FONTSIZE = 17
SUBPLOT_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
ANNOTATION_FONTSIZE = 9


def _ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OVERALL_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing overall portfolio summary file: {OVERALL_SUMMARY_PATH}")

    overall_summary = pd.read_csv(OVERALL_SUMMARY_PATH)

    # Use OR-Library rows from the overall summary for the two main line plots.
    full_results = overall_summary[
        (overall_summary["experiment"] == "OR-Library")
        & (overall_summary["category"].isin(CATEGORY_ORDER))
    ].copy()

    return full_results, overall_summary


def _prepare_full_orlibrary(full_results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "K",
        "method",
        "variance",
        "number_of_selected_assets",
    }
    missing = sorted(required - set(full_results.columns))
    if missing:
        raise ValueError(f"Missing required columns in full results: {missing}")

    df = full_results.copy()

    # Main dissertation figures should only use full-instance OR-Library rows.
    if "reduced_instance" in df.columns:
        df = df[df["reduced_instance"] == False]  # noqa: E712
    if "reduction_rule" in df.columns:
        df = df[df["reduction_rule"] == "none"]

    if df.empty:
        raise ValueError("No full-instance rows remain after filtering reduced_instance/reduction_rule")

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
    df["number_of_selected_assets"] = pd.to_numeric(df["number_of_selected_assets"], errors="coerce")
    if "solve_time" in df.columns:
        df["solve_time"] = pd.to_numeric(df["solve_time"], errors="coerce")

    df = df.dropna(subset=["dataset", "K", "method"])
    df = df.sort_values(["dataset", "K", "method"]).reset_index(drop=True)
    return df


def _plot_variance_by_k_and_method(df: pd.DataFrame) -> None:
    datasets = sorted(df["dataset"].unique())
    ncols = 3
    nrows = (len(datasets) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.1 * ncols, 4.0 * nrows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    legend_handles = None
    legend_labels = None

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["dataset"] == dataset]
        sns.lineplot(
            data=subset,
            x="K",
            y="variance",
            hue="method",
            hue_order=METHOD_ORDER,
            marker="o",
            ax=ax,
        )
        ax.set_title(dataset, fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlabel("K", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Portfolio Variance", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

        if ax.get_legend() is not None:
            ax.get_legend().set_title("Method")
            plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_FONTSIZE)
            plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONTSIZE)
            if legend_handles is None and legend_labels is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    for idx in range(len(datasets), len(axes) - 1):
        axes[idx].axis("off")

    legend_ax = axes[-1]
    legend_ax.axis("off")
    if legend_handles is not None and legend_labels is not None:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            title="Method",
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
            frameon=True,
        )

    fig.suptitle("OR-Library: Portfolio Variance by K and Method", fontsize=TITLE_FONTSIZE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(VARIANCE_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_selected_assets_by_k_and_method(df: pd.DataFrame) -> None:
    datasets = sorted(df["dataset"].unique())
    ncols = 3
    nrows = (len(datasets) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.1 * ncols, 4.0 * nrows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    legend_handles = None
    legend_labels = None

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["dataset"] == dataset]

        sns.lineplot(
            data=subset,
            x="K",
            y="number_of_selected_assets",
            hue="method",
            hue_order=METHOD_ORDER,
            marker="o",
            ax=ax,
        )

        k_values = sorted(subset["K"].dropna().unique())
        ax.plot(k_values, k_values, linestyle="--", linewidth=1, color="black", label="selected_assets = K")

        ax.set_title(dataset, fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlabel("K", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Number of Selected Assets", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

        if ax.get_legend() is not None:
            ax.get_legend().set_title("Method")
            plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_FONTSIZE)
            plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONTSIZE)
            if legend_handles is None and legend_labels is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    for idx in range(len(datasets), len(axes) - 1):
        axes[idx].axis("off")

    legend_ax = axes[-1]
    legend_ax.axis("off")
    if legend_handles is not None and legend_labels is not None:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            title="Method",
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
            frameon=True,
        )

    fig.suptitle("OR-Library: Number of Selected Assets by K and Method", fontsize=TITLE_FONTSIZE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(SELECTED_ASSETS_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime_heatmap(df: pd.DataFrame) -> bool:
    if "solve_time" not in df.columns:
        return False

    runtime_df = df.dropna(subset=["solve_time"]).copy()
    if runtime_df.empty:
        return False

    runtime_df["dataset_K"] = runtime_df.apply(lambda r: f"{r['dataset']} | K={int(r['K'])}", axis=1)
    pivot = runtime_df.pivot_table(
        index="dataset_K",
        columns="method",
        values="solve_time",
        aggfunc="mean",
    )

    if pivot.empty:
        return False

    method_cols = [m for m in METHOD_ORDER if m in pivot.columns]
    pivot = pivot[method_cols]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Solve Time (seconds)"},
        annot_kws={"fontsize": ANNOTATION_FONTSIZE},
        ax=ax,
    )
    ax.set_title("OR-Library: Runtime by Dataset, K, and Method", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Method", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Dataset and K", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    colorbar.set_label("Solve Time (seconds)", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout()
    fig.savefig(RUNTIME_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _category_slug(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _create_compact_orlibrary_table(overall_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment",
        "dataset",
        "K",
        "category",
        "method",
        "variance",
        "risk",
        "solve_time",
        "number_of_selected_assets",
    }
    missing = sorted(required - set(overall_summary.columns))
    if missing:
        raise ValueError(f"Missing required columns in overall summary: {missing}")

    df = overall_summary.copy()
    df = df[df["experiment"] == "OR-Library"].copy()
    df = df[df["category"].isin(CATEGORY_ORDER)].copy()
    if df.empty:
        raise ValueError("No OR-Library rows found in portfolio_overall_summary.csv")

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.sort_values(["dataset", "K", "category"]).reset_index(drop=True)

    metrics = ["method", "variance", "risk", "solve_time", "number_of_selected_assets"]
    pivot = df.pivot_table(
        index=["dataset", "K"],
        columns="category",
        values=metrics,
        aggfunc="first",
    )

    pivot.columns = [f"{metric}__{_category_slug(category)}" for metric, category in pivot.columns]
    compact = pivot.reset_index().sort_values(["dataset", "K"]).reset_index(drop=True)
    compact.to_csv(COMPACT_SUMMARY_PATH, index=False)
    return compact


def main() -> None:
    _ensure_dirs()

    sns.set_theme(style="whitegrid", context="paper")

    full_results, overall_summary = _load_tables()
    orlibrary_full = _prepare_full_orlibrary(full_results)

    _plot_variance_by_k_and_method(orlibrary_full)
    _plot_selected_assets_by_k_and_method(orlibrary_full)
    runtime_created = _plot_runtime_heatmap(orlibrary_full)
    compact = _create_compact_orlibrary_table(overall_summary)

    print("Saved figure:", VARIANCE_FIG_PATH)
    print("Saved figure:", SELECTED_ASSETS_FIG_PATH)
    if runtime_created:
        print("Saved figure:", RUNTIME_FIG_PATH)
    else:
        print("Skipped runtime figure: solve_time data not available")

    print("Saved table:", COMPACT_SUMMARY_PATH)
    print("Compact table shape:", compact.shape)


if __name__ == "__main__":
    main()
