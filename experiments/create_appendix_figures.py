"""Create appendix figure(s) for portfolio experiments.

Input:
- results/tables/portfolio_overall_summary.csv

Output:
- results/figures/appendix_portfolio_heuristic_gap_vs_gurobi.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "tables" / "portfolio_overall_summary.csv"
OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "appendix_portfolio_heuristic_gap_vs_gurobi.png"


CATEGORIES_REQUIRED = ["Gurobi Cardinality", "Genetic Algorithm", "Simulated Annealing"]


def _load_and_prepare() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = {"experiment", "dataset", "K", "category", "variance"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in input file: {missing_cols}")

    df = df[df["experiment"] == "OR-Library"].copy()
    df = df[df["category"].isin(CATEGORIES_REQUIRED)].copy()
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
    df = df.dropna(subset=["dataset", "K", "category", "variance"])

    if df.empty:
        raise ValueError("No valid OR-Library rows found for required categories")

    pivot = (
        df.pivot_table(
            index=["dataset", "K"],
            columns="category",
            values="variance",
            aggfunc="min",
        )
        .reset_index()
    )

    for category in CATEGORIES_REQUIRED:
        if category not in pivot.columns:
            raise ValueError(f"Required category '{category}' not found in prepared data")

    gaps = pivot[["dataset", "K", "Gurobi Cardinality", "Genetic Algorithm", "Simulated Annealing"]].copy()
    gaps["gap_genetic_algorithm"] = gaps["Genetic Algorithm"] - gaps["Gurobi Cardinality"]
    gaps["gap_simulated_annealing"] = gaps["Simulated Annealing"] - gaps["Gurobi Cardinality"]

    gaps = gaps.sort_values(["dataset", "K"]).reset_index(drop=True)
    return gaps


def _plot_gap(gaps: pd.DataFrame) -> None:
    labels = [f"{row.dataset} | K={int(row.K)}" for row in gaps.itertuples(index=False)]
    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.8))

    ga_vals = gaps["gap_genetic_algorithm"].tolist()
    sa_vals = gaps["gap_simulated_annealing"].tolist()

    ax.bar([i - width / 2 for i in x], ga_vals, width=width, label="Genetic Algorithm - Cardinality")
    ax.bar([i + width / 2 for i in x], sa_vals, width=width, label="Simulated Annealing - Cardinality")

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Appendix: Heuristic Variance Gap vs Gurobi Cardinality (OR-Library Full Instances)")
    ax.set_xlabel("Dataset and K")
    ax.set_ylabel("Variance Gap (Heuristic - Gurobi Cardinality)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gaps = _load_and_prepare()
    _plot_gap(gaps)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
