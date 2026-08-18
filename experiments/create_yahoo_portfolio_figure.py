"""Create a dissertation figure for Yahoo Finance portfolio variance results.

Output:
- results/figures/portfolio_yahoo_variance_by_method.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "results" / "figures"
OUTPUT_PATH = FIGURES_DIR / "portfolio_yahoo_variance_by_method.png"

TITLE_FONTSIZE = 15
SUBPLOT_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
ANNOTATION_FONTSIZE = 9

METHODS = [
    "No-sparsity baseline",
    "Gurobi cardinality",
    "Genetic Algorithm",
    "Simulated Annealing",
]

DATASETS = [
    {
        "name": "Yahoo_20stocks",
        "variance": [0.000155, 0.000158, 0.000158, 0.000160],
        "risk": [0.012436, 0.012569, 0.012569, 0.012643],
        "selected_assets": [14, 5, 5, 5],
        "k_values": ["-", "5", "5", "5"],
    },
    {
        "name": "Yahoo_5stocks",
        "variance": [0.000318, 0.000323, 0.000323, 0.000323],
        "risk": [0.017825, 0.017964, 0.017964, 0.017964],
        "selected_assets": [4, 3, 3, 3],
        "k_values": ["-", "3", "3", "3"],
    },
]

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def _plot_panel(ax: plt.Axes, dataset: dict[str, object]) -> None:
    variances = dataset["variance"]
    bars = ax.bar(METHODS, variances, color=COLORS, edgecolor="black", linewidth=0.6)

    ax.set_title(str(dataset["name"]), fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_ylabel("Portfolio Variance", fontsize=AXIS_LABEL_FONTSIZE)
    ax.ticklabel_format(axis="y", style="plain")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, variance in zip(bars, variances):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{variance:.6f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

    for ax, dataset in zip(axes, DATASETS):
        _plot_panel(ax, dataset)

    fig.suptitle("Yahoo Finance: Portfolio Variance by Method", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()