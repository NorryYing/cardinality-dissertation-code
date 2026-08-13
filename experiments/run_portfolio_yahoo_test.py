"""Run a portfolio optimization prototype on Yahoo Finance data.

This script downloads a small set of stocks, builds the mean return vector and
covariance matrix, solves both the dense and sparse portfolio problems, and
writes a comparison table to the results directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_data import load_yahoo_portfolio_data
from src.portfolio_gurobi import (
    solve_portfolio_cardinality,
    solve_portfolio_no_sparsity,
)
from src.portfolio_heuristics import (
    genetic_algorithm_portfolio,
    simulated_annealing_portfolio,
)


def _run_comparison(tickers: List[str], K: int, start_date: str, end_date: str, output_name: str) -> None:
    _, _, mu, sigma, _ = load_yahoo_portfolio_data(
        tickers,
        start_date=start_date,
        end_date=end_date,
    )

    target_return = float(mu.mean())

    dense_result = solve_portfolio_no_sparsity(
        mu,
        sigma,
        target_return=target_return,
        time_limit=600,
    )
    sparse_result = solve_portfolio_cardinality(
        mu,
        sigma,
        K=K,
        target_return=target_return,
        time_limit=600,
    )
    ga_result = genetic_algorithm_portfolio(
        mu,
        sigma,
        K=K,
        target_return=target_return,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        time_limit_seconds=600,
    )
    sa_result = simulated_annealing_portfolio(
        mu,
        sigma,
        K=K,
        target_return=target_return,
        iterations=1000,
        initial_temp=1.0,
        cooling_rate=0.995,
        time_limit_seconds=600,
    )

    rows = []
    for method_name, result in [
        ("no_sparsity", dense_result),
        ("cardinality", sparse_result),
        ("genetic_algorithm", ga_result),
        ("simulated_annealing", sa_result),
    ]:
        selected_indices = result.get("selected_assets", [])
        selected_tickers = [tickers[idx] for idx in selected_indices if 0 <= idx < len(tickers)]
        rows.append(
            {
                "method": method_name,
                "variance": result["variance"],
                "risk": result["risk"],
                "return": result["return"],
                "number_of_selected_assets": len(selected_indices),
                "selected_indices": selected_indices,
                "selected_tickers": selected_tickers,
                "solve_time": result.get("solve_time", 0.0),
                "mip_gap": result.get("mip_gap") if method_name == "cardinality" else None,
            }
        )

    comparison_df = pd.DataFrame(rows)
    print(f"\n=== {output_name} ===")
    print(comparison_df.to_string(index=False))

    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved comparison table to {output_path}")


def _run_repeated_heuristics(tickers: List[str], K: int, mu, sigma, output_dir: Path) -> None:
    target_return = float(mu.mean())
    run_rows = []
    summary_rows = []

    for method_name, runner, params in [
        (
            "genetic_algorithm",
            genetic_algorithm_portfolio,
            {
                "population_size": 20,
                "generations": 20,
                "mutation_rate": 0.1,
                "time_limit_seconds": 600,
            },
        ),
        (
            "simulated_annealing",
            simulated_annealing_portfolio,
            {
                "iterations": 100,
                "initial_temp": 1.0,
                "cooling_rate": 0.995,
                "time_limit_seconds": 600,
            },
        ),
    ]:
        variances = []
        risks = []
        runtimes = []
        best_tickers = None
        best_variance = None
        best_risk = None

        for seed in range(10):
            result = runner(
                mu,
                sigma,
                K=K,
                target_return=target_return,
                seed=seed,
                **params,
            )
            selected_indices = result.get("selected_assets", [])
            selected_tickers = [tickers[idx] for idx in selected_indices if 0 <= idx < len(tickers)]
            variance = float(result["variance"])
            risk = float(result["risk"])
            runtime = float(result.get("solve_time", 0.0))

            variances.append(variance)
            risks.append(risk)
            runtimes.append(runtime)
            run_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "variance": variance,
                    "risk": risk,
                    "selected_indices": selected_indices,
                    "selected_tickers": selected_tickers,
                    "runtime": runtime,
                }
            )

            if best_variance is None or variance < best_variance:
                best_variance = variance
                best_risk = risk
                best_tickers = selected_tickers

        summary_rows.append(
            {
                "method": method_name,
                "best_variance": best_variance,
                "average_variance": float(np.mean(variances)),
                "std_variance": float(np.std(variances)),
                "best_risk": best_risk,
                "average_risk": float(np.mean(risks)),
                "average_runtime": float(np.mean(runtimes)),
                "best_selected_tickers": best_tickers,
            }
        )

    heuristic_runs_df = pd.DataFrame(run_rows)
    summary_df = pd.DataFrame(summary_rows)

    heuristic_runs_df.to_csv(output_dir / "portfolio_yahoo_20stocks_heuristic_runs.csv", index=False)
    summary_df.to_csv(output_dir / "portfolio_yahoo_20stocks_summary.csv", index=False)

    print("\n=== 20-stock heuristic summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== 20-stock heuristic runs ===")
    print(heuristic_runs_df.to_string(index=False))


def main() -> None:
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    five_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    twenty_tickers = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "UNH",
        "HD",
        "PG",
        "MA",
        "DIS",
        "BAC",
        "XOM",
        "KO",
        "PFE",
        "CSCO",
        "PEP",
    ]

    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_comparison(five_tickers, K=3, start_date=start_date, end_date=end_date, output_name="portfolio_yahoo_test.csv")

    _, _, mu, sigma, _ = load_yahoo_portfolio_data(
        twenty_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    target_return = float(mu.mean())

    dense_result = solve_portfolio_no_sparsity(
        mu,
        sigma,
        target_return=target_return,
        time_limit=600,
    )
    sparse_result = solve_portfolio_cardinality(
        mu,
        sigma,
        K=5,
        target_return=target_return,
        time_limit=600,
    )
    rows = []
    for method_name, result in [
        ("no_sparsity", dense_result),
        ("cardinality", sparse_result),
    ]:
        selected_indices = result.get("selected_assets", [])
        selected_tickers = [twenty_tickers[idx] for idx in selected_indices if 0 <= idx < len(twenty_tickers)]
        rows.append(
            {
                "method": method_name,
                "variance": result["variance"],
                "risk": result["risk"],
                "return": result["return"],
                "number_of_selected_assets": len(selected_indices),
                "selected_indices": selected_indices,
                "selected_tickers": selected_tickers,
                "solve_time": result.get("solve_time", 0.0),
                "mip_gap": result.get("mip_gap") if method_name == "cardinality" else None,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "portfolio_yahoo_20stocks.csv", index=False)
    print("\n=== 20-stock baseline comparison ===")
    print(comparison_df.to_string(index=False))

    _run_repeated_heuristics(twenty_tickers, K=5, mu=mu, sigma=sigma, output_dir=output_dir)


if __name__ == "__main__":
    main()
