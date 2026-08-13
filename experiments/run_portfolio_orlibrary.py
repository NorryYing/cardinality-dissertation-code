"""Run the OR-Library portfolio experiment under one global time budget."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GurobiError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_data import load_orlibrary_portfolio
from src.portfolio_gurobi import solve_portfolio_cardinality, solve_portfolio_no_sparsity
from src.portfolio_heuristics import genetic_algorithm_portfolio, simulated_annealing_portfolio

global_time_budget_seconds = 600
PER_GUROBI_MODEL_CAP = 60
# Main full run: use complete OR-Library instances.
MAX_ASSETS = None
HEURISTIC_REPEATS = 3
GA_GENERATIONS = 50
SA_ITERATIONS = 500

RESULTS_FILE = "portfolio_orlibrary_full_results.csv"
HEURISTIC_RUNS_FILE = "portfolio_orlibrary_full_heuristic_runs.csv"
SUMMARY_FILE = "portfolio_orlibrary_full_summary.csv"
ORLIBRARY_DATASETS = ("port1", "port2", "port3", "port4", "port5")
K_VALUES = [5, 10, 20]


def _state(start_time):
    elapsed = time.perf_counter() - start_time
    remaining = max(0.0, global_time_budget_seconds - elapsed)
    return elapsed, remaining, remaining <= 0.0


def _print_progress(dataset, K, method, start_time, stopped=False):
    elapsed, remaining, expired = _state(start_time)
    print(
        f"dataset={dataset}, K={K}, method={method}, "
        f"elapsed_time={elapsed:.2f}s, remaining_time={remaining:.2f}s, "
        f"stopped_due_to_time_budget={stopped or expired}"
    )


def _main_row(dataset, K, method, result, elapsed, remaining, stopped, metadata):
    selected = result.get("selected_assets", [])
    return {
        "dataset": dataset,
        "K": K,
        "method": method,
        "variance": result.get("variance"),
        "risk": result.get("risk"),
        "return": result.get("return"),
        "number_of_selected_assets": len(selected),
        "selected_indices": selected,
        "solve_time": result.get("solve_time", 0.0),
        "mip_gap": result.get("mip_gap"),
        "status": result.get("status"),
        "elapsed_total_time": elapsed,
        "remaining_time_when_started": remaining,
        "stopped_due_to_time_budget": stopped,
        **metadata,
    }


def _summary_row(dataset, K, method, result, variances, risks, runtimes, elapsed, remaining, stopped, metadata):
    selected = result.get("selected_assets", [])
    return {
        "dataset": dataset,
        "K": K,
        "method": method,
        "best_variance": result.get("variance"),
        "average_variance": float(np.mean(variances)),
        "std_variance": float(np.std(variances)),
        "best_risk": result.get("risk"),
        "average_risk": float(np.mean(risks)),
        "average_solve_time": float(np.mean(runtimes)),
        "best_return": result.get("return"),
        "number_of_selected_assets": len(selected),
        "best_selected_indices": selected,
        "mip_gap": result.get("mip_gap"),
        "status": result.get("status"),
        "elapsed_total_time": elapsed,
        "remaining_time_when_started": remaining,
        "stopped_due_to_time_budget": stopped,
        **metadata,
    }


def _save_results(result_rows, run_rows, summary_rows):
    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_columns = [
        "dataset", "K", "method", "variance", "risk", "return",
        "number_of_selected_assets", "selected_indices", "solve_time", "mip_gap",
        "status", "elapsed_total_time", "remaining_time_when_started",
        "stopped_due_to_time_budget", "original_number_of_assets",
        "number_of_assets_used", "reduced_instance", "reduction_rule",
    ]
    run_columns = [
        "dataset", "K", "method", "run_id", "variance", "risk", "return",
        "number_of_selected_assets", "selected_indices", "solve_time", "status",
        "elapsed_total_time", "remaining_time_when_started",
        "stopped_due_to_time_budget", "original_number_of_assets",
        "number_of_assets_used", "reduced_instance", "reduction_rule",
    ]
    summary_columns = [
        "dataset", "K", "method", "best_variance", "average_variance",
        "std_variance", "best_risk", "average_risk", "average_solve_time",
        "best_return", "number_of_selected_assets", "best_selected_indices",
        "mip_gap", "status", "elapsed_total_time", "remaining_time_when_started",
        "stopped_due_to_time_budget", "original_number_of_assets",
        "number_of_assets_used", "reduced_instance", "reduction_rule",
    ]
    pd.DataFrame(result_rows, columns=result_columns).to_csv(
        output_dir / RESULTS_FILE, index=False
    )
    pd.DataFrame(run_rows, columns=run_columns).to_csv(
        output_dir / HEURISTIC_RUNS_FILE, index=False
    )
    pd.DataFrame(summary_rows, columns=summary_columns).to_csv(
        output_dir / SUMMARY_FILE, index=False
    )


def _error_result(status, message):
    """Return a serializable result when a solver call cannot be completed."""
    return {
        "variance": np.nan,
        "risk": np.nan,
        "return": np.nan,
        "selected_assets": [],
        "solve_time": 0.0,
        "mip_gap": np.nan,
        "status": status,
        "error": str(message),
    }


def _run_heuristic(
    dataset, K, method, runner, mu, sigma, target_return, start_time,
    result_rows, run_rows, summary_rows, parameters, metadata,
):
    """Run three seeded heuristic repetitions while the global budget remains."""
    method_elapsed, method_remaining, expired = _state(start_time)
    if expired:
        _print_progress(dataset, K, method, start_time, True)
        return True

    results = []
    runtimes = []
    for run_id in range(1, HEURISTIC_REPEATS + 1):
        _, remaining, expired = _state(start_time)
        _print_progress(dataset, K, f"{method}[run_id={run_id}]", start_time, expired)
        if expired:
            break

        run_start = time.perf_counter()
        try:
            result = runner(
                mu, sigma, K=K, target_return=target_return,
                time_limit_seconds=remaining, seed=run_id - 1, **parameters
            )
        except GurobiError as error:
            result = _error_result("GUROBI_ERROR", error)
        runtime = time.perf_counter() - run_start
        result["solve_time"] = runtime
        results.append(result)
        runtimes.append(runtime)

        elapsed_after, remaining_after, stopped = _state(start_time)
        run_rows.append({
            "dataset": dataset,
            "K": K,
            "method": method,
            "run_id": run_id,
            "variance": result.get("variance"),
            "risk": result.get("risk"),
            "return": result.get("return"),
            "number_of_selected_assets": len(result.get("selected_assets", [])),
            "selected_indices": result.get("selected_assets", []),
            "solve_time": runtime,
            "status": result.get("status"),
            "elapsed_total_time": elapsed_after,
            "remaining_time_when_started": remaining,
            "stopped_due_to_time_budget": stopped,
            **metadata,
        })
        if stopped:
            break

    if not results:
        return True

    feasible = [result for result in results if np.isfinite(result.get("variance", np.inf))]
    candidates = feasible or results
    best = min(candidates, key=lambda result: result.get("variance", np.inf))
    variances = [result["variance"] for result in feasible] or [np.inf]
    risks = [result["risk"] for result in feasible] or [np.inf]
    stopped = _state(start_time)[2]

    result_rows.append(
        _main_row(dataset, K, method, best, method_elapsed, method_remaining, stopped, metadata)
    )
    summary_rows.append(
        _summary_row(
            dataset, K, method, best, variances, risks, runtimes,
            method_elapsed, method_remaining, stopped, metadata,
        )
    )
    return stopped


def _run_experiment(dataset_paths, K_values):
    start_time = time.perf_counter()
    result_rows = []
    run_rows = []
    summary_rows = []
    stopped = False

    for file_path in dataset_paths:
        if stopped:
            break
        dataset = file_path.stem
        mu, sigma, _ = load_orlibrary_portfolio(file_path)
        original_number_of_assets = len(mu)
        reduced_instance = (
            MAX_ASSETS is not None and original_number_of_assets > MAX_ASSETS
        )
        if reduced_instance:
            print(
                f"Dataset {dataset} has {original_number_of_assets} assets. "
                f"Reducing to first {MAX_ASSETS} assets for quick testing."
            )
            mu = mu[:MAX_ASSETS]
            sigma = sigma[:MAX_ASSETS, :MAX_ASSETS]
            reduction_rule = f"first_{MAX_ASSETS}_assets"
        else:
            reduction_rule = "none"
        metadata = {
            "original_number_of_assets": original_number_of_assets,
            "number_of_assets_used": len(mu),
            "reduced_instance": reduced_instance,
            "reduction_rule": reduction_rule,
        }
        target_return = float(mu.mean())

        for K in K_values:
            if stopped:
                break
            for method in ("no_sparsity", "cardinality", "genetic_algorithm", "simulated_annealing"):
                elapsed, remaining, expired = _state(start_time)
                _print_progress(dataset, K, method, start_time, expired)
                if expired:
                    stopped = True
                    break

                if K > len(mu):
                    result = _error_result("SKIPPED_K_TOO_LARGE", f"K={K} exceeds {len(mu)} assets")
                    result_rows.append(_main_row(dataset, K, method, result, elapsed, remaining, False, metadata))
                    summary_rows.append(_summary_row(
                        dataset, K, method, result, [np.nan], [np.nan], [0.0],
                        elapsed, remaining, False, metadata,
                    ))
                    _save_results(result_rows, run_rows, summary_rows)
                    continue

                if method == "no_sparsity":
                    try:
                        result = solve_portfolio_no_sparsity(
                            mu, sigma, target_return=target_return,
                            time_limit=min(remaining, PER_GUROBI_MODEL_CAP),
                        )
                    except GurobiError as error:
                        result = _error_result("GUROBI_ERROR", error)
                    variances = [result["variance"]]
                    risks = [result["risk"]]
                    runtimes = [result.get("solve_time", 0.0)]
                elif method == "cardinality":
                    try:
                        result = solve_portfolio_cardinality(
                            mu, sigma, K=K, target_return=target_return,
                            time_limit=min(remaining, PER_GUROBI_MODEL_CAP),
                        )
                    except GurobiError as error:
                        result = _error_result("GUROBI_ERROR", error)
                    variances = [result["variance"]]
                    risks = [result["risk"]]
                    runtimes = [result.get("solve_time", 0.0)]
                else:
                    if method == "genetic_algorithm":
                        runner = genetic_algorithm_portfolio
                        parameters = {
                            "population_size": 50,
                            "generations": GA_GENERATIONS,
                            "mutation_rate": 0.1,
                        }
                    else:
                        runner = simulated_annealing_portfolio
                        parameters = {
                            "iterations": SA_ITERATIONS,
                            "initial_temp": 1.0,
                            "cooling_rate": 0.995,
                        }
                    stopped = _run_heuristic(
                        dataset, K, method, runner, mu, sigma, target_return,
                        start_time, result_rows, run_rows, summary_rows, parameters,
                        metadata,
                    )
                    _save_results(result_rows, run_rows, summary_rows)
                    if stopped:
                        break
                    continue

                elapsed_after, remaining_after, stopped = _state(start_time)
                result_rows.append(
                    _main_row(dataset, K, method, result, elapsed, remaining, stopped, metadata)
                )
                summary_rows.append(
                    _summary_row(
                        dataset, K, method, result, variances, risks, runtimes,
                        elapsed, remaining, stopped, metadata,
                    )
                )
                _save_results(result_rows, run_rows, summary_rows)
                if stopped:
                    break

    _save_results(result_rows, run_rows, summary_rows)
    elapsed, remaining, expired = _state(start_time)
    print(
        f"Experiment finished: elapsed_time={elapsed:.2f}s, "
        f"remaining_time={remaining:.2f}s, "
        f"stopped_due_to_time_budget={stopped or expired}"
    )


def main():
    dataset_paths = [ROOT / "data" / "raw" / "orlibrary" / f"{name}.txt" for name in ORLIBRARY_DATASETS]
    missing = [path for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"OR-Library datasets not found: {missing}")
    _run_experiment(dataset_paths, K_VALUES)


if __name__ == "__main__":
    main()
