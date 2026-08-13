"""Heuristic baselines for sparse portfolio optimization.

These methods are motivated by the cardinality-constrained portfolio
optimization literature, especially the genetic and local-search based
approaches discussed by Chang et al. (2000), Heuristics for Cardinality
Constrained Portfolio Optimisation. The heuristics operate by choosing a
subset of assets, then solving a continuous mean-variance portfolio problem on
that selected subset using Gurobi rather than using equal weights.
"""

from __future__ import annotations

import random
import time
from typing import List, Sequence

import numpy as np
from gurobipy import GRB, Model, quicksum


def evaluate_selected_subset(mu, Sigma, selected_assets, target_return=None, time_limit=60):
    """Solve the continuous portfolio subproblem for a fixed asset subset.

    Parameters
    ----------
    mu : array-like
        Mean return vector for all assets.
    Sigma : array-like
        Covariance matrix for all assets.
    selected_assets : sequence[int]
        Indices of the selected stocks.
    target_return : float, optional
        Minimum portfolio return target.
    time_limit : float, optional
        Gurobi time limit in seconds.

    Returns
    -------
    dict
        Dictionary with the selected asset indices, the full weight vector,
        variance, risk, return, solve time, and solver status.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if mu.ndim != 1:
        mu = mu.reshape(-1)

    selected_assets = [int(idx) for idx in selected_assets]
    if not selected_assets:
        raise ValueError("selected_assets must contain at least one asset index")

    if any(idx < 0 or idx >= len(mu) for idx in selected_assets):
        raise IndexError("selected_assets contains an invalid asset index")

    n_assets = len(mu)
    selected_idx = np.array(selected_assets, dtype=int)
    sub_mu = mu[selected_idx]
    sub_sigma = Sigma[np.ix_(selected_idx, selected_idx)]

    model = Model("portfolio_subproblem")
    model.setParam("TimeLimit", time_limit)

    weights = model.addVars(len(selected_idx), lb=0.0, name="w")
    model.addConstr(quicksum(weights[i] for i in range(len(selected_idx))) == 1.0)

    if target_return is not None:
        model.addConstr(quicksum(sub_mu[i] * weights[i] for i in range(len(selected_idx))) >= target_return)

    quad_expr = quicksum(
        sub_sigma[i, j] * weights[i] * weights[j]
        for i in range(len(selected_idx))
        for j in range(len(selected_idx))
    )
    model.setObjective(quad_expr, GRB.MINIMIZE)

    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    full_weights = np.zeros(n_assets, dtype=float)
    if model.status == GRB.OPTIMAL:
        sub_weights = np.array([weights[i].X for i in range(len(selected_idx))], dtype=float)
        full_weights[selected_idx] = sub_weights
        variance = float(full_weights @ Sigma @ full_weights)
        risk = float(np.sqrt(max(variance, 0.0)))
        portfolio_return = float(mu @ full_weights)
        status = model.status
        feasible = True
    else:
        variance = np.inf
        risk = np.inf
        portfolio_return = float(np.nan)
        status = model.status
        feasible = False

    return {
        "selected_assets": selected_assets,
        "full_weights": full_weights,
        "variance": variance,
        "risk": risk,
        "return": portfolio_return,
        "solve_time": solve_time,
        "status": status,
        "feasible": feasible,
    }


def _subset_score(result, target_return=None):
    """Create a scalar score for ranking candidate subsets."""
    if not result.get("feasible", True):
        penalty = 1e6
        if target_return is not None and np.isfinite(result.get("return", np.nan)):
            penalty += max(0.0, target_return - result["return"]) * 1e6
        return (penalty, np.inf, np.inf)

    return (result["variance"], result["risk"], result["return"])


def _make_random_subset(n_assets, K, rng):
    """Create a random subset of size K without replacement."""
    return tuple(sorted(rng.choice(n_assets, size=K, replace=False).tolist()))


def _crossover(parent1, parent2, K, n_assets, rng):
    """Create a child subset by combining two parent subsets."""
    if not parent1 or not parent2:
        return tuple(sorted(parent1))

    crossover_point = int(rng.integers(1, max(2, K)))
    child = list(parent1[:crossover_point])

    for asset in parent2:
        if len(child) >= K:
            break
        if asset not in child:
            child.append(asset)

    if len(child) < K:
        remaining = [asset for asset in range(n_assets) if asset not in child]
        child.extend(remaining[: K - len(child)])

    return tuple(sorted(child[:K]))


def _mutate(subset, n_assets, K, rng, mutation_rate=0.1):
    """Mutate a subset by swapping a few selected assets."""
    subset_list = list(subset)
    if len(subset_list) != K:
        subset_list = subset_list[:K]

    if rng.random() < mutation_rate:
        swap_count = 1
        for _ in range(swap_count):
            selected_idx = int(rng.integers(0, K))
            unselected_candidates = [asset for asset in range(n_assets) if asset not in subset_list]
            if not unselected_candidates:
                break
            replacement = unselected_candidates[int(rng.integers(0, len(unselected_candidates)))]
            subset_list[selected_idx] = replacement

    return tuple(sorted(subset_list))


def genetic_algorithm_portfolio(mu, Sigma, K, target_return=None, population_size=50, generations=100, mutation_rate=0.1, time_limit_seconds=600, seed=None):
    """Run a genetic algorithm for sparse portfolio selection.

    The population is composed of random subsets of size K. Each subset is
    evaluated by solving the continuous long-only subproblem on the selected
    assets. The best subsets survive across generations using crossover and
    mutation.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n_assets = len(mu)

    if K <= 0 or K > n_assets:
        raise ValueError("K must be positive and not exceed the number of assets")

    rng = np.random.default_rng(42 if seed is None else seed)
    population = [_make_random_subset(n_assets, K, rng) for _ in range(population_size)]
    evaluated_population = []
    start_time = time.time()
    best_result = None
    best_score = None

    for generation in range(generations):
        if time.time() - start_time >= time_limit_seconds:
            break

        evaluated_population = []
        for subset in population:
            remaining_time = time_limit_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                break
            result = evaluate_selected_subset(
                mu,
                Sigma,
                subset,
                target_return=target_return,
                time_limit=min(60, remaining_time),
            )
            score = _subset_score(result, target_return=target_return)
            evaluated_population.append((score, subset, result))
            if best_result is None or score < best_score:
                best_result = result
                best_score = score

        evaluated_population.sort(key=lambda item: item[0])
        selected_parents = [item[1] for item in evaluated_population[: max(2, population_size // 5)]]

        new_population = []
        while len(new_population) < population_size:
            if len(selected_parents) >= 2:
                parent1 = selected_parents[int(rng.integers(0, len(selected_parents)))]
                parent2 = selected_parents[int(rng.integers(0, len(selected_parents)))]
                child = _crossover(parent1, parent2, K, n_assets, rng)
                child = _mutate(child, n_assets, K, rng, mutation_rate=mutation_rate)
            else:
                child = _make_random_subset(n_assets, K, rng)
            new_population.append(child)

        population = new_population

    if best_result is None:
        return {"selected_assets": [], "full_weights": np.zeros(n_assets), "variance": np.inf, "risk": np.inf, "return": np.nan, "solve_time": 0.0, "status": None, "feasible": False}

    return best_result


def simulated_annealing_portfolio(mu, Sigma, K, target_return=None, iterations=1000, initial_temp=1.0, cooling_rate=0.995, time_limit_seconds=600, seed=None):
    """Run simulated annealing for sparse portfolio selection.

    The method starts from a random subset of assets and iteratively swaps one
    selected asset with one unselected asset. This is a simple local-search
    heuristic baseline for cardinality-constrained portfolio optimization.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n_assets = len(mu)

    if K <= 0 or K > n_assets:
        raise ValueError("K must be positive and not exceed the number of assets")

    rng = np.random.default_rng(123 if seed is None else seed)
    current_subset = _make_random_subset(n_assets, K, rng)
    current_result = evaluate_selected_subset(mu, Sigma, current_subset, target_return=target_return, time_limit=60)
    current_score = _subset_score(current_result, target_return=target_return)

    best_subset = current_subset
    best_result = current_result
    best_score = current_score
    start_time = time.time()

    temperature = initial_temp
    for iteration in range(iterations):
        if time.time() - start_time >= time_limit_seconds:
            break

        selected_set = set(current_subset)
        unselected = [asset for asset in range(n_assets) if asset not in selected_set]
        if not unselected:
            break

        swap_asset = current_subset[int(rng.integers(0, len(current_subset)))]
        replacement = unselected[int(rng.integers(0, len(unselected)))]
        candidate_subset = list(current_subset)
        candidate_subset[candidate_subset.index(swap_asset)] = replacement
        candidate_subset = tuple(sorted(candidate_subset))

        remaining_time = time_limit_seconds - (time.time() - start_time)
        if remaining_time <= 0:
            break
        candidate_result = evaluate_selected_subset(
            mu,
            Sigma,
            candidate_subset,
            target_return=target_return,
            time_limit=min(60, remaining_time),
        )
        candidate_score = _subset_score(candidate_result, target_return=target_return)

        if candidate_score < current_score:
            accept = True
        else:
            probability = np.exp(-(candidate_score[0] - current_score[0]) / max(temperature, 1e-12))
            accept = rng.random() < probability

        if accept:
            current_subset = candidate_subset
            current_result = candidate_result
            current_score = candidate_score

            if candidate_score < best_score:
                best_subset = candidate_subset
                best_result = candidate_result
                best_score = candidate_score

        temperature *= cooling_rate

    return best_result
