"""Gurobi-based portfolio optimization models.

This module contains a baseline long-only mean-variance optimizer and a
cardinality-constrained variant for sparse portfolio selection.
"""

from __future__ import annotations

import time

import numpy as np
from gurobipy import GRB, Model, quicksum


def solve_portfolio_no_sparsity(mu, Sigma, target_return=None, time_limit=600):
    """Solve the standard long-only mean-variance portfolio problem.

    Parameters
    ----------
    mu : array-like
        Mean return vector.
    Sigma : array-like
        Covariance matrix.
    target_return : float, optional
        Minimum portfolio return target.
    time_limit : float, optional
        Gurobi time limit in seconds.

    Returns
    -------
    dict
        Dictionary containing the portfolio weights, variance, risk, return,
        selected assets, and solve time.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    n_assets = len(mu)
    model = Model("portfolio_no_sparsity")
    model.setParam("TimeLimit", time_limit)

    weights = model.addVars(n_assets, lb=0.0, name="w")

    model.addConstr(quicksum(weights[i] for i in range(n_assets)) == 1.0)

    if target_return is not None:
        model.addConstr(quicksum(mu[i] * weights[i] for i in range(n_assets)) >= target_return)

    quad_expr = quicksum(Sigma[i, j] * weights[i] * weights[j] for i in range(n_assets) for j in range(n_assets))
    model.setObjective(quad_expr, GRB.MINIMIZE)

    model.optimize()
    solve_time = model.Runtime

    weight_vector = np.array([weights[i].X for i in range(n_assets)], dtype=float)
    variance = float(weight_vector @ Sigma @ weight_vector)
    risk = float(np.sqrt(max(variance, 0.0)))
    portfolio_return = float(mu @ weight_vector)
    selected_assets = np.flatnonzero(np.abs(weight_vector) > 1e-6).tolist()

    return {
        "weights": weight_vector,
        "variance": variance,
        "risk": risk,
        "return": portfolio_return,
        "selected_assets": selected_assets,
        "solve_time": solve_time,
    }


def solve_portfolio_cardinality(mu, Sigma, K, target_return=None, upper_bound=1.0, time_limit=600):
    """Solve the cardinality-constrained long-only portfolio problem.

    The model uses continuous portfolio weights and binary selection variables.
    The cardinality constraint limits the number of active assets, while the
    linking constraints ensure that a selected asset can receive a non-zero
    weight only if its binary selector is active.

    Parameters
    ----------
    mu : array-like
        Mean return vector.
    Sigma : array-like
        Covariance matrix.
    K : int
        Maximum number of selected assets.
    target_return : float, optional
        Minimum portfolio return target.
    upper_bound : float, optional
        Maximum weight allowed per selected asset.
    time_limit : float, optional
        Gurobi time limit in seconds.

    Returns
    -------
    dict
        Dictionary containing the portfolio weights, selected assets, variance,
        risk, return, solve time, MIP gap, and status.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    n_assets = len(mu)
    model = Model("portfolio_cardinality")
    model.setParam("TimeLimit", time_limit)

    weights = model.addVars(n_assets, lb=0.0, name="w")
    selected = model.addVars(n_assets, vtype=GRB.BINARY, name="z")

    model.addConstr(quicksum(weights[i] for i in range(n_assets)) == 1.0)
    model.addConstr(quicksum(selected[i] for i in range(n_assets)) <= K)

    for i in range(n_assets):
        # Cardinality is enforced through the binary selector variables.
        # The linking constraint below ensures that a positive weight can only
        # be assigned to an asset when that asset is selected.
        model.addConstr(weights[i] <= upper_bound * selected[i])

    if target_return is not None:
        model.addConstr(quicksum(mu[i] * weights[i] for i in range(n_assets)) >= target_return)

    quad_expr = quicksum(Sigma[i, j] * weights[i] * weights[j] for i in range(n_assets) for j in range(n_assets))
    model.setObjective(quad_expr, GRB.MINIMIZE)

    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        weight_vector = np.array([weights[i].X for i in range(n_assets)], dtype=float)
        selected_vector = np.array([selected[i].X for i in range(n_assets)], dtype=float)
    else:
        weight_vector = np.zeros(n_assets, dtype=float)
        selected_vector = np.zeros(n_assets, dtype=float)

    variance = float(weight_vector @ Sigma @ weight_vector)
    risk = float(np.sqrt(max(variance, 0.0)))
    portfolio_return = float(mu @ weight_vector)
    selected_assets = np.flatnonzero(selected_vector > 0.5).tolist()

    return {
        "weights": weight_vector,
        "selected_assets": selected_assets,
        "variance": variance,
        "risk": risk,
        "return": portfolio_return,
        "solve_time": solve_time,
        "mip_gap": model.MIPGap if hasattr(model, "MIPGap") else None,
        "status": model.status,
    }
