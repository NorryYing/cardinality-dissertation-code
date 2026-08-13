"""
Gurobi-based solver for sparse linear regression / best subset selection.

This module implements a compact hard cardinality formulation for best subset
selection using Gurobi's MIQP (Mixed-Integer Quadratic Programming) solver.
Unlike LASSO which uses L1 regularization, this formulation directly enforces
an exact cardinality constraint (at most K selected features).

This formulation is optimized for the size-limited Gurobi license by:
- Avoiding residual variables (one for each training observation)
- Directly expressing the least squares objective as a quadratic function
- Keeping the number of variables small: n_features + 1 continuous + n_features binary

The sparse regression problem is:
    minimize: sum_i (y_i - intercept - sum_j X_ij * beta_j)^2
    subject to: at most K features selected
                big-M linking constraints
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def solve_best_subset_gurobi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    K: int,
    time_limit: float = 60,
    big_m: float = 1000,
) -> Dict[str, Any]:
    """
    Solve sparse linear regression using Gurobi MIQP with compact formulation.
    
    This function solves the best subset selection problem:
        minimize: ||y - X*beta - intercept||_2^2
        subject to: at most K features selected
    
    This is a direct hard cardinality formulation (not a relaxation like LASSO).
    The number of selected features is enforced exactly to be <= K.
    
    COMPACT MODEL FORMULATION (optimized for size-limited license):
    - Decision variables (only 2*n_features + 1 variables):
      * beta_j: continuous coefficient for feature j (n_features variables)
      * z_j: binary indicator, 1 if feature j is selected (n_features binary vars)
      * intercept: continuous intercept (1 variable)
    
    - Objective: minimize sum_i (y_i - intercept - sum_j X_ij * beta_j)^2
      (direct quadratic expression, NO residual variables)
    
    - Constraints:
      1. beta_j <= big_m * z_j  (big-M upper bound)
      2. beta_j >= -big_m * z_j (big-M lower bound)
      3. sum_j z_j <= K         (cardinality constraint)
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
        K: int, maximum number of features to select (cardinality constraint)
        time_limit: float, Gurobi time limit in seconds (default: 60)
        big_m: float, big-M constant for linking constraints (default: 1000)
    
    Returns:
        Dictionary containing:
            - method: str, "Best Subset (Gurobi)"
            - K: int, cardinality constraint value
            - train_mse: float, mean squared error on training set
            - test_mse: float, mean squared error on test set
            - number_of_selected_features: int, number of selected features
            - selected_features: list, indices of selected features
            - coefficients: np.ndarray, fitted coefficients
            - intercept: float, fitted intercept
            - solve_time: float, computation time in seconds
            - mip_gap: float, MIP optimality gap (if available)
            - status: str, "optimal", "suboptimal", "time_limit", error message, etc.
    """
    
    start_time = time.time()
    
    # Check if Gurobi is available
    if not GUROBI_AVAILABLE:
        solve_time = time.time() - start_time
        return {
            "method": "Best Subset (Gurobi)",
            "K": K,
            "train_mse": None,
            "test_mse": None,
            "number_of_selected_features": None,
            "selected_features": None,
            "coefficients": None,
            "intercept": None,
            "solve_time": solve_time,
            "mip_gap": None,
            "status": "error: Gurobi not installed",
        }
    
    try:
        n_train, n_features = X_train.shape
        
        # Create Gurobi model
        model = gp.Model("best_subset_regression")
        model.Params.TimeLimit = time_limit
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # ============================================================================
        # Decision variables (compact formulation - no residual variables)
        # ============================================================================
        # Continuous coefficient variables (n_features)
        beta = model.addMVar(shape=n_features, lb=-GRB.INFINITY, name="beta")
        
        # Binary selection variables (n_features)
        z = model.addMVar(shape=n_features, vtype=GRB.BINARY, name="z")
        
        # Intercept variable (1)
        intercept = model.addVar(lb=-GRB.INFINITY, name="intercept")
        
        # ============================================================================
        # Objective: minimize least squares error directly
        # ============================================================================
        # minimize sum_i (y_i - intercept - sum_j X_ij * beta_j)^2
        # This is expressed as a quadratic function without residual variables
        
        # Build the objective as sum of squared residuals
        # Using nested quicksum with multiplication
        obj_expr = gp.quicksum(
            (y_train[i] - intercept - gp.quicksum(X_train[i, j] * beta[j] for j in range(n_features))) *
            (y_train[i] - intercept - gp.quicksum(X_train[i, j] * beta[j] for j in range(n_features)))
            for i in range(n_train)
        )
        
        model.setObjective(obj_expr)
        
        # ============================================================================
        # Constraints
        # ============================================================================
        
        # Big-M constraints to link beta_j to z_j
        # If z_j = 0, then beta_j must be 0 (enforced by big-M)
        # If z_j = 1, then beta_j can be nonzero (within [-big_m, big_m])
        for j in range(n_features):
            model.addConstr(beta[j] <= big_m * z[j])
            model.addConstr(beta[j] >= -big_m * z[j])
        
        # Cardinality constraint: at most K features selected
        model.addConstr(gp.quicksum(z[j] for j in range(n_features)) <= K)
        
        # ============================================================================
        # Optimize
        # ============================================================================
        model.optimize()
        
        solve_time = time.time() - start_time
        
        # ============================================================================
        # Extract results
        # ============================================================================
        
        # Check optimization status
        if model.Status == GRB.OPTIMAL:
            status_str = "optimal"
        elif model.Status == GRB.TIME_LIMIT:
            status_str = "time_limit"
        elif model.Status == GRB.SUBOPTIMAL:
            status_str = "suboptimal"
        elif model.Status == GRB.INFEASIBLE:
            status_str = "infeasible"
        elif model.Status == GRB.UNBOUNDED:
            status_str = "unbounded"
        else:
            status_str = f"status_code_{model.Status}"
        
        # Check if solution is feasible
        if model.SolCount == 0:
            # No feasible solution found
            return {
                "method": "Best Subset (Gurobi)",
                "K": K,
                "train_mse": None,
                "test_mse": None,
                "number_of_selected_features": None,
                "selected_features": None,
                "coefficients": None,
                "intercept": None,
                "solve_time": solve_time,
                "mip_gap": None,
                "status": f"no_feasible_solution ({status_str})",
            }
        
        # Extract solution from binary variables and coefficients
        beta_vals = beta.X
        intercept_val = intercept.X
        z_vals = z.X
        
        # ============================================================================
        # IMPORTANT: For hard cardinality constraints, selected features are
        # determined ONLY from the binary selection variables z_j, NOT from
        # coefficient magnitude. This ensures strict enforcement of the cardinality
        # constraint K and prevents spurious selections from rounding errors.
        # ============================================================================
        selected_features = np.where(z_vals > 0.5)[0].tolist()
        number_of_selected_features = len(selected_features)
        
        # ============================================================================
        # Enforce zero coefficients for non-selected features
        # ============================================================================
        # For a hard cardinality model, features not selected by z_j must have
        # coefficient exactly zero to respect the sparsity structure.
        coefficients = np.zeros(n_features)
        for j in selected_features:
            coefficients[j] = beta_vals[j]
        
        # ============================================================================
        # VALIDATION: Verify cardinality constraint is respected
        # ============================================================================
        if number_of_selected_features > K:
            status_str = f"CARDINALITY_VIOLATION (selected={number_of_selected_features}, K={K})"
            print(f"\n⚠️  WARNING: Cardinality constraint violated!")
            print(f"   K={K} but {number_of_selected_features} features were selected")
            print(f"   Selected features: {selected_features}")
        
        # Compute predictions and MSE with the enforced coefficient structure
        y_train_pred = X_train @ coefficients + intercept_val
        y_test_pred = X_test @ coefficients + intercept_val
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Compute MIP gap
        if model.Status == GRB.OPTIMAL:
            mip_gap = 0.0
        else:
            # For suboptimal solutions, report the gap if available
            if model.MIPGap is not None and not np.isnan(model.MIPGap):
                mip_gap = float(model.MIPGap)
            else:
                mip_gap = None
        
        return {
            "method": "Best Subset (Gurobi)",
            "K": K,
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "number_of_selected_features": number_of_selected_features,
            "selected_features": selected_features,
            "coefficients": coefficients,
            "intercept": float(intercept_val),
            "solve_time": solve_time,
            "mip_gap": mip_gap,
            "status": status_str,
        }
    
    except Exception as e:
        solve_time = time.time() - start_time
        error_msg = str(e)
        
        # Identify if it's a Gurobi error
        if "gurobipy" in str(type(e).__module__) or "GurobiError" in str(type(e).__name__):
            status_str = f"GUROBI_ERROR: {error_msg}"
        else:
            status_str = f"error: {error_msg}"
        
        return {
            "method": "Best Subset (Gurobi)",
            "K": K,
            "train_mse": None,
            "test_mse": None,
            "number_of_selected_features": None,
            "selected_features": None,
            "coefficients": None,
            "intercept": None,
            "solve_time": solve_time,
            "mip_gap": None,
            "status": status_str,
        }
