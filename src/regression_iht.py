"""
Iterative Hard Thresholding (IHT) for sparse linear regression.

This module implements IHT, a direct hard-thresholding method for sparse regression.
Unlike LASSO which uses L1 regularization to induce sparsity, IHT directly enforces
K-sparsity by keeping only the K largest coefficients after each gradient descent step.

IMPORTANT: IHT is a heuristic algorithm that may converge to local solutions. This
implementation includes stability improvements:
- Multiple random restarts to escape local minima
- Restricted OLS refitting on selected support to improve fit
- Safer learning rate schedule (0.5/L instead of 1/L)

IHT algorithm (per restart):
1. Center the response variable
2. Initialize coefficients (zero or random)
3. For each iteration:
   - Take a gradient descent step on least squares objective
   - Hard threshold: keep only K largest coefficients by absolute value
   - (Optional) Refit on selected support using OLS
   - Check convergence
4. Track best solution across all restarts based on training MSE

Advantages of IHT over LASSO:
- Directly controls sparsity (exactly K features)
- No need to tune regularization parameter alpha
- Can be more computationally efficient for certain problems
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def solve_iht_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    K: int,
    learning_rate: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    n_restarts: int = 10,
    refit_support: bool = True,
) -> Dict[str, Any]:
    """
    Solve sparse linear regression using Iterative Hard Thresholding (IHT) with restarts.
    
    IHT is a hard-thresholding method that directly enforces K-sparsity by keeping
    only the K largest coefficients after each gradient descent step. Unlike LASSO
    which uses L1 regularization, IHT directly controls the number of selected features.
    
    IMPORTANT: IHT is a heuristic algorithm that may converge to local solutions.
    This implementation uses multiple random restarts and optional restricted OLS
    refitting to improve stability and solution quality.
    
    Algorithm (per restart):
    1. Center y_train by subtracting its mean (store as intercept)
    2. Initialize coefficients as zero vector (first restart) or random small values
    3. Safely choose learning rate if not provided: learning_rate = 0.5/L
       where L is the largest eigenvalue of (X_train.T @ X_train) / n
    4. For each iteration:
       - Compute gradient: grad = 2/n * X_train.T @ (X_train @ beta - y_centered)
       - Gradient step: beta_new = beta - learning_rate * grad
       - Hard threshold: keep only K largest |beta_j|, zero out others
       - (Optional) Refit on selected support: fit OLS on selected features only
       - Check convergence: stop if ||beta_new - beta|| < tol
    5. Track solution quality (training MSE)
    6. Compute MSE on train/test sets using: prediction = intercept + X @ beta
    
    After all restarts, return the best solution (lowest training MSE).
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
        K: int, sparsity level (number of non-zero coefficients to keep)
        learning_rate: float, step size for gradient descent (default: None)
                       If None, automatically computed as 0.5/L where L is the
                       largest eigenvalue of the Gram matrix (safer than 1/L)
        max_iter: int, maximum number of iterations per restart (default: 1000)
        tol: float, convergence tolerance on ||beta_new - beta|| (default: 1e-6)
        n_restarts: int, number of random restarts (default: 10)
                    First restart uses zero initialization, rest use random small values
        refit_support: bool, whether to refit OLS on selected support (default: True)
                       If True, after hard thresholding, fit OLS on only the K selected
                       features to improve the fitted values on that support
    
    Returns:
        Dictionary containing:
            - method: str, "IHT"
            - K: int, sparsity constraint
            - train_mse: float, mean squared error on training set
            - test_mse: float, mean squared error on test set
            - number_of_selected_features: int, number of non-zero coefficients
            - selected_features: list, indices of non-zero coefficients
            - coefficients: np.ndarray, fitted coefficients
            - intercept: float, fitted intercept (mean of y_train)
            - solve_time: float, computation time in seconds
            - iterations: int, number of iterations to convergence
            - n_restarts: int, number of restarts performed
            - status: str, "converged" or "max_iter_reached"
    """
    
    start_time = time.time()
    
    try:
        # Convert to numpy arrays
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        
        n_train, n_features = X_train.shape
        
        # ====================================================================
        # Center response and store intercept
        # ====================================================================
        intercept = np.mean(y_train)
        y_centered = y_train - intercept
        
        # ====================================================================
        # Compute learning rate if not provided (safer: 0.5/L)
        # ====================================================================
        if learning_rate is None:
            # Compute largest eigenvalue of X_train.T @ X_train / n
            # Using 0.5/L for better stability compared to 1/L
            gram_matrix = X_train.T @ X_train / n_train
            eigenvalues = np.linalg.eigvalsh(gram_matrix)
            L = np.max(eigenvalues)
            learning_rate = 0.5 / L
        
        # ====================================================================
        # Track best solution across restarts
        # ====================================================================
        best_result = None
        best_train_mse = np.inf
        total_iterations = 0
        
        # ====================================================================
        # Multiple restarts
        # ====================================================================
        for restart_idx in range(n_restarts):
            # Initialize coefficients
            if restart_idx == 0:
                # First restart: zero initialization
                beta = np.zeros(n_features)
            else:
                # Other restarts: random small initialization
                beta = np.random.randn(n_features) * 0.01
            
            # ================================================================
            # IHT iterations (single restart)
            # ================================================================
            converged = False
            status = "max_iter_reached"
            
            for iteration in range(max_iter):
                # Gradient of MSE with respect to beta
                # MSE = 1/n * ||X @ beta - y_centered||_2^2
                # Gradient = 2/n * X.T @ (X @ beta - y_centered)
                residual = X_train @ beta - y_centered
                gradient = 2.0 / n_train * X_train.T @ residual
                
                # Gradient descent step
                beta_new = beta - learning_rate * gradient
                
                # ============================================================
                # Hard threshold: keep only K largest coefficients by absolute value
                # ============================================================
                if K > 0:
                    abs_beta = np.abs(beta_new)
                    
                    if K < n_features:
                        # Find the K-th largest absolute value using partition
                        if np.sum(abs_beta > 0) > K:
                            # Get indices that would sort the array
                            indices_sorted = np.argsort(-abs_beta)  # Sort descending
                            # Keep only top K indices
                            indices_keep = indices_sorted[:K]
                            # Zero out all others
                            mask = np.zeros(n_features, dtype=bool)
                            mask[indices_keep] = True
                            beta_new[~mask] = 0.0
                    # else: K >= n_features, so keep all coefficients
                else:
                    # K = 0 means no features selected
                    beta_new = np.zeros(n_features)
                
                # ============================================================
                # Optional: Refit OLS on selected support
                # ============================================================
                if refit_support and K > 0:
                    selected_indices = np.where(np.abs(beta_new) > 1e-10)[0]
                    
                    if len(selected_indices) > 0:
                        # Fit OLS on selected features only
                        try:
                            X_selected = X_train[:, selected_indices]
                            ols_model = LinearRegression(fit_intercept=False)
                            ols_model.fit(X_selected, y_centered)
                            
                            # Update beta with refitted values
                            beta_new[selected_indices] = ols_model.coef_
                        except Exception:
                            # If refitting fails, keep the hard-thresholded values
                            pass
                
                # ============================================================
                # Check convergence
                # ============================================================
                beta_change = np.linalg.norm(beta_new - beta)
                beta = beta_new
                
                if beta_change < tol:
                    converged = True
                    status = "converged"
                    break
            
            iterations_this_restart = iteration + 1
            total_iterations += iterations_this_restart
            
            # ================================================================
            # Evaluate this restart
            # ================================================================
            coefficients_this = beta
            y_train_pred = intercept + X_train @ coefficients_this
            train_mse_this = mean_squared_error(y_train, y_train_pred)
            
            # Track best solution
            if train_mse_this < best_train_mse:
                best_train_mse = train_mse_this
                
                # Extract solution
                selected_features = np.where(np.abs(coefficients_this) > 1e-10)[0].tolist()
                number_of_selected_features = len(selected_features)
                
                # Compute test MSE
                y_test_pred = intercept + X_test @ coefficients_this
                test_mse = mean_squared_error(y_test, y_test_pred)
                
                best_result = {
                    "method": "IHT",
                    "K": K,
                    "train_mse": float(best_train_mse),
                    "test_mse": float(test_mse),
                    "number_of_selected_features": number_of_selected_features,
                    "selected_features": selected_features,
                    "coefficients": coefficients_this,
                    "intercept": float(intercept),
                    "iterations": iterations_this_restart,
                    "status": status,
                }
        
        solve_time = time.time() - start_time
        
        # If no valid result found, return error
        if best_result is None:
            return {
                "method": "IHT",
                "K": K,
                "train_mse": None,
                "test_mse": None,
                "number_of_selected_features": None,
                "selected_features": None,
                "coefficients": None,
                "intercept": None,
                "solve_time": solve_time,
                "iterations": None,
                "n_restarts": n_restarts,
                "status": "error: no valid solution found",
            }
        
        # Add timing and restart info to best result
        best_result["solve_time"] = solve_time
        best_result["n_restarts"] = n_restarts
        
        return best_result
    
    except Exception as e:
        solve_time = time.time() - start_time
        return {
            "method": "IHT",
            "K": K,
            "train_mse": None,
            "test_mse": None,
            "number_of_selected_features": None,
            "selected_features": None,
            "coefficients": None,
            "intercept": None,
            "solve_time": solve_time,
            "iterations": None,
            "n_restarts": n_restarts,
            "status": f"error: {str(e)}",
        }
