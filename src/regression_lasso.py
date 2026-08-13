"""
Baseline regression methods: OLS and LASSO.

This module provides two baseline methods for regression problems:
- OLS (Ordinary Least Squares): No sparsity baseline
- LASSO (L1 regularization): L1 relaxation baseline for cardinality constraints

Note: LASSO uses L1 regularization to induce sparsity but does NOT directly
enforce an exact cardinality constraint. The cardinality is determined by the
alpha parameter and may not match a specific target cardinality.
"""

import time
from typing import Dict, List, Any
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error


def solve_ols_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Solve ordinary least squares (OLS) regression.
    
    OLS is the no-sparsity baseline that fits all features without any
    regularization or feature selection constraints.
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
    
    Returns:
        Dictionary containing:
            - method: str, "OLS"
            - train_mse: float, mean squared error on training set
            - test_mse: float, mean squared error on test set
            - number_of_selected_features: int, number of features with |coeff| > 1e-8
            - selected_features: list, indices of selected features
            - coefficients: np.ndarray, fitted coefficients
            - intercept: float, fitted intercept
            - solve_time: float, computation time in seconds
            - status: str, "success" or error message
    """
    start_time = time.time()
    
    try:
        # Fit OLS regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Compute predictions and MSE
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Select features based on coefficient magnitude threshold
        coefficients = model.coef_
        threshold = 1e-8
        selected_features = np.where(np.abs(coefficients) > threshold)[0].tolist()
        number_of_selected_features = len(selected_features)
        
        solve_time = time.time() - start_time
        
        return {
            "method": "OLS",
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "number_of_selected_features": number_of_selected_features,
            "selected_features": selected_features,
            "coefficients": coefficients,
            "intercept": float(model.intercept_),
            "solve_time": solve_time,
            "status": "success",
        }
    
    except Exception as e:
        solve_time = time.time() - start_time
        return {
            "method": "OLS",
            "train_mse": None,
            "test_mse": None,
            "number_of_selected_features": None,
            "selected_features": None,
            "coefficients": None,
            "intercept": None,
            "solve_time": solve_time,
            "status": f"error: {str(e)}",
        }


def solve_lasso_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """
    Solve LASSO (Least Absolute Shrinkage and Selection Operator) regression.
    
    LASSO is an L1 relaxation baseline for cardinality-constrained regression.
    It uses L1 regularization to induce sparsity in the solution. However,
    LASSO does NOT directly enforce an exact cardinality constraint - the
    number of selected features depends on the alpha parameter and the data.
    
    Args:
        X_train: Training features (n_train, n_features)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
        alpha: float, regularization strength (default: 0.1)
               Higher alpha leads to more regularization and fewer selected features
    
    Returns:
        Dictionary containing:
            - method: str, "LASSO"
            - alpha: float, regularization parameter used
            - train_mse: float, mean squared error on training set
            - test_mse: float, mean squared error on test set
            - number_of_selected_features: int, number of features with |coeff| > 1e-8
            - selected_features: list, indices of selected features
            - coefficients: np.ndarray, fitted coefficients
            - intercept: float, fitted intercept
            - solve_time: float, computation time in seconds
            - status: str, "success" or error message
    """
    start_time = time.time()
    
    try:
        # Fit LASSO regression
        model = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
        model.fit(X_train, y_train)
        
        # Compute predictions and MSE
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Select features based on coefficient magnitude threshold
        coefficients = model.coef_
        threshold = 1e-8
        selected_features = np.where(np.abs(coefficients) > threshold)[0].tolist()
        number_of_selected_features = len(selected_features)
        
        solve_time = time.time() - start_time
        
        return {
            "method": "LASSO",
            "alpha": alpha,
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "number_of_selected_features": number_of_selected_features,
            "selected_features": selected_features,
            "coefficients": coefficients,
            "intercept": float(model.intercept_),
            "solve_time": solve_time,
            "status": "success",
        }
    
    except Exception as e:
        solve_time = time.time() - start_time
        return {
            "method": "LASSO",
            "alpha": alpha,
            "train_mse": None,
            "test_mse": None,
            "number_of_selected_features": None,
            "selected_features": None,
            "coefficients": None,
            "intercept": None,
            "solve_time": solve_time,
            "status": f"error: {str(e)}",
        }
