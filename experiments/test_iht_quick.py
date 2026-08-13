"""
Quick test of the IHT regression implementation.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from regression_data import load_diabetes_data
from regression_iht import solve_iht_regression

# Load data
X_train, X_test, y_train, y_test, feature_names = load_diabetes_data()

# Test IHT with K=5
result = solve_iht_regression(X_train, y_train, X_test, y_test, K=5)

print("IHT Result (K=5):")
print(f"  Status: {result['status']}")
print(f"  Train MSE: {result['train_mse']:.6f}")
print(f"  Test MSE: {result['test_mse']:.6f}")
print(f"  Selected features: {result['selected_features']}")
print(f"  Iterations: {result['iterations']}")
print(f"  Solve time: {result['solve_time']:.6f}s")
