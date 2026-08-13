"""Test script for Communities and Crime regression data loading.

This script validates that the load_communities_crime_data function correctly
loads and preprocesses the UCI Communities and Crime dataset (ID: 183).
This is a large, real-world regression dataset used for testing sparse regression
methods on a realistic high-dimensional problem with missing values.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path so we can import the data loading utilities
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from regression_data import load_communities_crime_data


def main():
    """Load Communities and Crime dataset and validate structure."""
    print("=" * 80)
    print("TESTING COMMUNITIES AND CRIME REGRESSION DATA LOADING")
    print("=" * 80)
    print()

    # Load the dataset
    print("Loading Communities and Crime dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_communities_crime_data(
        test_size=0.2,
        random_state=42,
        standardize=True,
    )
    print("✓ Dataset loaded successfully")
    print()

    # Print dataset shapes
    print("Dataset Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    print()

    # Print feature information
    print("Feature Information:")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  First 20 feature names:")
    for i, name in enumerate(feature_names[:20], 1):
        print(f"    {i:2d}. {name}")
    print()

    # Print first 5 rows of X_train
    print("First 5 rows of X_train:")
    print(X_train[:5])
    print()

    # Print first 5 values of y_train
    print("First 5 values of y_train:")
    print(y_train[:5])
    print()

    # Validation checks
    print("Validation Checks:")
    all_passed = True

    # Check 1: X_train rows match y_train length
    if X_train.shape[0] != len(y_train):
        print(f"  ✗ X_train rows ({X_train.shape[0]}) != y_train length ({len(y_train)})")
        all_passed = False
    else:
        print(f"  ✓ X_train rows ({X_train.shape[0]}) match y_train length ({len(y_train)})")

    # Check 2: X_test rows match y_test length
    if X_test.shape[0] != len(y_test):
        print(f"  ✗ X_test rows ({X_test.shape[0]}) != y_test length ({len(y_test)})")
        all_passed = False
    else:
        print(f"  ✓ X_test rows ({X_test.shape[0]}) match y_test length ({len(y_test)})")

    # Check 3: X_train has no NaN values
    nan_count_train = np.isnan(X_train).sum()
    if nan_count_train > 0:
        print(f"  ✗ X_train contains {nan_count_train} NaN values")
        all_passed = False
    else:
        print(f"  ✓ X_train has no NaN values (imputation successful)")

    # Check 4: X_test has no NaN values
    nan_count_test = np.isnan(X_test).sum()
    if nan_count_test > 0:
        print(f"  ✗ X_test contains {nan_count_test} NaN values")
        all_passed = False
    else:
        print(f"  ✓ X_test has no NaN values (imputation successful)")

    print()

    # Final result
    if all_passed:
        print("=" * 80)
        print("✓ Communities and Crime data loading works")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("✗ Some validation checks failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
