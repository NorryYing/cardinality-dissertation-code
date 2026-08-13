"""Test script for PMLB regression data loading (197_cpu_act dataset).

This script validates that the load_pmlb_regression_data function correctly
loads and preprocesses the 197_cpu_act dataset from PMLB (Penn Machine Learning
Benchmarks). This is a medium-size regression dataset used for testing sparse
regression methods after prototyping on the small Diabetes dataset.
"""

import sys
from pathlib import Path

# Add src to path so we can import the data loading utilities
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from regression_data import load_pmlb_regression_data


def main():
    """Load PMLB 197_cpu_act dataset and validate structure."""
    print("=" * 80)
    print("TESTING PMLB REGRESSION DATA LOADING (197_cpu_act)")
    print("=" * 80)
    print()

    # Load the dataset
    print("Loading 197_cpu_act dataset from PMLB...")
    X_train, X_test, y_train, y_test, feature_names = load_pmlb_regression_data(
        dataset_name="197_cpu_act",
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
    print(f"  Feature names: {feature_names}")
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

    # Check 3: Features in X match feature_names
    if X_train.shape[1] != len(feature_names):
        print(f"  ✗ X_train features ({X_train.shape[1]}) != len(feature_names) ({len(feature_names)})")
        all_passed = False
    else:
        print(f"  ✓ X_train features ({X_train.shape[1]}) match len(feature_names) ({len(feature_names)})")

    # Check 4: X_test features match X_train features
    if X_test.shape[1] != X_train.shape[1]:
        print(f"  ✗ X_test features ({X_test.shape[1]}) != X_train features ({X_train.shape[1]})")
        all_passed = False
    else:
        print(f"  ✓ X_test features ({X_test.shape[1]}) match X_train features ({X_train.shape[1]})")

    print()

    # Final result
    if all_passed:
        print("=" * 80)
        print("✓ PMLB regression data loading works")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("✗ Some validation checks failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
