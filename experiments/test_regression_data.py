"""Smoke test for the diabetes regression data loader."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_data import load_diabetes_data


def main() -> None:
    X_train, X_test, y_train, y_test, feature_names = load_diabetes_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Feature names:", feature_names)
    print("First 5 rows of X_train:")
    print(X_train[:5])
    print("First 5 values of y_train:")
    print(y_train[:5])

    train_rows_match = X_train.shape[0] == y_train.shape[0]
    test_rows_match = X_test.shape[0] == y_test.shape[0]

    print("Training rows match y_train:", train_rows_match)
    print("Test rows match y_test:", test_rows_match)

    if train_rows_match and test_rows_match:
        print("Regression data loading works")


if __name__ == "__main__":
    main()
