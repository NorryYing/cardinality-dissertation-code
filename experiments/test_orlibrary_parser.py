"""Smoke test for the OR-Library portfolio parser.

This script loads the sample OR-Library portfolio file and validates that the
parsed mean-return vector and covariance matrix have the expected structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_data import load_orlibrary_portfolio


def main() -> None:
    data_path = ROOT / "data" / "raw" / "orlibrary" / "port1.txt"
    mu, sigma, asset_names = load_orlibrary_portfolio(data_path)

    print(f"Number of assets: {len(mu)}")
    print("First 5 mean returns:", mu[:5])
    print("First 5 diagonal values of Sigma:", np.diag(sigma)[:5])

    is_symmetric = bool(np.allclose(sigma, sigma.T))
    positive_diagonals = bool(np.all(np.diag(sigma) > 0))

    print("Sigma is symmetric:", is_symmetric)
    print("All diagonal values of Sigma are positive:", positive_diagonals)

    if is_symmetric and positive_diagonals:
        print("OR-Library parser works")


if __name__ == "__main__":
    import numpy as np

    main()
