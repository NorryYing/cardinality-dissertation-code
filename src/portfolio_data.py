"""Utilities for loading and preparing portfolio price data from Yahoo Finance.

This module downloads historical market data, filters to the close-price
series, removes assets with missing observations, and derives daily returns
along with the mean return vector and covariance matrix used by the
portfolio optimization models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def load_yahoo_portfolio_data(tickers, start_date, end_date):
    """Download Yahoo Finance price data and prepare portfolio inputs.

    Parameters
    ----------
    tickers : list[str] | str
        Ticker symbols to download.
    start_date : str
        Start date for the historical sample.
    end_date : str
        End date for the historical sample.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, list[str]]
        A tuple containing the cleaned price DataFrame, daily return DataFrame,
        mean return vector, covariance matrix, and final ticker list.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Close" in raw_data.columns.get_level_values(0):
            price_df = raw_data["Close"]
        else:
            price_df = raw_data
    else:
        price_df = raw_data

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()

    price_df = price_df.dropna(axis=1).dropna(axis=0)
    price_df = price_df.loc[:, ~price_df.columns.duplicated()]

    returns = price_df.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.cov()

    final_tickers = list(price_df.columns)
    return price_df, returns, mu, sigma, final_tickers


def load_orlibrary_portfolio(file_path):
    """Load an OR-Library portfolio instance from a text file.

    The OR-Library format is assumed to contain:
    1. the number of assets,
    2. one mean return and one standard deviation per asset,
    3. correlation values in triplet form ``i j corr`` for each pair.

    Parameters
    ----------
    file_path : str | Path
        Path to the OR-Library portfolio file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str] | None]
        A tuple containing the expected return vector, covariance matrix, and
        optionally asset names if present in the file.
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip().split() for line in handle if line.strip()]

    if not lines:
        raise ValueError("OR-Library portfolio file is empty")

    n_assets = int(lines[0][0])
    if len(lines) < 1 + n_assets:
        raise ValueError("OR-Library portfolio file does not contain enough asset rows")

    mu = np.zeros(n_assets, dtype=float)
    std = np.zeros(n_assets, dtype=float)

    for idx in range(n_assets):
        values = lines[1 + idx]
        if len(values) >= 2:
            mu[idx] = float(values[0])
            std[idx] = float(values[1])
        elif len(values) == 1:
            mu[idx] = float(values[0])
            std[idx] = 1.0
        else:
            raise ValueError("Unexpected asset row in OR-Library file")

    corr = np.eye(n_assets, dtype=float)
    for line in lines[1 + n_assets :]:
        if len(line) == 3:
            i_idx = int(line[0]) - 1
            j_idx = int(line[1]) - 1
            value = float(line[2])
            corr[i_idx, j_idx] = value
            corr[j_idx, i_idx] = value

    sigma = corr * np.outer(std, std)

    asset_names = None
    return mu, sigma, asset_names
