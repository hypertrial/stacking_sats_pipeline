"""
Data extraction and loading utilities for BTC price data.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Logging configuration
# ---------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_btc_data_from_web() -> pd.DataFrame:
    """
    Download CoinMetrics' BTC daily time-series directly into memory.

    Returns
    -------
    pd.DataFrame
        DataFrame with BTC data, indexed by datetime.
    """
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    logging.info("Downloading BTC data from %s", url)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Process data directly in memory
        btc_df = pd.read_csv(StringIO(resp.text), low_memory=False)
        btc_df["time"] = pd.to_datetime(btc_df["time"]).dt.normalize()
        btc_df["time"] = btc_df["time"].dt.tz_localize(None)
        btc_df.set_index("time", inplace=True)

        # Remove duplicates and sort
        btc_df = btc_df.loc[~btc_df.index.duplicated(keep="last")].sort_index()

        logging.info("Loaded BTC data into memory (%d rows)", len(btc_df))
        validate_price_data(btc_df)

        return btc_df

    except Exception as e:
        logging.error("Failed to download BTC data: %s", e)
        raise


def extract_btc_data_to_csv(local_path: str | Path | None = None) -> None:
    """
    Download CoinMetrics' BTC daily time‑series and store them locally.

    Parameters
    ----------
    local_path : str or Path, optional
        Destination CSV path. If None, defaults to "btc_data.csv" in the data folder.
    """
    if local_path is None:
        # Default to saving in the data folder
        data_dir = Path(__file__).parent
        local_path = data_dir / "btc_data.csv"
    else:
        local_path = Path(local_path)

    # Use the in-memory loader and save to CSV
    btc_df = load_btc_data_from_web()
    btc_df.to_csv(local_path)
    logging.info("Saved BTC data ➜ %s", local_path)


def load_data(path: str | Path | None = None, use_memory: bool = True) -> pd.DataFrame:
    """
    Load BTC price data either from memory (web) or from a local CSV file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. If None and use_memory=False, defaults to "btc_data.csv"
        in the data folder. Ignored if use_memory=True.
    use_memory : bool, default True
        If True, loads data directly from web into memory.
        If False, loads from local CSV file (downloads if doesn't exist).

    Returns
    -------
    pd.DataFrame
        DataFrame with BTC data, indexed by datetime.
    """
    if use_memory:
        logging.info("Loading BTC data directly into memory from web...")
        return load_btc_data_from_web()

    # Legacy file-based loading
    if path is None:
        # Default to looking in the data folder
        data_dir = Path(__file__).parent
        path = data_dir / "btc_data.csv"
    else:
        path = Path(path)

    if not path.exists():
        logging.info("BTC data file not found. Downloading automatically...")
        extract_btc_data_to_csv(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    validate_price_data(df)
    return df


def validate_price_data(df: pd.DataFrame) -> None:
    """
    Basic sanity‑check on the input dataframe.
    """
    if df.empty or "PriceUSD" not in df.columns:
        raise ValueError("Invalid BTC price data – 'PriceUSD' column missing.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")
