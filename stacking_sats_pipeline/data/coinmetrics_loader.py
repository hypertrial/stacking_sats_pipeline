"""
CoinMetrics data loader for BTC price data.
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


class CoinMetricsLoader:
    """Loader for CoinMetrics BTC data."""

    BASE_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    DEFAULT_FILENAME = "btc_coinmetrics.csv"

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize CoinMetrics loader.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory to store/load CSV files. If None, uses current file's parent directory.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)

    def load_from_web(self) -> pd.DataFrame:
        """
        Download CoinMetrics' BTC daily time-series directly into memory.

        Returns
        -------
        pd.DataFrame
            DataFrame with BTC data, indexed by datetime.
        """
        logging.info("Downloading BTC data from %s", self.BASE_URL)

        try:
            resp = requests.get(self.BASE_URL, timeout=30)
            resp.raise_for_status()

            # Process data directly in memory
            btc_df = pd.read_csv(StringIO(resp.text), low_memory=False)
            btc_df["time"] = pd.to_datetime(btc_df["time"]).dt.normalize()
            btc_df["time"] = btc_df["time"].dt.tz_localize("UTC")
            btc_df.set_index("time", inplace=True)

            # Remove duplicates and sort
            btc_df = btc_df.loc[~btc_df.index.duplicated(keep="last")].sort_index()

            logging.info("Loaded CoinMetrics BTC data into memory (%d rows)", len(btc_df))
            self._validate_data(btc_df)

            return btc_df

        except Exception as e:
            logging.error("Failed to download CoinMetrics BTC data: %s", e)
            raise

    def extract_to_csv(self, local_path: str | Path | None = None) -> Path:
        """
        Download CoinMetrics' BTC daily time‑series and store them locally as CSV.

        Parameters
        ----------
        local_path : str or Path, optional
            Destination CSV path. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        Path
            Path where the data was saved.
        """
        if local_path is None:
            local_path = self.data_dir / self.DEFAULT_FILENAME
        else:
            local_path = Path(local_path)

        # Use the in-memory loader and save to CSV
        btc_df = self.load_from_web()
        btc_df.to_csv(local_path)
        logging.info("Saved CoinMetrics BTC data ➜ %s", local_path)

        return local_path

    def extract_to_parquet(self, local_path: str | Path | None = None) -> Path:
        """
        Download CoinMetrics' BTC daily time‑series and store them locally as Parquet.

        Parameters
        ----------
        local_path : str or Path, optional
            Destination Parquet path. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        Path
            Path where the data was saved.
        """
        if local_path is None:
            # Change extension from .csv to .parquet
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            local_path = self.data_dir / parquet_filename
        else:
            local_path = Path(local_path)

        # Use the in-memory loader and save to Parquet
        btc_df = self.load_from_web()
        btc_df.to_parquet(local_path)
        logging.info("Saved CoinMetrics BTC data ➜ %s", local_path)

        return local_path

    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load CoinMetrics BTC data from a local CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the CSV file. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with BTC data, indexed by datetime.
        """
        if path is None:
            path = self.data_dir / self.DEFAULT_FILENAME
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "CoinMetrics data file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_csv(path)

        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        # Convert naive datetime index to UTC timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        self._validate_data(df)
        return df

    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load CoinMetrics BTC data from a local Parquet file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the Parquet file. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with BTC data, indexed by datetime.
        """
        if path is None:
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            path = self.data_dir / parquet_filename
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "CoinMetrics Parquet file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_parquet(path)

        df = pd.read_parquet(path)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        # Convert naive datetime index to UTC timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        self._validate_data(df)
        return df

    def load(
        self,
        use_memory: bool = True,
        path: str | Path | None = None,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Load CoinMetrics BTC data either from memory (web) or from a local file.

        Parameters
        ----------
        use_memory : bool, default True
            If True, loads data directly from web into memory.
            If False, loads from local file (downloads if doesn't exist).
        path : str or Path, optional
            Path to the file. Only used if use_memory=False.
        file_format : str, default "csv"
            File format to use when use_memory=False. Options: "csv", "parquet".

        Returns
        -------
        pd.DataFrame
            DataFrame with BTC data, indexed by datetime.
        """
        if use_memory:
            logging.info("Loading CoinMetrics BTC data directly from web...")
            return self.load_from_web()
        else:
            if file_format.lower() == "parquet":
                return self.load_from_parquet(path)
            else:
                return self.load_from_file(path)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Basic sanity‑check on the CoinMetrics dataframe.
        """
        if df.empty or "PriceUSD" not in df.columns:
            raise ValueError("Invalid CoinMetrics BTC data – 'PriceUSD' column missing.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex.")
        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware.")
        # Check if timezone is UTC (accept both pytz.UTC and pandas UTC)
        if str(df.index.tz) != "UTC":
            raise ValueError("DatetimeIndex must be in UTC timezone.")


# Convenience functions for backward compatibility
def load_btc_data_from_web() -> pd.DataFrame:
    """Load CoinMetrics BTC data from web (backward compatibility)."""
    loader = CoinMetricsLoader()
    return loader.load_from_web()


def extract_btc_data_to_csv(local_path: str | Path | None = None) -> None:
    """Extract CoinMetrics BTC data to CSV (backward compatibility)."""
    loader = CoinMetricsLoader()
    loader.extract_to_csv(local_path)


def extract_btc_data_to_parquet(local_path: str | Path | None = None) -> None:
    """Extract CoinMetrics BTC data to Parquet (backward compatibility)."""
    loader = CoinMetricsLoader()
    loader.extract_to_parquet(local_path)
