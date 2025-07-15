"""
Fear & Greed Index loader https://alternative.me/crypto/api/
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import requests

# Load environment variables if available
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Logging configuration
# ---------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class FearGreedLoader:
    """Loader for Fear & Greed Index data."""

    BASE_URL = "https://api.alternative.me/fng/"
    DEFAULT_FILENAME = "fear_greed.csv"

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize Fear & Greed Index loader.

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
        Download Fear & Greed Index daily time-series directly into memory.

        Returns
        -------
        pd.DataFrame
            DataFrame with Fear & Greed Index data, indexed by datetime.
        """
        logging.info("Downloading data from Fear & Greed Index API...")

        # Fear & Greed Index API doesn't require API key for basic usage
        params = {
            "limit": 0,  # 0 means no limit, get all available data
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()

            if "data" not in data:
                raise ValueError(f"Invalid response from Fear & Greed Index API: {data}")

            fear_greed_data = data["data"]

            if not fear_greed_data:
                raise ValueError("No data returned from Fear & Greed Index API")

            # Convert to DataFrame
            df_data = []
            for item in fear_greed_data:
                value = item.get("value")
                value_classification = item.get("value_classification")
                timestamp = item.get("timestamp")

                if value and timestamp:
                    # Convert timestamp to datetime (timestamp is in seconds) and
                    # normalize to date-only
                    dt = datetime.fromtimestamp(int(timestamp), tz=pytz.UTC)
                    # Normalize to midnight UTC to remove hours/minutes/seconds
                    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)

                    df_data.append(
                        {
                            "date": dt,
                            "fear_greed_value": int(value),
                            "value_classification": value_classification,
                        }
                    )

            if not df_data:
                raise ValueError("No valid data points found in Fear & Greed Index response")

            fear_greed_df = pd.DataFrame(df_data)
            fear_greed_df.set_index("date", inplace=True)
            fear_greed_df.index.name = "time"

            # Remove duplicates and sort
            fear_greed_df = fear_greed_df.loc[
                ~fear_greed_df.index.duplicated(keep="last")
            ].sort_index()

            logging.info(
                "Loaded Fear & Greed Index data into memory (%d rows)",
                len(fear_greed_df),
            )
            self._validate_data(fear_greed_df)

            return fear_greed_df

        except requests.exceptions.RequestException as e:
            logging.error("Failed to download Fear & Greed Index data: %s", e)
            raise
        except Exception as e:
            logging.error("Failed to process Fear & Greed Index data: %s", e)
            raise

    def extract_to_csv(self, local_path: str | Path | None = None) -> Path:
        """
        Download Fear & Greed Index daily time‑series and store them locally as CSV.

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
        fear_greed_df = self.load_from_web()
        fear_greed_df.to_csv(local_path)
        logging.info("Saved Fear & Greed Index data ➜ %s", local_path)

        return local_path

    def extract_to_parquet(self, local_path: str | Path | None = None) -> Path:
        """
        Download Fear & Greed Index daily time‑series and store them locally as Parquet.

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
        fear_greed_df = self.load_from_web()
        fear_greed_df.to_parquet(local_path)
        logging.info("Saved Fear & Greed Index data ➜ %s", local_path)

        return local_path

    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load Fear & Greed Index data from a local CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the CSV file. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with Fear & Greed Index data, indexed by datetime.
        """
        if path is None:
            path = self.data_dir / self.DEFAULT_FILENAME
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "Fear & Greed Index data file not found at %s. Downloading automatically...",
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
        Load Fear & Greed Index data from a local Parquet file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the Parquet file. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with Fear & Greed Index data, indexed by datetime.
        """
        if path is None:
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            path = self.data_dir / parquet_filename
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "Fear & Greed Index Parquet file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_parquet(path)

        df = pd.read_parquet(path)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        self._validate_data(df)
        return df

    def load(
        self,
        use_memory: bool = True,
        path: str | Path | None = None,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Load Fear & Greed Index data either from memory (web) or from a local file.

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
            DataFrame with Fear & Greed Index data, indexed by datetime.
        """
        if use_memory:
            logging.info("Loading Fear & Greed Index data directly from web...")
            return self.load_from_web()
        else:
            if file_format.lower() == "parquet":
                return self.load_from_parquet(path)
            else:
                return self.load_from_file(path)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Basic sanity‑check on the Fear & Greed Index dataframe.
        """
        if df.empty or "fear_greed_value" not in df.columns:
            raise ValueError("Invalid Fear & Greed Index data – 'fear_greed_value' column missing.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex.")
        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware.")
        # Check if timezone is UTC (accept both pytz.UTC and pandas UTC)
        if str(df.index.tz) != "UTC":
            raise ValueError("DatetimeIndex must be in UTC timezone.")


# Convenience functions for backward compatibility
def load_fear_greed_data_from_web() -> pd.DataFrame:
    """Load Fear & Greed Index data from web (backward compatibility)."""
    loader = FearGreedLoader()
    return loader.load_from_web()


def extract_fear_greed_data_to_csv(local_path: str | Path | None = None) -> None:
    """Extract Fear & Greed Index data to CSV (backward compatibility)."""
    loader = FearGreedLoader()
    loader.extract_to_csv(local_path)


def extract_fear_greed_data_to_parquet(local_path: str | Path | None = None) -> None:
    """Extract Fear & Greed Index data to Parquet (backward compatibility)."""
    loader = FearGreedLoader()
    loader.extract_to_parquet(local_path)
