"""
FRED (Federal Reserve Economic Data) loader for U.S. Dollar Index data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
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


class FREDLoader:
    """Loader for FRED economic data."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    DEFAULT_FILENAME = "dxy_fred.csv"
    SERIES_ID = "DTWEXBGS"  # Nominal Broad U.S. Dollar Index

    def __init__(
        self, data_dir: str | Path | None = None, api_key: Optional[str] = None
    ):
        """
        Initialize FRED loader.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory to store/load CSV files. If None, uses current file's parent directory.
        api_key : str, optional
            FRED API key. If None, will try to get from environment variable FRED_API_KEY.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("FRED_API_KEY")

        if not self.api_key:
            raise ValueError(
                "FRED API key is required. Please set FRED_API_KEY in your .env file or "
                "pass api_key parameter. Get a free API key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def load_from_web(self) -> pd.DataFrame:
        """
        Download FRED U.S. Dollar Index daily time-series directly into memory.

        Returns
        -------
        pd.DataFrame
            DataFrame with U.S. Dollar Index data, indexed by datetime.
        """
        logging.info("Downloading U.S. Dollar Index data from FRED API...")

        params = {
            "series_id": self.SERIES_ID,
            "api_key": self.api_key,
            "file_type": "json",
            "frequency": "d",  # daily frequency
            "output_type": 1,  # observations only
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()

            if "observations" not in data:
                raise ValueError(f"Invalid response from FRED API: {data}")

            observations = data["observations"]

            if not observations:
                raise ValueError("No data returned from FRED API")

            # Convert to DataFrame
            df_data = []
            for obs in observations:
                date = obs["date"]
                value = obs["value"]

                # Skip missing values (marked as '.' in FRED)
                if value != ".":
                    df_data.append(
                        {"date": pd.to_datetime(date), "DXY_Value": float(value)}
                    )

            if not df_data:
                raise ValueError("No valid data points found in FRED response")

            dxy_df = pd.DataFrame(df_data)
            dxy_df.set_index("date", inplace=True)
            dxy_df.index.name = "time"

            # Remove duplicates and sort
            dxy_df = dxy_df.loc[~dxy_df.index.duplicated(keep="last")].sort_index()

            logging.info(
                "Loaded FRED U.S. Dollar Index data into memory (%d rows)", len(dxy_df)
            )
            self._validate_data(dxy_df)

            return dxy_df

        except requests.exceptions.RequestException as e:
            logging.error("Failed to download FRED data: %s", e)
            raise
        except Exception as e:
            logging.error("Failed to process FRED data: %s", e)
            raise

    def extract_to_csv(self, local_path: str | Path | None = None) -> Path:
        """
        Download FRED U.S. Dollar Index daily time‑series and store them locally as CSV.

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
        dxy_df = self.load_from_web()
        dxy_df.to_csv(local_path)
        logging.info("Saved FRED U.S. Dollar Index data ➜ %s", local_path)

        return local_path

    def extract_to_parquet(self, local_path: str | Path | None = None) -> Path:
        """
        Download FRED U.S. Dollar Index daily time‑series and store them locally as Parquet.

        Parameters
        ----------
        local_path : str or Path, optional
            Destination Parquet path. If None, defaults to DEFAULT_FILENAME with .parquet extension in data_dir.

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
        dxy_df = self.load_from_web()
        dxy_df.to_parquet(local_path)
        logging.info("Saved FRED U.S. Dollar Index data ➜ %s", local_path)

        return local_path

    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load FRED U.S. Dollar Index data from a local CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the CSV file. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with U.S. Dollar Index data, indexed by datetime.
        """
        if path is None:
            path = self.data_dir / self.DEFAULT_FILENAME
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "FRED data file not found at %s. Downloading automatically...",
                path,
            )
            self.extract_to_csv(path)

        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        self._validate_data(df)
        return df

    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load FRED U.S. Dollar Index data from a local Parquet file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the Parquet file. If None, defaults to DEFAULT_FILENAME with .parquet extension in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with U.S. Dollar Index data, indexed by datetime.
        """
        if path is None:
            parquet_filename = self.DEFAULT_FILENAME.replace(".csv", ".parquet")
            path = self.data_dir / parquet_filename
        else:
            path = Path(path)

        if not path.exists():
            logging.info(
                "FRED Parquet file not found at %s. Downloading automatically...",
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
        Load FRED U.S. Dollar Index data either from memory (web) or from a local file.

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
            DataFrame with U.S. Dollar Index data, indexed by datetime.
        """
        if use_memory:
            logging.info("Loading FRED U.S. Dollar Index data directly from web...")
            return self.load_from_web()
        else:
            if file_format.lower() == "parquet":
                return self.load_from_parquet(path)
            else:
                return self.load_from_file(path)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Basic sanity‑check on the FRED dataframe.
        """
        if df.empty or "DXY_Value" not in df.columns:
            raise ValueError(
                "Invalid FRED U.S. Dollar Index data – 'DXY_Value' column missing."
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex.")


# Convenience functions for backward compatibility
def load_dxy_data_from_web(api_key: Optional[str] = None) -> pd.DataFrame:
    """Load FRED U.S. Dollar Index data from web (backward compatibility)."""
    loader = FREDLoader(api_key=api_key)
    return loader.load_from_web()


def extract_dxy_data_to_csv(
    local_path: str | Path | None = None, api_key: Optional[str] = None
) -> None:
    """Extract FRED U.S. Dollar Index data to CSV (backward compatibility)."""
    loader = FREDLoader(api_key=api_key)
    loader.extract_to_csv(local_path)


def extract_dxy_data_to_parquet(
    local_path: str | Path | None = None, api_key: Optional[str] = None
) -> None:
    """Extract FRED U.S. Dollar Index data to Parquet (backward compatibility)."""
    loader = FREDLoader(api_key=api_key)
    loader.extract_to_parquet(local_path)
