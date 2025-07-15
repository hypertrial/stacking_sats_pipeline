"""
FRED (Federal Reserve Economic Data) loader for multiple economic time series.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

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
    """Loader for FRED economic data supporting multiple series."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    DEFAULT_FILENAME = "fred_data.csv"

    # Series ID constants organized by category
    # Rates / Curve
    RATES_CURVE = {
        "DGS1": "1-Year Treasury Constant Maturity",
        "DGS2": "2-Year Treasury Constant Maturity",
        "DGS5": "5-Year Treasury Constant Maturity",
        "DGS10": "10-Year Treasury Constant Maturity",
        "DGS30": "30-Year Treasury Constant Maturity",
        "DFF": "Federal Funds Effective Rate",
        "SOFR": "Secured Overnight Financing Rate",
        "SOFR30DAYAVG": "30-Day Average SOFR",
        "OBFR": "Overnight Bank Funding Rate",
    }

    # Inflation & real rates
    INFLATION_REAL = {
        "DFII5": "5-Year Treasury Real Yield",
        "DFII10": "10-Year Treasury Real Yield",
        "DFII30": "30-Year Treasury Real Yield",
        "T5YIE": "5-Year Breakeven Inflation Rate",
        "T10YIE": "10-Year Breakeven Inflation Rate",
        "T5YIFR": "5-Year Forward Inflation Expectation Rate",
    }

    # Credit & spreads
    CREDIT_SPREADS = {
        "BAMLC0A0CM": "ICE BofA US Corporate Index Option-Adjusted Spread",
        "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
        "BAMLC0A4CBBB": "ICE BofA US Corporate BBB Index Option-Adjusted Spread",
    }

    # Equities & vol
    EQUITIES_VOL = {
        "SP500": "S&P 500 Index",
        "DJIA": "Dow Jones Industrial Average",
        "NASDAQCOM": "NASDAQ Composite Index",
        "VIXCLS": "CBOE Volatility Index: VIX",
    }

    # FX
    FX_RATES = {
        "DTWEXBGS": "Nominal Broad U.S. Dollar Index",  # Original series
        "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
        "DEXJPUS": "Japan / U.S. Foreign Exchange Rate",
        "DEXUSUK": "U.S. / U.K. Foreign Exchange Rate",
        "DEXCAUS": "Canada / U.S. Foreign Exchange Rate",
        "DTWEXAFEGS": "Nominal Advanced Foreign Economies U.S. Dollar Index",
        "DTWEXEMEGS": "Nominal Emerging Markets U.S. Dollar Index",
    }

    # Commodities
    COMMODITIES = {
        "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate",
        "DCOILBRENTEU": "Crude Oil Prices: Brent - Europe",
        "GOLDAMGBD228NLBM": "Gold Fixing Price AM London",
        "DHHNGSP": "Henry Hub Natural Gas Spot Price",
        "PCOPPUSDM": "Global Price of Copper",
    }

    # Crypto
    CRYPTO = {
        "CBBTCUSD": "Coinbase Bitcoin",
        "CBETHUSD": "Coinbase Ethereum",
    }

    # Alt-high-freq
    ALT_HIGH_FREQ = {
        "JOBPOSTNUS": "Indeed Job Postings Index: National",
    }

    # All series combined
    ALL_SERIES = {
        **RATES_CURVE,
        **INFLATION_REAL,
        **CREDIT_SPREADS,
        **EQUITIES_VOL,
        **FX_RATES,
        **COMMODITIES,
        **CRYPTO,
        **ALT_HIGH_FREQ,
    }

    # Legacy series ID for backward compatibility
    SERIES_ID = "DTWEXBGS"  # Nominal Broad U.S. Dollar Index

    def __init__(
        self,
        data_dir: str | Path | None = None,
        api_key: str | None = None,
        series_ids: list[str] | None = None,
    ):
        """
        Initialize FRED loader.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory to store/load CSV files. If None, uses current file's parent directory.
        api_key : str, optional
            FRED API key. If None, will try to get from environment variable FRED_API_KEY.
        series_ids : List[str], optional
            List of FRED series IDs to load. If None, defaults to original DXY series.
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

        # Set series IDs - default to original DXY for backward compatibility
        self.series_ids = series_ids or [self.SERIES_ID]

        # Validate series IDs
        invalid_series = [sid for sid in self.series_ids if sid not in self.ALL_SERIES]
        if invalid_series:
            logging.warning(
                "Unknown series IDs: %s. Known series: %s",
                invalid_series,
                list(self.ALL_SERIES.keys()),
            )

    def load_from_web(self) -> pd.DataFrame:
        """
        Download FRED economic time-series directly into memory.

        Returns
        -------
        pd.DataFrame
            DataFrame with economic data, indexed by datetime, with columns named like 'DGS10_Value'.
        """
        logging.info(
            "Downloading %d series from FRED API: %s", len(self.series_ids), self.series_ids
        )

        frames = []

        for series_id in self.series_ids:
            logging.info("Fetching series: %s", series_id)

            params = {
                "series_id": series_id,
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
                    logging.warning("Invalid response for series %s: %s", series_id, data)
                    continue

                observations = data["observations"]

                if not observations:
                    logging.warning("No data returned for series %s", series_id)
                    continue

                # Convert to DataFrame
                df_data = []
                for obs in observations:
                    date = obs["date"]
                    value = obs["value"]

                    # Skip missing values (marked as '.' in FRED)
                    if value != ".":
                        try:
                            # Parse date and normalize to midnight UTC
                            naive_dt = pd.to_datetime(date).normalize()
                            utc_dt = naive_dt.tz_localize("UTC")

                            df_data.append({"date": utc_dt, f"{series_id}_Value": float(value)})
                        except (ValueError, TypeError) as e:
                            logging.warning("Skipping invalid data point for %s: %s", series_id, e)
                            continue

                if df_data:
                    series_df = pd.DataFrame(df_data)
                    series_df.set_index("date", inplace=True)
                    series_df.index.name = "time"

                    # Remove duplicates and sort
                    series_df = series_df.loc[~series_df.index.duplicated(keep="last")].sort_index()
                    frames.append(series_df)

                    logging.info("Loaded series %s (%d rows)", series_id, len(series_df))
                else:
                    logging.warning("No valid data points for series %s", series_id)

            except requests.exceptions.RequestException as e:
                logging.error("Failed to download series %s: %s", series_id, e)
                continue
            except Exception as e:
                logging.error("Failed to process series %s: %s", series_id, e)
                continue

        if not frames:
            raise ValueError("No valid data retrieved for any series")

        # Combine all series into a single DataFrame
        combined_df = pd.concat(frames, axis=1, sort=True)

        # Forward fill missing values to handle holidays/weekends
        combined_df = combined_df.ffill()

        logging.info(
            "Combined FRED data loaded (%d rows, %d series)", len(combined_df), len(frames)
        )
        self._validate_data(combined_df)

        return combined_df

    def load_series(self, series_ids: list[str]) -> pd.DataFrame:
        """
        Load specific series IDs (convenience method).

        Parameters
        ----------
        series_ids : List[str]
            List of FRED series IDs to load.

        Returns
        -------
        pd.DataFrame
            DataFrame with requested series data.
        """
        original_series = self.series_ids
        self.series_ids = series_ids
        try:
            return self.load_from_web()
        finally:
            self.series_ids = original_series

    def load_category(self, category: str) -> pd.DataFrame:
        """
        Load all series from a specific category.

        Parameters
        ----------
        category : str
            Category name: 'rates', 'inflation', 'credit', 'equities', 'fx', 'commodities', 'crypto', 'alt'

        Returns
        -------
        pd.DataFrame
            DataFrame with category series data.
        """
        category_map = {
            "rates": self.RATES_CURVE,
            "inflation": self.INFLATION_REAL,
            "credit": self.CREDIT_SPREADS,
            "equities": self.EQUITIES_VOL,
            "fx": self.FX_RATES,
            "commodities": self.COMMODITIES,
            "crypto": self.CRYPTO,
            "alt": self.ALT_HIGH_FREQ,
        }

        if category not in category_map:
            raise ValueError(
                f"Unknown category: {category}. Available: {list(category_map.keys())}"
            )

        return self.load_series(list(category_map[category].keys()))

    def extract_to_csv(self, local_path: str | Path | None = None) -> Path:
        """
        Download FRED economic time‑series and store them locally as CSV.

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
        df = self.load_from_web()
        df.to_csv(local_path)
        logging.info("Saved FRED data ➜ %s", local_path)

        return local_path

    def extract_to_parquet(self, local_path: str | Path | None = None) -> Path:
        """
        Download FRED economic time‑series and store them locally as Parquet.

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
        df = self.load_from_web()
        df.to_parquet(local_path)
        logging.info("Saved FRED data ➜ %s", local_path)

        return local_path

    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load FRED economic data from a local CSV file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the CSV file. If None, defaults to DEFAULT_FILENAME in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with economic data, indexed by datetime.
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

        # Convert naive datetime index to UTC timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        self._validate_data(df)
        return df

    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load FRED economic data from a local Parquet file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the Parquet file. If None, defaults to DEFAULT_FILENAME with
            .parquet extension in data_dir.

        Returns
        -------
        pd.DataFrame
            DataFrame with economic data, indexed by datetime.
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
        Load FRED economic data either from memory (web) or from a local file.

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
            DataFrame with economic data, indexed by datetime.
        """
        if use_memory:
            logging.info("Loading FRED data directly from web...")
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
        if df.empty:
            raise ValueError("FRED dataframe is empty")

        # Check for at least one value column
        value_cols = [col for col in df.columns if col.endswith("_Value")]
        if not value_cols:
            raise ValueError("No value columns found in FRED data")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware")

        # Check if timezone is UTC (accept both pytz.UTC and pandas UTC)
        if str(df.index.tz) != "UTC":
            raise ValueError("DatetimeIndex must be in UTC timezone")

    @classmethod
    def get_available_series(cls) -> dict[str, str]:
        """Get all available series IDs with descriptions."""
        return cls.ALL_SERIES.copy()

    @classmethod
    def get_series_by_category(cls, category: str) -> dict[str, str]:
        """Get series IDs for a specific category."""
        category_map = {
            "rates": cls.RATES_CURVE,
            "inflation": cls.INFLATION_REAL,
            "credit": cls.CREDIT_SPREADS,
            "equities": cls.EQUITIES_VOL,
            "fx": cls.FX_RATES,
            "commodities": cls.COMMODITIES,
            "crypto": cls.CRYPTO,
            "alt": cls.ALT_HIGH_FREQ,
        }

        if category not in category_map:
            raise ValueError(
                f"Unknown category: {category}. Available: {list(category_map.keys())}"
            )

        return category_map[category].copy()


# Convenience functions for backward compatibility
def load_dxy_data_from_web(api_key: str | None = None) -> pd.DataFrame:
    """Load FRED U.S. Dollar Index data from web (backward compatibility)."""
    loader = FREDLoader(api_key=api_key, series_ids=["DTWEXBGS"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "DTWEXBGS_Value" in df.columns:
        df = df.rename(columns={"DTWEXBGS_Value": "DXY_Value"})
    return df


def extract_dxy_data_to_csv(
    local_path: str | Path | None = None, api_key: str | None = None
) -> None:
    """Extract FRED U.S. Dollar Index data to CSV (backward compatibility)."""
    loader = FREDLoader(api_key=api_key, series_ids=["DTWEXBGS"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "DTWEXBGS_Value" in df.columns:
        df = df.rename(columns={"DTWEXBGS_Value": "DXY_Value"})

    if local_path is None:
        local_path = loader.data_dir / "dxy_fred.csv"
    else:
        local_path = Path(local_path)

    df.to_csv(local_path)
    logging.info("Saved FRED U.S. Dollar Index data ➜ %s", local_path)


def extract_dxy_data_to_parquet(
    local_path: str | Path | None = None, api_key: str | None = None
) -> None:
    """Extract FRED U.S. Dollar Index data to Parquet (backward compatibility)."""
    loader = FREDLoader(api_key=api_key, series_ids=["DTWEXBGS"])
    df = loader.load_from_web()
    # Rename column to match legacy format
    if "DTWEXBGS_Value" in df.columns:
        df = df.rename(columns={"DTWEXBGS_Value": "DXY_Value"})

    if local_path is None:
        local_path = loader.data_dir / "dxy_fred.parquet"
    else:
        local_path = Path(local_path)

    df.to_parquet(local_path)
    logging.info("Saved FRED U.S. Dollar Index data ➜ %s", local_path)
