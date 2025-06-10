"""
Generic data extraction and loading utilities for multiple BTC price data sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import pandas as pd

from .coinmetrics_loader import CoinMetricsLoader

# Logging configuration
# ---------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def load(
        self, use_memory: bool = True, path: str | Path | None = None
    ) -> pd.DataFrame:
        """Load data from source."""
        ...


class MultiSourceDataLoader:
    """
    Generic data loader that can ingest and merge data from multiple sources.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize multi-source data loader.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory to store/load CSV files. If None, uses current file's parent directory.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)

        # Initialize available loaders
        self.loaders: Dict[str, DataLoader] = {
            "coinmetrics": CoinMetricsLoader(self.data_dir)
        }

    def add_loader(self, name: str, loader: DataLoader) -> None:
        """
        Add a new data loader.

        Parameters
        ----------
        name : str
            Name to identify the loader.
        loader : DataLoader
            The loader instance implementing the DataLoader protocol.
        """
        self.loaders[name] = loader
        logging.info("Added data loader: %s", name)

    def load_from_source(
        self, source: str, use_memory: bool = True, path: str | Path | None = None
    ) -> pd.DataFrame:
        """
        Load data from a specific source.

        Parameters
        ----------
        source : str
            Name of the data source loader to use.
        use_memory : bool, default True
            If True, loads data directly from web into memory.
            If False, loads from local CSV file.
        path : str or Path, optional
            Path to the CSV file. Only used if use_memory=False.

        Returns
        -------
        pd.DataFrame
            DataFrame with data from the specified source.
        """
        if source not in self.loaders:
            raise ValueError(
                f"Unknown data source: {source}. Available sources: {list(self.loaders.keys())}"
            )

        loader = self.loaders[source]
        return loader.load(use_memory=use_memory, path=path)

    def load_and_merge(
        self,
        sources: List[str],
        use_memory: bool = True,
        merge_on: str = "time",
        how: str = "outer",
    ) -> pd.DataFrame:
        """
        Load data from multiple sources and merge them.

        Parameters
        ----------
        sources : List[str]
            List of data source names to load and merge.
        use_memory : bool, default True
            If True, loads data directly from web into memory.
            If False, loads from local CSV files.
        merge_on : str, default 'time'
            Column or index to merge on.
        how : str, default 'outer'
            Type of merge to perform ('inner', 'outer', 'left', 'right').

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with data from all specified sources.
        """
        if not sources:
            raise ValueError("At least one source must be specified")

        # Load first source
        merged_df = self.load_from_source(sources[0], use_memory=use_memory)

        # Add source suffix to columns (except index)
        if len(sources) > 1:
            merged_df = merged_df.add_suffix(f"_{sources[0]}")

        # Merge additional sources
        for source in sources[1:]:
            source_df = self.load_from_source(source, use_memory=use_memory)
            source_df = source_df.add_suffix(f"_{source}")

            # Merge on index if merge_on is 'time' or index name
            if merge_on == "time" or merge_on == merged_df.index.name:
                merged_df = merged_df.merge(
                    source_df, left_index=True, right_index=True, how=how
                )
            else:
                merged_df = merged_df.merge(source_df, on=merge_on, how=how)

        logging.info(
            "Merged data from sources: %s (shape: %s)", sources, merged_df.shape
        )
        return merged_df

    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.

        Returns
        -------
        List[str]
            List of available data source names.
        """
        return list(self.loaders.keys())


# Main convenience functions
def load_data(
    source: str = "coinmetrics", use_memory: bool = True, path: str | Path | None = None
) -> pd.DataFrame:
    """
    Load BTC price data from a specific source.

    Parameters
    ----------
    source : str, default 'coinmetrics'
        Name of the data source to use.
    use_memory : bool, default True
        If True, loads data directly from web into memory.
        If False, loads from local CSV file.
    path : str or Path, optional
        Path to the CSV file. Only used if use_memory=False.

    Returns
    -------
    pd.DataFrame
        DataFrame with BTC data, indexed by datetime.
    """
    loader = MultiSourceDataLoader()
    return loader.load_from_source(source, use_memory=use_memory, path=path)


def load_and_merge_data(
    sources: List[str],
    use_memory: bool = True,
    merge_on: str = "time",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Load and merge BTC price data from multiple sources.

    Parameters
    ----------
    sources : List[str]
        List of data source names to load and merge.
    use_memory : bool, default True
        If True, loads data directly from web into memory.
        If False, loads from local CSV files.
    merge_on : str, default 'time'
        Column or index to merge on.
    how : str, default 'outer'
        Type of merge to perform ('inner', 'outer', 'left', 'right').

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with data from all specified sources.
    """
    loader = MultiSourceDataLoader()
    return loader.load_and_merge(
        sources, use_memory=use_memory, merge_on=merge_on, how=how
    )


def validate_price_data(
    df: pd.DataFrame, price_columns: Optional[List[str]] = None
) -> None:
    """
    Basic sanity‑check on the input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    price_columns : List[str], optional
        List of expected price columns. If None, looks for any column containing 'Price'.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")

    if price_columns is None:
        # Look for any column containing 'Price'
        price_columns = [col for col in df.columns if "Price" in col]

    if not price_columns:
        raise ValueError("No price columns found in the data.")

    # Check if at least one price column exists
    missing_cols = [col for col in price_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing price columns: {missing_cols}")


# Backward compatibility functions
def load_btc_data_from_web() -> pd.DataFrame:
    """Load CoinMetrics BTC data from web (backward compatibility)."""
    return load_data("coinmetrics", use_memory=True)


def extract_btc_data_to_csv(local_path: str | Path | None = None) -> None:
    """Extract CoinMetrics BTC data to CSV (backward compatibility)."""
    loader = CoinMetricsLoader()
    loader.extract_to_csv(local_path)
