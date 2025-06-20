"""
Data loading and extraction utilities.
"""

from .coinmetrics_loader import CoinMetricsLoader
from .data_loader import (
    MultiSourceDataLoader,
    extract_btc_data_to_csv,
    extract_btc_data_to_parquet,
    load_and_merge_data,
    load_btc_data_from_web,
    load_data,
    validate_price_data,
)
from .fred_loader import FREDLoader

__all__ = [
    # Main classes
    "MultiSourceDataLoader",
    "CoinMetricsLoader",
    "FREDLoader",
    # Main functions
    "load_data",
    "load_and_merge_data",
    "validate_price_data",
    # File export functions
    "extract_btc_data_to_csv",
    "extract_btc_data_to_parquet",
    # Backward compatibility
    "load_btc_data_from_web",
]
