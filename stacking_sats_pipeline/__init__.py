"""
Stacking Sats Pipeline - Data Engineering for Cryptocurrency and Financial Data
============================================================================

This package provides tools for extracting, loading, and merging cryptocurrency
and financial data from multiple sources.

Quick Start:
    >>> from stacking_sats_pipeline import extract_all_data
    >>> extract_all_data("csv")  # Extract all data to CSV
    >>> extract_all_data("parquet", "data/")  # Extract to Parquet in data/ folder

Data Loading:
    >>> from stacking_sats_pipeline import load_data
    >>> df = load_data()  # Load Bitcoin price data
"""

# Load environment variables from .env file if available
try:
    from pathlib import Path

    from dotenv import load_dotenv

    # Look for .env file in current working directory first, then package directory
    env_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent / ".env",  # Package directory
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    pass

# Configuration constants
from .config import (
    BACKTEST_END,
    BACKTEST_START,
)

# Data engineering imports
from .data import (
    extract_btc_data_to_csv,
    extract_btc_data_to_parquet,
    load_btc_data_from_web,
    load_data,
    validate_price_data,
)
from .main import extract_all_data

__version__ = "0.3.0"

__all__ = [
    # Configuration constants
    "BACKTEST_END",
    "BACKTEST_START",
    # Data loading and extraction
    "extract_btc_data_to_csv",
    "extract_btc_data_to_parquet",
    "extract_all_data",
    "load_data",
    "load_btc_data_from_web",
    "validate_price_data",
]
