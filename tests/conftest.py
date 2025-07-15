"""
Pytest configuration and fixtures for stacking_sats_pipeline tests.
"""

import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Test data cache configuration
CACHE_DIR = Path(__file__).parent / ".test_cache"
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours


@pytest.fixture
def test_cache_dir():
    """Create and return the test cache directory."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR


@pytest.fixture
def cached_api_data(test_cache_dir):
    """Cache API responses to disk for faster test runs."""

    def _get_cached_data(source_name, api_function, cache_key=None):
        """Get cached data or fetch and cache new data."""
        cache_key = cache_key or source_name
        cache_file = test_cache_dir / f"{cache_key}.pkl"

        # Check if cache exists and is fresh
        if cache_file.exists():
            cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    # If cache is corrupted, continue to fetch fresh data
                    pass

        # Fetch fresh data
        try:
            data = api_function()
            # Cache the data
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            return data
        except Exception as e:
            # If API call fails, try to use stale cache
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
            raise e

    return _get_cached_data


@pytest.fixture
def cached_coinmetrics_data(cached_api_data):
    """Get cached CoinMetrics data."""

    def _fetch_coinmetrics():
        from stacking_sats_pipeline.data import load_data

        return load_data("coinmetrics", use_memory=True)

    return cached_api_data("coinmetrics", _fetch_coinmetrics)


@pytest.fixture
def cached_fred_data(cached_api_data):
    """Get cached FRED data."""

    def _fetch_fred():
        from stacking_sats_pipeline.data import load_data

        return load_data("fred", use_memory=True)

    return cached_api_data("fred", _fetch_fred)


@pytest.fixture
def performance_comparison_data(large_mock_data):
    """Generate performance comparison data with different sizes."""
    sizes = [100, 500, 1000, 5000]
    datasets = {}

    for size in sizes:
        base_dates = pd.date_range("2020-01-01", periods=size, name="time", tz="UTC")
        datasets[size] = pd.DataFrame(
            {
                "PriceUSD_coinmetrics": range(30000, 30000 + size),
                "VolumeUSD_coinmetrics": range(1000000, 1000000 + size),
                "DTWEXBGS_Value_fred": [100 + i * 0.01 for i in range(size)],
                "DGS10_Value_fred": [1.5 + i * 0.001 for i in range(size)],
                "fear_greed_value_feargreed": [50 + (i % 100) - 50 for i in range(size)],
            },
            index=base_dates,
        )

    return datasets


@pytest.fixture
def mock_coinmetrics_data():
    """Mock CoinMetrics data for testing."""
    return pd.DataFrame(
        {
            "PriceUSD": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0],
            "VolumeUSD": [1000000.0, 1100000.0, 1200000.0, 1300000.0, 1400000.0],
        },
        index=pd.date_range("2020-01-01", periods=5, name="time", tz="UTC"),
    )


@pytest.fixture
def mock_fred_data():
    """Mock FRED data for testing."""
    return pd.DataFrame(
        {
            "DTWEXBGS_Value": [100.0, 100.5, 101.0, 101.5, 102.0],
            "DGS10_Value": [1.5, 1.6, 1.7, 1.8, 1.9],
            "DFF_Value": [0.25, 0.25, 0.5, 0.5, 0.75],
        },
        index=pd.date_range("2020-01-01", periods=5, name="time", tz="UTC"),
    )


@pytest.fixture
def mock_feargreed_data():
    """Mock Fear & Greed data for testing."""
    return pd.DataFrame(
        {
            "fear_greed_value": [50, 45, 40, 35, 30],
            "fear_greed_classification": ["Neutral", "Fear", "Fear", "Fear", "Extreme Fear"],
        },
        index=pd.date_range("2020-01-01", periods=5, name="time", tz="UTC"),
    )


@pytest.fixture
def mock_merged_data(mock_coinmetrics_data, mock_fred_data, mock_feargreed_data):
    """Mock merged data from all sources."""
    # Merge all data sources with proper column naming
    merged = pd.DataFrame()
    merged["PriceUSD_coinmetrics"] = mock_coinmetrics_data["PriceUSD"]
    merged["VolumeUSD_coinmetrics"] = mock_coinmetrics_data["VolumeUSD"]
    merged["DTWEXBGS_Value_fred"] = mock_fred_data["DTWEXBGS_Value"]
    merged["DGS10_Value_fred"] = mock_fred_data["DGS10_Value"]
    merged["DFF_Value_fred"] = mock_fred_data["DFF_Value"]
    merged["fear_greed_value_feargreed"] = mock_feargreed_data["fear_greed_value"]
    merged["fear_greed_classification_feargreed"] = mock_feargreed_data["fear_greed_classification"]
    merged.index = mock_coinmetrics_data.index
    return merged


@pytest.fixture
def mock_extract_all_data(mock_merged_data):
    """Mock extract_all_data function that returns realistic data without API calls."""

    def _mock_extract_all_data(file_format=None, output_dir=None, format_type=None):
        """Mock implementation that creates realistic test files."""
        # Support both old and new function signatures
        if file_format is not None:
            format_type = file_format
        if format_type is None:
            format_type = "csv"
        if output_dir is None:
            output_dir = "."

        output_path = Path(output_dir)

        if format_type.lower() == "csv":
            file_path = output_path / "merged_crypto_data.csv"
            mock_merged_data.to_csv(file_path)
        elif format_type.lower() == "parquet":
            file_path = output_path / "merged_crypto_data.parquet"
            mock_merged_data.to_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return file_path

    return _mock_extract_all_data


@pytest.fixture
def large_mock_data():
    """Create larger mock dataset for performance testing."""
    dates = pd.date_range("2020-01-01", periods=1000, name="time", tz="UTC")
    return pd.DataFrame(
        {
            "PriceUSD_coinmetrics": range(30000, 31000),
            "VolumeUSD_coinmetrics": range(1000000, 1001000),
            "DTWEXBGS_Value_fred": [100 + i * 0.01 for i in range(1000)],
            "DGS10_Value_fred": [1.5 + i * 0.001 for i in range(1000)],
            "fear_greed_value_feargreed": [50 + (i % 100) - 50 for i in range(1000)],
        },
        index=dates,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_api_responses():
    """Mock API responses for FRED and CoinMetrics."""
    return {
        "fred": {
            "observations": [
                {"date": "2020-01-01", "value": "100.0"},
                {"date": "2020-01-02", "value": "100.5"},
                {"date": "2020-01-03", "value": "101.0"},
            ]
        },
        "coinmetrics": """time,PriceUSD
2020-01-01T00:00:00.000000000Z,30000.0
2020-01-02T00:00:00.000000000Z,31000.0
2020-01-03T00:00:00.000000000Z,32000.0""",
        "feargreed": {
            "data": [
                {"value": "50", "value_classification": "Neutral", "timestamp": "1577836800"},
                {"value": "45", "value_classification": "Fear", "timestamp": "1577923200"},
                {"value": "40", "value_classification": "Fear", "timestamp": "1578009600"},
            ]
        },
    }


@pytest.fixture
def mock_requests_get(mock_api_responses):
    """Mock requests.get for API calls."""

    def _mock_get(url, *args, **kwargs):
        response = MagicMock()
        response.raise_for_status.return_value = None

        if "stlouisfed.org" in url or "fred" in url.lower():
            response.json.return_value = mock_api_responses["fred"]
        elif "coinmetrics" in url or "community-api" in url:
            response.text = mock_api_responses["coinmetrics"]
        elif "feargreed" in url or "alternative.me" in url:
            response.json.return_value = mock_api_responses["feargreed"]
        else:
            response.text = ""

        return response

    return _mock_get


@pytest.fixture
def performance_timer():
    """Utility to measure test performance."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None

        def start(self):
            self.start_time = time.time()

        def elapsed(self):
            if self.start_time is None:
                return 0
            return time.time() - self.start_time

    return Timer()


# Cache cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_cache():
    """Clean up old test cache files on session start."""
    if CACHE_DIR.exists():
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep cache for 7 days max
        for cache_file in CACHE_DIR.glob("*.pkl"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                try:
                    cache_file.unlink()
                except OSError:
                    pass
    yield
    # Session cleanup could go here if needed
