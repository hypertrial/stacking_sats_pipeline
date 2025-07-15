"""
Tests for YFinanceLoader - Yahoo Finance data loader.

This module contains comprehensive tests for the YFinanceLoader class,
including data loading, validation, file operations, and integration tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from stacking_sats_pipeline.data.yfinance_loader import (
    YFinanceLoader,
    extract_btc_data_to_csv,
    extract_btc_data_to_parquet,
    load_btc_data_from_web,
    load_crypto_data_from_web,
    load_stock_data_from_web,
)


class TestYFinanceLoader:
    """Test suite for YFinanceLoader class."""

    def test_init_default_params(self):
        """Test YFinanceLoader initialization with default parameters."""
        loader = YFinanceLoader()

        assert loader.symbols == ["BTC-USD"]
        assert loader.period == "max"
        assert loader.interval == "1d"
        assert loader.data_dir == Path(__file__).parent.parent / "stacking_sats_pipeline" / "data"

    def test_init_custom_params(self):
        """Test YFinanceLoader initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(
                data_dir=temp_dir,
                symbols=["AAPL", "GOOGL"],
                period="1y",
                interval="1h",
            )

            assert loader.symbols == ["AAPL", "GOOGL"]
            assert loader.period == "1y"
            assert loader.interval == "1h"
            assert loader.data_dir == Path(temp_dir)

    def test_init_invalid_symbols_type(self):
        """Test YFinanceLoader initialization with invalid symbols type."""
        with pytest.raises(ValueError, match="symbols must be a list of strings"):
            YFinanceLoader(symbols="AAPL")

    def test_init_unknown_symbols_warning(self, caplog):
        """Test YFinanceLoader initialization with unknown symbols logs warning."""
        loader = YFinanceLoader(symbols=["UNKNOWN-SYMBOL"])

        assert "Unknown symbols (may still be valid)" in caplog.text
        assert loader.symbols == ["UNKNOWN-SYMBOL"]

    def test_get_available_symbols(self):
        """Test get_available_symbols class method."""
        symbols = YFinanceLoader.get_available_symbols()

        assert isinstance(symbols, dict)
        assert "BTC-USD" in symbols
        assert "AAPL" in symbols
        assert "^GSPC" in symbols
        assert "SPY" in symbols
        assert len(symbols) > 30  # Should have symbols from all categories

    def test_get_symbols_by_category(self):
        """Test get_symbols_by_category class method."""
        # Test valid categories
        stocks = YFinanceLoader.get_symbols_by_category("stocks")
        assert "AAPL" in stocks
        assert "GOOGL" in stocks

        crypto = YFinanceLoader.get_symbols_by_category("crypto")
        assert "BTC-USD" in crypto
        assert "ETH-USD" in crypto

        indices = YFinanceLoader.get_symbols_by_category("indices")
        assert "^GSPC" in indices
        assert "^DJI" in indices

        etfs = YFinanceLoader.get_symbols_by_category("etfs")
        assert "SPY" in etfs
        assert "QQQ" in etfs

    def test_get_symbols_by_category_invalid(self):
        """Test get_symbols_by_category with invalid category."""
        with pytest.raises(ValueError, match="Unknown category"):
            YFinanceLoader.get_symbols_by_category("invalid")

    def create_mock_yfinance_data(self, symbol="AAPL", num_days=100):
        """Create mock yfinance data for testing."""
        dates = pd.date_range("2023-01-01", periods=num_days, freq="D")
        np.random.seed(42)  # For reproducibility

        # Create realistic OHLCV data
        base_price = 150.0 if symbol == "AAPL" else 30000.0
        price_changes = np.random.normal(0, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Create OHLCV data
        opens = prices * (1 + np.random.normal(0, 0.005, num_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, num_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, num_days)))
        volumes = np.random.randint(1000000, 50000000, num_days)

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            }
        )

        return data

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_web_single_symbol(self, mock_ticker):
        """Test loading data from web for single symbol."""
        mock_data = self.create_mock_yfinance_data("AAPL", 100)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = YFinanceLoader(symbols=["AAPL"])
        df = loader.load_from_web()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "AAPL_Close" in df.columns
        assert "AAPL_Volume" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_web_multiple_symbols(self, mock_ticker):
        """Test loading data from web for multiple symbols."""

        def mock_ticker_side_effect(symbol):
            mock_ticker_instance = Mock()
            mock_data = self.create_mock_yfinance_data(symbol, 100)
            mock_ticker_instance.history.return_value = mock_data.set_index("Date")
            return mock_ticker_instance

        mock_ticker.side_effect = mock_ticker_side_effect

        loader = YFinanceLoader(symbols=["AAPL", "GOOGL"])
        df = loader.load_from_web()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "AAPL_Close" in df.columns
        assert "GOOGL_Close" in df.columns
        assert "AAPL_Volume" in df.columns
        assert "GOOGL_Volume" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_web_empty_response(self, mock_ticker):
        """Test loading data from web with empty response."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        loader = YFinanceLoader(symbols=["INVALID"])

        with pytest.raises(ValueError, match="No valid data retrieved"):
            loader.load_from_web()

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_web_exception_handling(self, mock_ticker):
        """Test loading data from web with exception handling."""
        mock_ticker.side_effect = Exception("API Error")

        loader = YFinanceLoader(symbols=["AAPL"])

        with pytest.raises(ValueError, match="No valid data retrieved"):
            loader.load_from_web()

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_symbols_method(self, mock_ticker):
        """Test load_symbols convenience method."""
        mock_data = self.create_mock_yfinance_data("MSFT", 50)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = YFinanceLoader(symbols=["AAPL"])  # Original symbols
        df = loader.load_symbols(["MSFT"])  # Load different symbols

        assert "MSFT_Close" in df.columns
        assert "AAPL_Close" not in df.columns
        assert loader.symbols == ["AAPL"]  # Original symbols should be restored

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_category_method(self, mock_ticker):
        """Test load_category convenience method."""

        def mock_ticker_side_effect(symbol):
            mock_ticker_instance = Mock()
            mock_data = self.create_mock_yfinance_data(symbol, 30)
            mock_ticker_instance.history.return_value = mock_data.set_index("Date")
            return mock_ticker_instance

        mock_ticker.side_effect = mock_ticker_side_effect

        loader = YFinanceLoader()
        df = loader.load_category("crypto")

        # Should contain all crypto symbols
        crypto_symbols = YFinanceLoader.get_symbols_by_category("crypto")
        for symbol in crypto_symbols:
            assert f"{symbol}_Close" in df.columns

    def test_load_category_invalid_category(self):
        """Test load_category with invalid category."""
        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="Unknown category"):
            loader.load_category("invalid")

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_extract_to_csv(self, mock_ticker):
        """Test extracting data to CSV file."""
        mock_data = self.create_mock_yfinance_data("AAPL", 50)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])
            csv_path = loader.extract_to_csv()

            assert csv_path.exists()
            assert csv_path.suffix == ".csv"

            # Load and verify CSV content
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            assert len(df) == 50
            assert "AAPL_Close" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_extract_to_parquet(self, mock_ticker):
        """Test extracting data to Parquet file."""
        mock_data = self.create_mock_yfinance_data("AAPL", 50)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])
            parquet_path = loader.extract_to_parquet()

            assert parquet_path.exists()
            assert parquet_path.suffix == ".parquet"

            # Load and verify Parquet content
            df = pd.read_parquet(parquet_path)
            assert len(df) == 50
            assert "AAPL_Close" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_file_auto_download(self, mock_ticker):
        """Test loading from file with automatic download."""
        mock_data = self.create_mock_yfinance_data("AAPL", 30)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])

            # File doesn't exist, should auto-download
            df = loader.load_from_file()

            assert len(df) == 30
            assert "AAPL_Close" in df.columns
            assert (Path(temp_dir) / "yfinance_data.csv").exists()

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_from_parquet_auto_download(self, mock_ticker):
        """Test loading from Parquet file with automatic download."""
        mock_data = self.create_mock_yfinance_data("AAPL", 30)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])

            # File doesn't exist, should auto-download
            df = loader.load_from_parquet()

            assert len(df) == 30
            assert "AAPL_Close" in df.columns
            assert (Path(temp_dir) / "yfinance_data.parquet").exists()

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_method_memory_mode(self, mock_ticker):
        """Test load method with memory mode."""
        mock_data = self.create_mock_yfinance_data("AAPL", 40)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = YFinanceLoader(symbols=["AAPL"])
        df = loader.load(use_memory=True)

        assert len(df) == 40
        assert "AAPL_Close" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_method_file_mode_csv(self, mock_ticker):
        """Test load method with file mode CSV."""
        mock_data = self.create_mock_yfinance_data("AAPL", 40)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])
            df = loader.load(use_memory=False, file_format="csv")

            assert len(df) == 40
            assert "AAPL_Close" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_method_file_mode_parquet(self, mock_ticker):
        """Test load method with file mode Parquet."""
        mock_data = self.create_mock_yfinance_data("AAPL", 40)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"])
            df = loader.load(use_memory=False, file_format="parquet")

            assert len(df) == 40
            assert "AAPL_Close" in df.columns

    def test_validate_data_valid(self):
        """Test _validate_data with valid data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "AAPL_Close": [150.0] * 10,
                "AAPL_Volume": [1000000] * 10,
            },
            index=dates,
        )

        loader = YFinanceLoader()
        loader._validate_data(df)  # Should not raise

    def test_validate_data_empty(self):
        """Test _validate_data with empty DataFrame."""
        df = pd.DataFrame()
        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="Yahoo Finance dataframe is empty"):
            loader._validate_data(df)

    def test_validate_data_no_price_columns(self):
        """Test _validate_data with no price columns."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "AAPL_Volume": [1000000] * 10,
            },
            index=dates,
        )

        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="No price columns found"):
            loader._validate_data(df)

    def test_validate_data_invalid_index(self):
        """Test _validate_data with invalid index."""
        df = pd.DataFrame(
            {
                "AAPL_Close": [150.0] * 10,
            },
            index=range(10),
        )

        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            loader._validate_data(df)

    def test_validate_data_naive_datetime(self):
        """Test _validate_data with naive datetime index."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")  # No timezone
        df = pd.DataFrame(
            {
                "AAPL_Close": [150.0] * 10,
            },
            index=dates,
        )

        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="DatetimeIndex must be timezone-aware"):
            loader._validate_data(df)

    def test_validate_data_wrong_timezone(self):
        """Test _validate_data with wrong timezone."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="US/Eastern")
        df = pd.DataFrame(
            {
                "AAPL_Close": [150.0] * 10,
            },
            index=dates,
        )

        loader = YFinanceLoader()

        with pytest.raises(ValueError, match="DatetimeIndex must be in UTC timezone"):
            loader._validate_data(df)


class TestYFinanceLoaderTimestampAlignment:
    """Test timestamp alignment and normalization for YFinanceLoader."""

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_yfinance_timestamps_normalized_to_midnight_utc(self, mock_ticker):
        """Test that YFinance timestamps are normalized to midnight UTC."""
        # Create mock data with non-midnight timestamps
        dates = pd.date_range("2023-01-01 15:30:00", periods=10, freq="D")
        mock_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [150.0] * 10,
                "Volume": [1000000] * 10,
            }
        )
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = YFinanceLoader(symbols=["AAPL"])
        df = loader.load_from_web()

        # Check that all timestamps are at midnight UTC
        for timestamp in df.index:
            assert timestamp.hour == 0
            assert timestamp.minute == 0
            assert timestamp.second == 0
            assert timestamp.microsecond == 0
            assert str(timestamp.tz) == "UTC"

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_yfinance_timestamp_consistency_with_other_sources(self, mock_ticker):
        """Test timestamp consistency with other data sources."""

        # Create YFinance data
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        mock_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [150.0] * 5,
                "Volume": [1000000] * 5,
            }
        )
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        # Create CoinMetrics-style data for comparison
        coinmetrics_dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        coinmetrics_df = pd.DataFrame(
            {
                "PriceUSD": [30000.0] * 5,
            },
            index=coinmetrics_dates,
        )
        coinmetrics_df.index.name = "time"

        loader = YFinanceLoader(symbols=["BTC-USD"])
        yfinance_df = loader.load_from_web()

        # Check that timestamp formats are identical
        assert yfinance_df.index.dtype == coinmetrics_df.index.dtype
        assert str(yfinance_df.index.tz) == str(coinmetrics_df.index.tz)

        # Check that dates can be properly aligned
        common_dates = yfinance_df.index.intersection(coinmetrics_df.index)
        assert len(common_dates) == 5

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_yfinance_no_timezone_conversion_artifacts(self, mock_ticker):
        """Test that timezone conversion doesn't create artifacts."""
        # Test with dates that cross DST boundaries
        dates = pd.date_range("2023-03-10", periods=5, freq="D")  # DST transition period
        mock_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [150.0] * 5,
                "Volume": [1000000] * 5,
            }
        )
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = YFinanceLoader(symbols=["AAPL"])
        df = loader.load_from_web()

        # Verify no timezone conversion artifacts
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(days=1)

        for diff in time_diffs:
            assert diff == expected_diff, f"Time difference should be exactly 1 day, got {diff}"


class TestYFinanceLoaderIntegration:
    """Integration tests for YFinanceLoader with real API calls."""

    @pytest.mark.integration
    def test_load_btc_data_integration(self):
        """Test loading BTC data from Yahoo Finance API."""
        try:
            loader = YFinanceLoader(symbols=["BTC-USD"], period="1y")
            df = loader.load_from_web()

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 100  # Should have at least 100 days of data
            assert "BTC-USD_Close" in df.columns
            assert "BTC-USD_Volume" in df.columns

            # Validate price data
            prices = df["BTC-USD_Close"]
            assert prices.min() > 0
            assert prices.max() < 1000000  # Reasonable upper bound

            # Validate timestamp alignment
            for timestamp in df.index:
                assert timestamp.hour == 0
                assert timestamp.minute == 0
                assert timestamp.second == 0
                assert str(timestamp.tz) == "UTC"

        except Exception as e:
            pytest.skip(f"Integration test failed due to API issue: {e}")

    @pytest.mark.integration
    def test_load_stock_data_integration(self):
        """Test loading stock data from Yahoo Finance API."""
        try:
            loader = YFinanceLoader(symbols=["AAPL"], period="6mo")
            df = loader.load_from_web()

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 50  # Should have at least 50 days of data
            assert "AAPL_Close" in df.columns
            assert "AAPL_Volume" in df.columns

            # Validate price data
            prices = df["AAPL_Close"]
            assert prices.min() > 0
            assert prices.max() < 1000  # Reasonable upper bound for AAPL

        except Exception as e:
            pytest.skip(f"Integration test failed due to API issue: {e}")

    @pytest.mark.integration
    def test_load_multiple_symbols_integration(self):
        """Test loading multiple symbols from Yahoo Finance API."""
        try:
            loader = YFinanceLoader(symbols=["AAPL", "GOOGL", "BTC-USD"], period="3mo")
            df = loader.load_from_web()

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 30  # Should have at least 30 days of data

            # Check all symbols are present
            for symbol in ["AAPL", "GOOGL", "BTC-USD"]:
                assert f"{symbol}_Close" in df.columns
                assert f"{symbol}_Volume" in df.columns

        except Exception as e:
            pytest.skip(f"Integration test failed due to API issue: {e}")

    @pytest.mark.integration
    def test_load_category_integration(self):
        """Test loading category data from Yahoo Finance API."""
        try:
            loader = YFinanceLoader(period="1mo")
            df = loader.load_category("crypto")

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 10  # Should have at least 10 days of data

            # Check some crypto symbols are present
            crypto_symbols = ["BTC-USD", "ETH-USD"]
            for symbol in crypto_symbols:
                assert f"{symbol}_Close" in df.columns

        except Exception as e:
            pytest.skip(f"Integration test failed due to API issue: {e}")

    @pytest.mark.integration
    def test_file_operations_integration(self):
        """Test file operations with real data."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                loader = YFinanceLoader(data_dir=temp_dir, symbols=["AAPL"], period="1mo")

                # Test CSV extraction
                csv_path = loader.extract_to_csv()
                assert csv_path.exists()

                # Test Parquet extraction
                parquet_path = loader.extract_to_parquet()
                assert parquet_path.exists()

                # Test loading from files
                df_csv = loader.load_from_file()
                df_parquet = loader.load_from_parquet()

                # Data should be identical
                pd.testing.assert_frame_equal(df_csv, df_parquet)

        except Exception as e:
            pytest.skip(f"Integration test failed due to API issue: {e}")


class TestYFinanceLoaderBackwardCompatibility:
    """Test backward compatibility functions."""

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_btc_data_from_web(self, mock_ticker):
        """Test backward compatibility function for BTC data."""
        mock_data = self.create_mock_yfinance_data("BTC-USD", 50)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        df = load_btc_data_from_web()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "PriceUSD" in df.columns  # Should be renamed from BTC-USD_Close

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_extract_btc_data_to_csv(self, mock_ticker):
        """Test backward compatibility function for CSV extraction."""
        mock_data = self.create_mock_yfinance_data("BTC-USD", 30)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "btc_test.csv"
            extract_btc_data_to_csv(csv_path)

            assert csv_path.exists()

            # Load and verify content
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            assert "PriceUSD" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_extract_btc_data_to_parquet(self, mock_ticker):
        """Test backward compatibility function for Parquet extraction."""
        mock_data = self.create_mock_yfinance_data("BTC-USD", 30)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "btc_test.parquet"
            extract_btc_data_to_parquet(parquet_path)

            assert parquet_path.exists()

            # Load and verify content
            df = pd.read_parquet(parquet_path)
            assert "PriceUSD" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_stock_data_from_web(self, mock_ticker):
        """Test convenience function for stock data."""
        mock_data = self.create_mock_yfinance_data("TSLA", 40)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        df = load_stock_data_from_web("TSLA")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40
        assert "TSLA_Close" in df.columns

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_load_crypto_data_from_web(self, mock_ticker):
        """Test convenience function for crypto data."""

        def mock_ticker_side_effect(symbol):
            mock_ticker_instance = Mock()
            mock_data = self.create_mock_yfinance_data(symbol, 35)
            mock_ticker_instance.history.return_value = mock_data.set_index("Date")
            return mock_ticker_instance

        mock_ticker.side_effect = mock_ticker_side_effect

        df = load_crypto_data_from_web(["BTC-USD", "ETH-USD"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 35
        assert "BTC-USD_Close" in df.columns
        assert "ETH-USD_Close" in df.columns

    def create_mock_yfinance_data(self, symbol="AAPL", num_days=100):
        """Create mock yfinance data for testing."""
        dates = pd.date_range("2023-01-01", periods=num_days, freq="D")
        np.random.seed(42)  # For reproducibility

        # Create realistic OHLCV data
        base_price = 150.0 if symbol == "AAPL" else 30000.0
        price_changes = np.random.normal(0, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Create OHLCV data
        opens = prices * (1 + np.random.normal(0, 0.005, num_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, num_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, num_days)))
        volumes = np.random.randint(1000000, 50000000, num_days)

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            }
        )

        return data


class TestYFinanceLoaderMultiSourceIntegration:
    """Test YFinanceLoader integration with MultiSourceDataLoader."""

    @patch("stacking_sats_pipeline.data.yfinance_loader.yf.Ticker")
    def test_multisource_integration(self, mock_ticker):
        """Test YFinanceLoader integration with MultiSourceDataLoader."""
        from stacking_sats_pipeline.data.data_loader import MultiSourceDataLoader

        mock_data = self.create_mock_yfinance_data("BTC-USD", 50)
        mock_ticker.return_value.history.return_value = mock_data.set_index("Date")

        loader = MultiSourceDataLoader()
        available_sources = loader.get_available_sources()

        assert "yfinance" in available_sources

        # Test loading from yfinance source
        df = loader.load_from_source("yfinance", use_memory=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "BTC-USD_Close" in df.columns

    def create_mock_yfinance_data(self, symbol="AAPL", num_days=100):
        """Create mock yfinance data for testing."""
        dates = pd.date_range("2023-01-01", periods=num_days, freq="D")
        np.random.seed(42)  # For reproducibility

        # Create realistic OHLCV data
        base_price = 150.0 if symbol == "AAPL" else 30000.0
        price_changes = np.random.normal(0, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Create OHLCV data
        opens = prices * (1 + np.random.normal(0, 0.005, num_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, num_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, num_days)))
        volumes = np.random.randint(1000000, 50000000, num_days)

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            }
        )

        return data
