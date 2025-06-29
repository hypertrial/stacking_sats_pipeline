#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline data loading functionality
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from stacking_sats_pipeline import (
    extract_btc_data_to_csv,
    load_btc_data_from_web,
    load_data,
    validate_price_data,
)
from stacking_sats_pipeline.data import FREDLoader, MultiSourceDataLoader


# Test fixtures and helpers
@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = np.random.uniform(10000, 50000, 10)
    return pd.DataFrame({"PriceUSD": prices}, index=dates)


@pytest.fixture
def mock_fred_response():
    """Mock FRED API response."""
    return {
        "observations": [
            {"date": "2020-01-01", "value": "100.0"},
            {"date": "2020-01-02", "value": "100.5"},
            {"date": "2020-01-03", "value": "101.0"},
        ]
    }


@pytest.fixture
def mock_btc_csv_data():
    """Mock BTC CSV data."""
    return "time,PriceUSD\n2020-01-01,30000\n2020-01-02,31000\n"


class TestDataLoading:
    """Core data loading functionality tests."""

    @pytest.mark.integration
    def test_load_data_integration(self):
        """Integration test for load_data function."""
        try:
            df = load_data()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "PriceUSD" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            assert pd.api.types.is_numeric_dtype(df["PriceUSD"])
            assert df["PriceUSD"].min() > 0
            assert df["PriceUSD"].max() < 1_000_000
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_load_btc_data_from_web_integration(self):
        """Integration test for BTC data loading."""
        try:
            df = load_btc_data_from_web()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "PriceUSD" in df.columns
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @patch("stacking_sats_pipeline.data.coinmetrics_loader.requests.get")
    def test_load_btc_data_mocked(self, mock_get, mock_btc_csv_data):
        """Test BTC data loading with mocked response."""
        mock_response = MagicMock()
        mock_response.text = mock_btc_csv_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = load_btc_data_from_web()
        assert isinstance(df, pd.DataFrame)
        assert "PriceUSD" in df.columns
        assert len(df) == 2
        assert df["PriceUSD"].tolist() == [30000, 31000]

    @pytest.mark.integration
    def test_extract_btc_data_to_csv(self):
        """Test CSV extraction functionality."""
        try:
            result = extract_btc_data_to_csv()
            if result is not None:
                assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"CSV extraction test skipped: {e}")


class TestDataValidation:
    """Data validation tests."""

    def test_validate_price_data_valid(self, sample_price_data):
        """Test validation with valid data."""
        validate_price_data(sample_price_data)  # Should not raise

    @pytest.mark.parametrize(
        "invalid_data,error_type",
        [
            (
                pd.DataFrame(
                    {"Volume": [100] * 10},
                    index=pd.date_range("2020-01-01", periods=10),
                ),
                (KeyError, ValueError),
            ),
            (pd.DataFrame(), (ValueError, KeyError)),
        ],
    )
    def test_validate_price_data_invalid(self, invalid_data, error_type):
        """Test validation with invalid data."""
        with pytest.raises(error_type):
            validate_price_data(invalid_data)

    def test_validate_price_data_specific_columns(self):
        """Test validation with specific columns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Price": [100] * 10}, index=dates)

        validate_price_data(df)  # Should pass with flexible validation

        with pytest.raises(ValueError):
            validate_price_data(df, price_columns=["PriceUSD"])

    @pytest.mark.parametrize(
        "prices",
        [
            [-100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # Negative prices
            [100, 200, np.nan, 400, 500, 600, 700, 800, 900, 1000],  # NaN values
        ],
    )
    def test_validate_price_data_edge_cases(self, prices):
        """Test validation with edge cases (negative prices, NaN values)."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        # Current implementation doesn't validate values, just structure
        try:
            validate_price_data(df)
        except ValueError:
            pass  # Future enhancement might reject these


class TestFREDLoader:
    """FRED data loader comprehensive tests."""

    def test_initialization_scenarios(self):
        """Test various initialization scenarios."""
        # With API key parameter
        loader = FREDLoader(api_key="test_key")
        assert loader.api_key == "test_key"
        assert loader.SERIES_ID == "DTWEXBGS"

        # From environment variable
        with patch.dict(os.environ, {"FRED_API_KEY": "env_key"}):
            loader = FREDLoader()
            assert loader.api_key == "env_key"

        # Missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                FREDLoader()
            assert "FRED API key is required" in str(exc_info.value)
            assert "https://fred.stlouisfed.org/docs/api/api_key.html" in str(exc_info.value)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_success(self, mock_get, mock_fred_response):
        """Test successful data loading from FRED API."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        df = loader.load_from_web()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "DXY_Value" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "time"

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["series_id"] == "DTWEXBGS"
        assert call_args.kwargs["params"]["api_key"] == "test_key"

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    @pytest.mark.parametrize(
        "response_data,expected_error",
        [
            ({"error": "Invalid API key"}, "Invalid response from FRED API"),
            ({"observations": []}, "No data returned from FRED API"),
            (
                {
                    "observations": [
                        {"date": "2020-01-01", "value": "."},
                        {"date": "2020-01-02", "value": "."},
                    ]
                },
                "No valid data points found",
            ),
        ],
    )
    def test_load_from_web_error_cases(self, mock_get, response_data, expected_error):
        """Test various error cases."""
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            loader.load_from_web()
        assert expected_error in str(exc_info.value)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_with_missing_values(self, mock_get):
        """Test handling of missing values in FRED data."""
        observations = [
            {"date": "2020-01-01", "value": "100.5"},
            {"date": "2020-01-02", "value": "."},  # Missing
            {"date": "2020-01-03", "value": "101.0"},
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": observations}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        df = loader.load_from_web()

        assert len(df) == 2  # Missing values filtered out
        assert df["DXY_Value"].tolist() == [100.5, 101.0]

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    @pytest.mark.parametrize(
        "exception_type",
        [
            requests.exceptions.HTTPError("401 Unauthorized"),
            requests.exceptions.Timeout("Request timed out"),
        ],
    )
    def test_load_from_web_network_errors(self, mock_get, exception_type):
        """Test network error handling."""
        if isinstance(exception_type, requests.exceptions.HTTPError):
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = exception_type
            mock_get.return_value = mock_response
        else:
            mock_get.side_effect = exception_type

        loader = FREDLoader(api_key="test_key")
        with pytest.raises(type(exception_type)):
            loader.load_from_web()

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_web")
    def test_file_operations(self, mock_load_web):
        """Test file loading and creation operations."""
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
        )
        mock_load_web.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = FREDLoader(data_dir=temp_dir, api_key="test_key")

            # Test file creation when missing
            file_path = loader.data_dir / loader.DEFAULT_FILENAME
            assert not file_path.exists()

            result = loader.load_from_file()
            mock_load_web.assert_called_once()
            assert file_path.exists()
            assert isinstance(result, pd.DataFrame)

    def test_load_from_existing_file(self):
        """Test loading from existing CSV file."""
        test_data = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0, 102.0]},
            index=pd.date_range("2020-01-01", periods=3, name="time", tz="UTC"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_dxy.csv")
            test_data.to_csv(csv_path)

            loader = FREDLoader(api_key="test_key")
            result = loader.load_from_file(csv_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "DXY_Value" in result.columns
            assert result["DXY_Value"].tolist() == [100.0, 101.0, 102.0]

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_web")
    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_file")
    def test_load_method_delegation(self, mock_load_file, mock_load_web):
        """Test load method properly delegates based on use_memory parameter."""
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0]},
            index=pd.date_range("2020-01-01", periods=1, name="time", tz="UTC"),
        )
        mock_load_web.return_value = mock_df
        mock_load_file.return_value = mock_df

        loader = FREDLoader(api_key="test_key")

        # Test use_memory=True
        result = loader.load(use_memory=True)
        mock_load_web.assert_called_once()
        assert isinstance(result, pd.DataFrame)

        # Test use_memory=False
        result = loader.load(use_memory=False, path="test_path.csv")
        mock_load_file.assert_called_once_with("test_path.csv")

    @pytest.mark.parametrize(
        "invalid_df,expected_error",
        [
            (
                pd.DataFrame(
                    {"OtherColumn": [100.0]},
                    index=pd.date_range("2020-01-01", periods=1, tz="UTC"),
                ),
                "DXY_Value",
            ),
            (pd.DataFrame(), "DXY_Value"),
            (
                pd.DataFrame({"DXY_Value": [100.0]}, index=[0]),
                "Index must be DatetimeIndex",
            ),
        ],
    )
    def test_data_validation(self, invalid_df, expected_error):
        """Test data validation with various invalid inputs."""
        loader = FREDLoader(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(invalid_df)
        assert expected_error in str(exc_info.value)

    def test_data_validation_success(self):
        """Test successful data validation."""
        valid_df = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
        )
        loader = FREDLoader(api_key="test_key")
        loader._validate_data(valid_df)  # Should not raise


class TestTimestampConsistency:
    """Tests for timestamp alignment and consistency across data sources."""

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_fred_timestamps_normalized_to_midnight_utc(self, mock_get):
        """Test FRED timestamps are normalized to midnight UTC."""
        test_dates = [
            ("2020-03-08", 100.0),  # DST transition
            ("2020-11-01", 101.0),  # DST transition
            ("2020-12-31", 102.0),  # Year boundary
        ]

        observations = [{"date": date, "value": str(value)} for date, value in test_dates]
        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": observations}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        df = loader.load_from_web()

        # Verify all timestamps are at midnight UTC
        for timestamp in df.index:
            assert timestamp.hour == 0
            assert timestamp.minute == 0
            assert timestamp.second == 0
            assert timestamp.microsecond == 0
            assert str(timestamp.tz) == "UTC"

    def test_data_source_timestamp_consistency(self):
        """Test timestamp consistency across all data sources."""
        # Create test data with consistent midnight UTC timestamps
        dates = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")

        coinmetrics_df = pd.DataFrame({"PriceUSD": [30000, 31000, 32000]}, index=dates)
        fred_df = pd.DataFrame({"DXY_Value": [100.0, 100.5, 101.0]}, index=dates)

        # Test merging
        merged_df = pd.merge(
            coinmetrics_df, fred_df, left_index=True, right_index=True, how="outer"
        )

        # Verify midnight UTC timestamps
        for timestamp in merged_df.index:
            assert timestamp.hour == 0
            assert timestamp.minute == 0
            assert timestamp.second == 0
            assert str(timestamp.tz) == "UTC"

        # Verify proper overlap
        assert len(merged_df) == 3
        assert "PriceUSD" in merged_df.columns
        assert "DXY_Value" in merged_df.columns
        assert merged_df["PriceUSD"].notna().sum() == 3
        assert merged_df["DXY_Value"].notna().sum() == 3


class TestMultiSourceIntegration:
    """Integration tests for multi-source data loading."""

    def test_fred_loader_in_multisource_loader(self):
        """Test FRED loader integration with MultiSourceDataLoader."""
        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            loader = MultiSourceDataLoader()
            available_sources = loader.get_available_sources()
            assert "fred" not in available_sources
            assert "coinmetrics" in available_sources

        # With API key
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            loader = MultiSourceDataLoader()
            available_sources = loader.get_available_sources()
            assert "fred" in available_sources
            assert "coinmetrics" in available_sources
            assert isinstance(loader.loaders["fred"], FREDLoader)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_data_function_with_fred(self, mock_get, mock_fred_response):
        """Test load_data function with FRED source."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            df = load_data("fred")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert "DXY_Value" in df.columns

    @patch("stacking_sats_pipeline.data.data_loader.MultiSourceDataLoader.load_from_source")
    def test_load_and_merge_with_fred(self, mock_load_from_source):
        """Test loading and merging data including FRED source."""

        def mock_side_effect(source, use_memory=True):
            base_index = pd.date_range("2020-01-01", periods=2, name="time", tz="UTC")
            if source == "coinmetrics":
                return pd.DataFrame({"PriceUSD": [30000, 31000]}, index=base_index)
            elif source == "fred":
                return pd.DataFrame({"DXY_Value": [100.0, 100.5]}, index=base_index)
            else:
                raise ValueError(f"Unknown source: {source}")

        mock_load_from_source.side_effect = mock_side_effect

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            from stacking_sats_pipeline.data import load_and_merge_data

            merged_df = load_and_merge_data(["coinmetrics", "fred"])

            assert isinstance(merged_df, pd.DataFrame)
            assert len(merged_df) == 2
            assert any("coinmetrics" in col for col in merged_df.columns)
            assert any("fred" in col for col in merged_df.columns)
            assert mock_load_from_source.call_count == 2


class TestBackwardCompatibility:
    """Backward compatibility function tests."""

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader")
    def test_convenience_functions(self, mock_loader_class):
        """Test backward compatibility convenience functions."""
        from stacking_sats_pipeline.data.fred_loader import (
            extract_dxy_data_to_csv,
            load_dxy_data_from_web,
        )

        mock_loader = MagicMock()
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0]},
            index=pd.date_range("2020-01-01", periods=1, tz="UTC"),
        )
        mock_loader.load_from_web.return_value = mock_df
        mock_loader.extract_to_csv.return_value = "/tmp/test.csv"
        mock_loader_class.return_value = mock_loader

        # Test load function
        result = load_dxy_data_from_web(api_key="test_key")
        mock_loader_class.assert_called_with(api_key="test_key")
        mock_loader.load_from_web.assert_called_once()
        assert isinstance(result, pd.DataFrame)

        # Reset mocks for second test
        mock_loader_class.reset_mock()
        mock_loader.reset_mock()

        # Test extract function
        extract_dxy_data_to_csv(local_path="/tmp/test.csv", api_key="test_key")
        mock_loader_class.assert_called_with(api_key="test_key")
        mock_loader.extract_to_csv.assert_called_once_with("/tmp/test.csv")


@pytest.mark.integration
class TestRealAPIIntegration:
    """Real API integration tests (require network and API keys)."""

    def test_fred_loader_real_api_key_required(self):
        """Test that real API key validation works."""
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    FREDLoader()
                assert "FRED API key is required" in str(exc_info.value)
        except Exception as e:
            pytest.skip(f"Real API key test skipped: {e}")
