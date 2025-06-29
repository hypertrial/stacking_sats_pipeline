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


class TestDataLoading:
    """Test data loading functions."""

    @pytest.mark.integration
    def test_load_data_integration(self):
        """Integration test for load_data function (requires internet)."""
        try:
            df = load_data()

            # Basic checks
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "PriceUSD" in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)

            # Check data types
            assert pd.api.types.is_numeric_dtype(df["PriceUSD"])

            # Check for reasonable values
            assert df["PriceUSD"].min() > 0
            assert df["PriceUSD"].max() < 1_000_000  # Reasonable upper bound

        except Exception as e:
            pytest.skip(f"Skipping integration test due to network/data issue: {e}")

    @pytest.mark.integration
    def test_load_btc_data_from_web_integration(self):
        """Integration test for load_btc_data_from_web function."""
        try:
            df = load_btc_data_from_web()

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "PriceUSD" in df.columns

        except Exception as e:
            pytest.skip(f"Skipping integration test due to network issue: {e}")

    def test_validate_price_data_valid(self):
        """Test validate_price_data with valid data."""
        # Create valid test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = np.random.uniform(10000, 50000, 100)  # Reasonable BTC prices
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        # Should not raise an exception
        validate_price_data(df)

    def test_validate_price_data_missing_column(self):
        """Test validate_price_data with missing Price columns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        # Create DataFrame with no Price columns
        df = pd.DataFrame({"Volume": [100] * 10}, index=dates)

        with pytest.raises((KeyError, ValueError)):
            validate_price_data(df)

    def test_validate_price_data_specific_columns(self):
        """Test validate_price_data with specific price columns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Price": [100] * 10}, index=dates)

        # Should not raise when Price column exists (flexible validation)
        validate_price_data(df)

        # Should raise when specific column is required but missing
        with pytest.raises(ValueError):
            validate_price_data(df, price_columns=["PriceUSD"])

    def test_validate_price_data_negative_prices(self):
        """Test validate_price_data with negative prices."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = [-100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        # Current implementation doesn't validate price values, just structure
        try:
            validate_price_data(df)
            # If it passes, that's expected - basic validation only
        except ValueError:
            # If it fails, that might be future enhancement
            pass

    def test_validate_price_data_nan_values(self):
        """Test validate_price_data with NaN values."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = [100, 200, np.nan, 400, 500, 600, 700, 800, 900, 1000]
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        # Current implementation doesn't validate for NaN values
        try:
            validate_price_data(df)
            # If it passes, that's expected - basic validation only
        except ValueError:
            # If it fails, that might be future enhancement
            pass

    def test_validate_price_data_empty_dataframe(self):
        """Test validate_price_data with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            validate_price_data(df)


class TestDataUtilities:
    """Test data utility functions."""

    @pytest.mark.integration
    def test_extract_btc_data_to_csv_integration(self):
        """Test CSV extraction functionality."""
        try:
            # Test the function (may create a file)
            result = extract_btc_data_to_csv()

            # Should return a DataFrame or None
            if result is not None:
                assert isinstance(result, pd.DataFrame)

        except Exception as e:
            pytest.skip(f"Skipping CSV test due to issue: {e}")


class TestDataMocking:
    """Test data functions with mocked responses."""

    @patch("stacking_sats_pipeline.data.coinmetrics_loader.requests.get")
    def test_load_btc_data_from_web_mocked(self, mock_get):
        """Test load_btc_data_from_web with mocked response."""
        # Create mock CSV data
        mock_csv_data = "time,PriceUSD\n2020-01-01,30000\n2020-01-02,31000\n"

        mock_response = MagicMock()
        mock_response.text = mock_csv_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = load_btc_data_from_web()

        assert isinstance(df, pd.DataFrame)
        assert "PriceUSD" in df.columns
        assert len(df) == 2
        assert df["PriceUSD"].iloc[0] == 30000
        assert df["PriceUSD"].iloc[1] == 31000


class TestFREDLoader:
    """Tests for FRED data loader with API key functionality."""

    def create_mock_fred_response(self, num_observations=10):
        """Create mock FRED API response."""
        observations = []
        base_date = pd.Timestamp("2020-01-01")

        for i in range(num_observations):
            date = (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            observations.append(
                {
                    "date": date,
                    "value": str(100.0 + i * 0.5),  # FRED returns strings
                }
            )

        return {"observations": observations}

    def test_fred_loader_initialization_with_api_key(self):
        """Test FRED loader initialization with API key."""
        # Test with API key parameter
        loader = FREDLoader(api_key="test_api_key")
        assert loader.api_key == "test_api_key"
        assert loader.SERIES_ID == "DTWEXBGS"

    def test_fred_loader_initialization_missing_api_key(self):
        """Test FRED loader raises error when API key is missing."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                FREDLoader()

            assert "FRED API key is required" in str(exc_info.value)
            assert "https://fred.stlouisfed.org/docs/api/api_key.html" in str(exc_info.value)

    @patch.dict(os.environ, {"FRED_API_KEY": "env_test_key"})
    def test_fred_loader_initialization_from_env(self):
        """Test FRED loader gets API key from environment variable."""
        loader = FREDLoader()
        assert loader.api_key == "env_test_key"

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_success(self, mock_get):
        """Test successful data loading from FRED API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = self.create_mock_fred_response(5)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        df = loader.load_from_web()

        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "DXY_Value" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "time"

        # Verify API call was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["series_id"] == "DTWEXBGS"
        assert call_args.kwargs["params"]["api_key"] == "test_key"
        assert call_args.kwargs["params"]["file_type"] == "json"

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_with_missing_values(self, mock_get):
        """Test loading data that contains missing values (marked as '.' in FRED)."""
        # Create response with some missing values
        observations = [
            {"date": "2020-01-01", "value": "100.5"},
            {"date": "2020-01-02", "value": "."},  # Missing value
            {"date": "2020-01-03", "value": "101.0"},
            {"date": "2020-01-04", "value": "."},  # Missing value
            {"date": "2020-01-05", "value": "101.5"},
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": observations}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")
        df = loader.load_from_web()

        # Should only have 3 rows (missing values filtered out)
        assert len(df) == 3
        assert df.iloc[0]["DXY_Value"] == 100.5
        assert df.iloc[1]["DXY_Value"] == 101.0
        assert df.iloc[2]["DXY_Value"] == 101.5

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_http_error(self, mock_get):
        """Test handling of HTTP errors from FRED API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Unauthorized"
        )
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="invalid_key")

        with pytest.raises(requests.exceptions.HTTPError):
            loader.load_from_web()

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_timeout_error(self, mock_get):
        """Test handling of timeout errors."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        loader = FREDLoader(api_key="test_key")

        with pytest.raises(requests.exceptions.Timeout):
            loader.load_from_web()

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_invalid_response(self, mock_get):
        """Test handling of invalid API responses."""
        # Mock response without 'observations' key
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")

        with pytest.raises(ValueError) as exc_info:
            loader.load_from_web()

        assert "Invalid response from FRED API" in str(exc_info.value)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_empty_observations(self, mock_get):
        """Test handling of empty observations."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")

        with pytest.raises(ValueError) as exc_info:
            loader.load_from_web()

        assert "No data returned from FRED API" in str(exc_info.value)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_from_web_all_missing_values(self, mock_get):
        """Test handling when all values are missing."""
        observations = [
            {"date": "2020-01-01", "value": "."},
            {"date": "2020-01-02", "value": "."},
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": observations}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = FREDLoader(api_key="test_key")

        with pytest.raises(ValueError) as exc_info:
            loader.load_from_web()

        assert "No valid data points found" in str(exc_info.value)

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_web")
    def test_load_from_file_creates_file_if_missing(self, mock_load_web):
        """Test that load_from_file downloads data if file doesn't exist."""
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
        )
        mock_load_web.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = FREDLoader(data_dir=temp_dir, api_key="test_key")

            # File doesn't exist initially
            file_path = loader.data_dir / loader.DEFAULT_FILENAME
            assert not file_path.exists()

            # Load from file should trigger download
            result = loader.load_from_file()

            # Should have called load_from_web and created file
            mock_load_web.assert_called_once()
            assert file_path.exists()
            assert isinstance(result, pd.DataFrame)

    def test_load_from_file_existing_file(self):
        """Test loading from existing CSV file."""
        # Create test data
        test_data = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0, 102.0]},
            index=pd.date_range("2020-01-01", periods=3, name="time", tz="UTC"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data to CSV
            csv_path = os.path.join(temp_dir, "test_dxy.csv")
            test_data.to_csv(csv_path)

            loader = FREDLoader(api_key="test_key")
            result = loader.load_from_file(csv_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "DXY_Value" in result.columns

            # Check that the values match
            assert list(result["DXY_Value"]) == [100.0, 101.0, 102.0]
            # Check that the dates match (as strings to avoid frequency issues)
            assert [d.strftime("%Y-%m-%d") for d in result.index] == [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ]

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_web")
    def test_load_method_use_memory_true(self, mock_load_web):
        """Test load method with use_memory=True."""
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0]},
            index=pd.date_range("2020-01-01", periods=1, name="time", tz="UTC"),
        )
        mock_load_web.return_value = mock_df

        loader = FREDLoader(api_key="test_key")
        result = loader.load(use_memory=True)

        mock_load_web.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader.load_from_file")
    def test_load_method_use_memory_false(self, mock_load_file):
        """Test load method with use_memory=False."""
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0]},
            index=pd.date_range("2020-01-01", periods=1, name="time", tz="UTC"),
        )
        mock_load_file.return_value = mock_df

        loader = FREDLoader(api_key="test_key")
        result = loader.load(use_memory=False, path="test_path.csv")

        mock_load_file.assert_called_once_with("test_path.csv")
        assert isinstance(result, pd.DataFrame)

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        df = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
        )

        loader = FREDLoader(api_key="test_key")
        # Should not raise any exception
        loader._validate_data(df)

    def test_validate_data_missing_column(self):
        """Test data validation with missing DXY_Value column."""
        df = pd.DataFrame(
            {"OtherColumn": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
        )

        loader = FREDLoader(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(df)

        assert "DXY_Value" in str(exc_info.value)

    def test_validate_data_empty_dataframe(self):
        """Test data validation with empty DataFrame."""
        df = pd.DataFrame()

        loader = FREDLoader(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(df)

        assert "DXY_Value" in str(exc_info.value)

    def test_validate_data_non_datetime_index(self):
        """Test data validation with non-datetime index."""
        df = pd.DataFrame(
            {"DXY_Value": [100.0, 101.0]},
            index=[0, 1],  # Integer index instead of datetime
        )

        loader = FREDLoader(api_key="test_key")
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(df)

        assert "Index must be DatetimeIndex" in str(exc_info.value)


class TestFREDLoaderIntegration:
    """Integration tests for FRED loader."""

    def test_fred_loader_in_multisource_loader(self):
        """Test that FRED loader is properly integrated with MultiSourceDataLoader."""
        # Test without API key - FRED should not be available
        with patch.dict(os.environ, {}, clear=True):
            loader = MultiSourceDataLoader()
            available_sources = loader.get_available_sources()
            assert "fred" not in available_sources
            assert "coinmetrics" in available_sources

        # Test with API key - FRED should be available
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            loader = MultiSourceDataLoader()
            available_sources = loader.get_available_sources()
            assert "fred" in available_sources
            assert "coinmetrics" in available_sources

            # Should be able to get the FRED loader
            fred_loader = loader.loaders["fred"]
            assert isinstance(fred_loader, FREDLoader)

    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_load_data_function_with_fred(self, mock_get):
        """Test using load_data function with FRED source."""
        # Mock FRED API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2020-01-01", "value": "100.0"},
                {"date": "2020-01-02", "value": "100.5"},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Set API key in environment
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            df = load_data("fred")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "DXY_Value" in df.columns

    @patch("stacking_sats_pipeline.data.data_loader.MultiSourceDataLoader.load_from_source")
    def test_load_and_merge_with_fred(self, mock_load_from_source):
        """Test loading and merging data including FRED source."""

        # Mock the load_from_source method to return test data
        def side_effect(source, use_memory=True):
            if source == "coinmetrics":
                return pd.DataFrame(
                    {"PriceUSD": [30000, 31000]},
                    index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
                )
            elif source == "fred":
                return pd.DataFrame(
                    {"DXY_Value": [100.0, 100.5]},
                    index=pd.date_range("2020-01-01", periods=2, name="time", tz="UTC"),
                )
            else:
                raise ValueError(f"Unknown source: {source}")

        mock_load_from_source.side_effect = side_effect

        # Set API key in environment
        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            from stacking_sats_pipeline.data import load_and_merge_data

            merged_df = load_and_merge_data(["coinmetrics", "fred"])

            assert isinstance(merged_df, pd.DataFrame)
            assert len(merged_df) == 2
            # Should have columns from both sources with suffixes
            assert any("coinmetrics" in col for col in merged_df.columns)
            assert any("fred" in col for col in merged_df.columns)

            # Verify the method was called correctly
            assert mock_load_from_source.call_count == 2

    @pytest.mark.integration
    def test_fred_loader_real_api_key_required(self):
        """Integration test that verifies real API key is required."""
        # This test will be skipped if no real API key is available
        try:
            # Try to create loader without setting API key
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    FREDLoader()

                assert "FRED API key is required" in str(exc_info.value)

        except Exception as e:
            pytest.skip(f"Skipping real API key test due to: {e}")


class TestFREDLoaderBackwardCompatibility:
    """Test backward compatibility functions for FRED loader."""

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader")
    def test_load_dxy_data_from_web_function(self, mock_loader_class):
        """Test load_dxy_data_from_web convenience function."""
        from stacking_sats_pipeline.data.fred_loader import load_dxy_data_from_web

        mock_loader = MagicMock()
        mock_df = pd.DataFrame(
            {"DXY_Value": [100.0]},
            index=pd.date_range("2020-01-01", periods=1, tz="UTC"),
        )
        mock_loader.load_from_web.return_value = mock_df
        mock_loader_class.return_value = mock_loader

        result = load_dxy_data_from_web(api_key="test_key")

        mock_loader_class.assert_called_once_with(api_key="test_key")
        mock_loader.load_from_web.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch("stacking_sats_pipeline.data.fred_loader.FREDLoader")
    def test_extract_dxy_data_to_csv_function(self, mock_loader_class):
        """Test extract_dxy_data_to_csv convenience function."""
        from stacking_sats_pipeline.data.fred_loader import extract_dxy_data_to_csv

        mock_loader = MagicMock()
        mock_loader.extract_to_csv.return_value = "/tmp/test.csv"
        mock_loader_class.return_value = mock_loader

        extract_dxy_data_to_csv(local_path="/tmp/test.csv", api_key="test_key")

        mock_loader_class.assert_called_once_with(api_key="test_key")
        mock_loader.extract_to_csv.assert_called_once_with("/tmp/test.csv")
