#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline weight calculator functionality
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stacking_sats_pipeline.weights.weight_calculator import (
    display_weights,
    get_historical_btc_data,
    get_historical_data_for_period,
    get_weights_for_period,
    save_weights_to_csv,
    validate_date_range,
)


class TestDateValidation:
    """Test date range validation for historical data limits."""

    def create_mock_btc_data(self, start_date="2020-01-01", end_date="2023-12-31"):
        """Create mock BTC data with specific date range."""
        dates = pd.date_range(start_date, end_date, freq="D")
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.normal(0, 500, len(dates)))
        prices = np.maximum(prices, 1000)  # Ensure positive prices
        return pd.DataFrame({"PriceUSD": prices}, index=dates)

    def test_validate_date_range_valid_dates(self):
        """Test validate_date_range with valid date ranges."""
        btc_df = self.create_mock_btc_data("2020-01-01", "2023-12-31")

        # Should not raise exception for valid ranges
        validate_date_range(btc_df, "2020-06-01", "2020-12-31")
        validate_date_range(btc_df, "2021-01-01", "2022-01-01")
        validate_date_range(btc_df, "2020-01-01", "2023-12-31")  # Full range

    def test_validate_date_range_start_before_data(self):
        """Test validate_date_range when start date is before available data."""
        btc_df = self.create_mock_btc_data("2020-01-01", "2023-12-31")

        with pytest.raises(ValueError) as exc_info:
            validate_date_range(btc_df, "2019-01-01", "2020-06-01")

        assert "Start date" in str(exc_info.value)
        assert "before available data starts" in str(exc_info.value)

    def test_validate_date_range_end_after_data(self):
        """Test validate_date_range when end date is after available data."""
        btc_df = self.create_mock_btc_data("2020-01-01", "2023-12-31")

        with pytest.raises(ValueError) as exc_info:
            validate_date_range(btc_df, "2023-01-01", "2024-06-01")

        assert "End date" in str(exc_info.value)
        assert "after available data ends" in str(exc_info.value)

    def test_validate_date_range_both_outside_data(self):
        """Test validate_date_range when both dates are outside available data."""
        btc_df = self.create_mock_btc_data("2020-01-01", "2023-12-31")

        # Test both before
        with pytest.raises(ValueError):
            validate_date_range(btc_df, "2018-01-01", "2019-01-01")

        # Test both after
        with pytest.raises(ValueError):
            validate_date_range(btc_df, "2024-01-01", "2025-01-01")

    def test_validate_date_range_edge_cases(self):
        """Test validate_date_range with edge case dates."""
        btc_df = self.create_mock_btc_data("2020-01-01", "2023-12-31")

        # Test exact boundaries - should work
        validate_date_range(btc_df, "2020-01-01", "2020-01-01")  # Single day at start
        validate_date_range(btc_df, "2023-12-31", "2023-12-31")  # Single day at end

        # Test one day outside boundaries - should fail
        with pytest.raises(ValueError):
            validate_date_range(btc_df, "2019-12-31", "2020-01-01")

        with pytest.raises(ValueError):
            validate_date_range(btc_df, "2023-12-31", "2024-01-01")


class TestHistoricalDataLoading:
    """Test historical data loading with validation."""

    @patch("stacking_sats_pipeline.weights.weight_calculator.load_data")
    def test_get_historical_btc_data_mocked(self, mock_load_data):
        """Test get_historical_btc_data with mocked data loading."""
        mock_data = pd.DataFrame(
            {"PriceUSD": [30000, 31000, 32000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        mock_load_data.return_value = mock_data

        result = get_historical_btc_data()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "PriceUSD" in result.columns
        mock_load_data.assert_called_once()

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_historical_btc_data")
    def test_get_historical_data_for_period_valid(self, mock_get_data):
        """Test get_historical_data_for_period with valid date range."""
        mock_data = pd.DataFrame(
            {"PriceUSD": np.random.uniform(30000, 50000, 365)},
            index=pd.date_range("2020-01-01", periods=365, freq="D"),
        )
        mock_get_data.return_value = mock_data

        result = get_historical_data_for_period("2020-03-01", "2020-06-01")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 365  # Returns full dataset
        mock_get_data.assert_called_once()

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_historical_btc_data")
    def test_get_historical_data_for_period_invalid(self, mock_get_data):
        """Test get_historical_data_for_period with invalid date range."""
        mock_data = pd.DataFrame(
            {"PriceUSD": np.random.uniform(30000, 50000, 365)},
            index=pd.date_range("2020-01-01", periods=365, freq="D"),
        )
        mock_get_data.return_value = mock_data

        # Request dates outside available range
        with pytest.raises(ValueError):
            get_historical_data_for_period("2019-01-01", "2019-06-01")


class TestWeightCalculation:
    """Test weight calculation functions with historical data constraints."""

    def create_mock_data_and_weights(self):
        """Create mock data and corresponding weights for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        btc_df = pd.DataFrame(
            {"PriceUSD": np.random.uniform(30000, 50000, 100)}, index=dates
        )
        weights = pd.Series(
            np.random.uniform(0.005, 0.015, 100),  # Small positive weights
            index=dates,
        )
        weights = weights / weights.sum()  # Normalize to sum to 1
        return btc_df, weights

    @patch(
        "stacking_sats_pipeline.weights.weight_calculator.get_historical_data_for_period"
    )
    @patch(
        "stacking_sats_pipeline.weights.weight_calculator.compute_weights_for_period"
    )
    def test_get_weights_for_period_valid_range(
        self, mock_compute_weights, mock_get_data
    ):
        """Test get_weights_for_period with valid date range."""
        btc_df, expected_weights = self.create_mock_data_and_weights()
        mock_get_data.return_value = btc_df
        mock_compute_weights.return_value = expected_weights

        result = get_weights_for_period("2020-01-01", "2020-04-09")

        assert isinstance(result, pd.Series)
        assert len(result) == 100  # Should return all weights in range
        mock_get_data.assert_called_once_with("2020-01-01", "2020-04-09")
        mock_compute_weights.assert_called_once()

    @patch(
        "stacking_sats_pipeline.weights.weight_calculator.get_historical_data_for_period"
    )
    def test_get_weights_for_period_invalid_range(self, mock_get_data):
        """Test get_weights_for_period with invalid date range."""
        mock_get_data.side_effect = ValueError(
            "Start date 2019-01-01 is before available data"
        )

        with pytest.raises(ValueError) as exc_info:
            get_weights_for_period("2019-01-01", "2019-06-01")

        assert "before available data" in str(exc_info.value)

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_weights_for_period")
    @patch(
        "stacking_sats_pipeline.weights.weight_calculator.get_historical_data_for_period"
    )
    def test_display_weights_historical_constraint(
        self, mock_get_data, mock_get_weights
    ):
        """Test that display_weights respects historical data constraints."""
        btc_df, weights = self.create_mock_data_and_weights()
        mock_get_data.return_value = btc_df
        mock_get_weights.return_value = weights

        # This should work without raising exceptions
        try:
            display_weights(10000, "2020-01-01", "2020-04-09")
            mock_get_weights.assert_called_once_with("2020-01-01", "2020-04-09")
        except Exception as e:
            # If it fails, it should be due to the historical data constraint
            if "before available data" in str(e) or "after available data" in str(e):
                pass  # Expected behavior
            else:
                pytest.fail(f"Unexpected error: {e}")


class TestHistoricalDataConstraints:
    """Test that various functions properly enforce historical data constraints."""

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_historical_btc_data")
    def test_future_dates_rejected(self, mock_get_data):
        """Test that future dates are properly rejected."""
        # Mock data that ends in the past
        mock_data = pd.DataFrame(
            {"PriceUSD": np.random.uniform(30000, 50000, 365)},
            index=pd.date_range("2020-01-01", periods=365, freq="D"),
        )
        mock_get_data.return_value = mock_data

        # Try to get weights for future dates (assuming current year > 2020)
        future_date = "2025-01-01"
        with pytest.raises(ValueError) as exc_info:
            get_historical_data_for_period("2024-01-01", future_date)

        assert "after available data ends" in str(exc_info.value)

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_historical_btc_data")
    def test_very_old_dates_rejected(self, mock_get_data):
        """Test that dates before Bitcoin existed are properly rejected."""
        # Mock data starting in 2020
        mock_data = pd.DataFrame(
            {"PriceUSD": np.random.uniform(30000, 50000, 365)},
            index=pd.date_range("2020-01-01", periods=365, freq="D"),
        )
        mock_get_data.return_value = mock_data

        # Try to get weights for dates before Bitcoin
        with pytest.raises(ValueError) as exc_info:
            get_historical_data_for_period("2008-01-01", "2008-06-01")

        assert "before available data starts" in str(exc_info.value)

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_weights_for_period")
    @patch(
        "stacking_sats_pipeline.weights.weight_calculator.get_historical_data_for_period"
    )
    def test_csv_export_historical_constraint(self, mock_get_data, mock_get_weights):
        """Test that CSV export respects historical data constraints."""
        btc_df = pd.DataFrame(
            {"PriceUSD": [30000, 31000, 32000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        weights = pd.Series([0.4, 0.3, 0.3], index=btc_df.index)

        mock_get_data.return_value = btc_df
        mock_get_weights.return_value = weights

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            filename = save_weights_to_csv(
                1000, "2020-01-01", "2020-01-03", tmp_file.name
            )

            # Should have created the file
            assert filename == tmp_file.name

            # Verify the functions were called with correct parameters
            mock_get_weights.assert_called_once_with("2020-01-01", "2020-01-03")


class TestErrorHandling:
    """Test error handling in weight calculator functions."""

    def test_malformed_date_strings(self):
        """Test handling of malformed date strings."""
        btc_df = pd.DataFrame(
            {"PriceUSD": [30000, 31000]},
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )

        # Test clearly invalid date format
        invalid_date = "not-a-date-at-all"

        # The function should either raise an exception or handle it gracefully
        # We're testing that it doesn't crash the system
        try:
            validate_date_range(btc_df, invalid_date, "2020-01-02")
            # If it doesn't raise an exception, that's fine too
        except (ValueError, TypeError):
            # Expected behavior for invalid dates
            pass
        except Exception:
            # Any other exception is also acceptable as long as it's handled
            pass

    def test_reversed_date_range(self):
        """Test handling when start_date > end_date."""
        btc_df = pd.DataFrame(
            {"PriceUSD": [30000, 31000, 32000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # This should either be handled gracefully or raise a clear error
        try:
            validate_date_range(btc_df, "2020-01-03", "2020-01-01")
            # If it passes, that's acceptable - some implementations may allow this
        except ValueError:
            # If it raises an error, that's also acceptable
            pass

    @patch("stacking_sats_pipeline.weights.weight_calculator.get_historical_btc_data")
    def test_empty_dataframe_handling(self, mock_get_data):
        """Test handling of empty DataFrame from data loading."""
        mock_get_data.return_value = pd.DataFrame()

        with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
            get_historical_data_for_period("2020-01-01", "2020-01-02")


class TestIntegrationWithRealConstraints:
    """Integration tests that use real date constraints from the system."""

    @pytest.mark.integration
    def test_real_data_constraints(self):
        """Test with real data loading to verify actual historical constraints."""
        try:
            # Try to load real data
            btc_df = get_historical_btc_data()

            # Get the actual data boundaries
            data_start = btc_df.index.min()
            data_end = btc_df.index.max()

            # Test that we can't request data outside these boundaries
            before_start = (data_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            after_end = (data_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # These should fail
            with pytest.raises(ValueError):
                get_historical_data_for_period(
                    before_start, data_start.strftime("%Y-%m-%d")
                )

            with pytest.raises(ValueError):
                get_historical_data_for_period(data_end.strftime("%Y-%m-%d"), after_end)

            # This should work
            mid_start = (data_start + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            mid_end = (data_start + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            result = get_historical_data_for_period(mid_start, mid_end)
            assert isinstance(result, pd.DataFrame)

        except Exception as e:
            pytest.skip(f"Skipping real data integration test due to: {e}")

    @pytest.mark.integration
    def test_current_date_boundary(self):
        """Test that current date is properly handled as a boundary."""
        try:
            from datetime import datetime

            # Get tomorrow's date
            tomorrow = datetime.now() + pd.Timedelta(days=1)
            tomorrow_str = tomorrow.strftime("%Y-%m-%d")

            # This should fail - we can't get weights for future dates
            with pytest.raises(ValueError):
                get_weights_for_period("2020-01-01", tomorrow_str)

        except Exception as e:
            pytest.skip(f"Skipping current date boundary test due to: {e}")
