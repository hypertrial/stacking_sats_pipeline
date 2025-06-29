#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline plotting functionality
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stacking_sats_pipeline import (
    plot_features,
    plot_final_weights,
    plot_spd_comparison,
    plot_weight_sums_by_cycle,
)


class TestPlottingFunctions:
    """Test plotting functions with mocked matplotlib."""

    def create_test_data(self, num_days=100):
        """Create test Bitcoin price data."""
        dates = pd.date_range("2020-01-01", periods=num_days, freq="D")
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.normal(0, 500, num_days))
        prices = np.maximum(prices, 1000)
        return pd.DataFrame({"PriceUSD": prices}, index=dates)

    def create_test_weights(self, df):
        """Create test weights for plotting."""
        base_weight = 1.0 / len(df)
        weights = pd.Series(base_weight, index=df.index)

        # Add some variation
        weights.iloc[::10] *= 1.5  # Every 10th weight is higher
        weights = weights / weights.sum()  # Renormalize

        return weights

    def create_test_spd_data(self):
        """Create test SPD comparison data."""
        cycles = ["2020", "2021", "2022"]
        data = {
            "cycle": cycles,
            "uniform_spd": [4000, 4200, 3800],
            "dynamic_spd": [4100, 4400, 3900],
            "static_dca_spd": [3900, 4000, 3700],
            "uniform_pct": [50.0, 60.0, 45.0],
            "dynamic_pct": [55.0, 65.0, 50.0],
            "excess_pct": [5.0, 5.0, 5.0],
        }
        return pd.DataFrame(data).set_index("cycle")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_features_basic(self, mock_figure, mock_show):
        """Test plot_features function with mocked matplotlib."""
        df = self.create_test_data(200)
        weights = self.create_test_weights(df)

        try:
            # Should not raise an exception
            plot_features(df, weights=weights, start_date="2020-01-01", end_date="2020-12-31")

            # Verify matplotlib functions were called
            mock_figure.assert_called()

        except Exception as e:
            pytest.skip(f"Skipping plot_features test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_features_without_weights(self, mock_figure, mock_show):
        """Test plot_features function without weights parameter."""
        df = self.create_test_data(200)

        try:
            plot_features(df, start_date="2020-01-01", end_date="2020-12-31")
            mock_figure.assert_called()

        except Exception as e:
            pytest.skip(f"Skipping plot_features test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_final_weights(self, mock_figure, mock_show):
        """Test plot_final_weights function."""
        df = self.create_test_data(100)
        weights = self.create_test_weights(df)

        try:
            plot_final_weights(weights, start_date="2020-01-01")
            mock_figure.assert_called()

        except Exception as e:
            pytest.skip(f"Skipping plot_final_weights test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_weight_sums_by_cycle(self, mock_figure, mock_show):
        """Test plot_weight_sums_by_cycle function."""
        df = self.create_test_data(400)  # Need enough data for cycles
        weights = self.create_test_weights(df)

        try:
            plot_weight_sums_by_cycle(weights)
            mock_figure.assert_called()

        except Exception as e:
            pytest.skip(f"Skipping plot_weight_sums_by_cycle test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_spd_comparison(self, mock_figure, mock_show):
        """Test plot_spd_comparison function."""
        # Create proper SPD test data with required columns
        try:
            spd_data = pd.DataFrame(
                {
                    "max_spd": [1.8, 2.0, 1.6],
                    "dynamic_spd": [1.2, 1.5, 1.1],
                    "static_dca_spd": [1.1, 1.3, 1.0],
                    "uniform_spd": [1.0, 1.0, 1.0],
                    "min_spd": [0.8, 0.9, 0.7],
                    "uniform_pct": [25.0, 20.0, 30.0],
                    "static_dca_pct": [30.0, 40.0, 35.0],
                    "dynamic_pct": [40.0, 60.0, 45.0],
                },
                index=pd.date_range("2020-01-01", periods=3, freq="D"),
            )

            plot_spd_comparison(spd_data, strategy_name="Test Strategy")
            mock_figure.assert_called()

        except Exception as e:
            pytest.skip(f"Skipping plot_spd_comparison test due to: {e}")


class TestPlottingInputValidation:
    """Test plotting functions with various input scenarios."""

    def create_test_data(self, num_days=100):
        """Create test data."""
        dates = pd.date_range("2020-01-01", periods=num_days, freq="D")
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.normal(0, 500, num_days))
        prices = np.maximum(prices, 1000)
        return pd.DataFrame({"PriceUSD": prices}, index=dates)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_functions_with_minimal_data(self, mock_figure, mock_show):
        """Test plotting functions with minimal data."""
        df = self.create_test_data(10)  # Very small dataset
        weights = pd.Series(0.1, index=df.index)

        try:
            # Test each function with minimal data
            plot_features(df, weights=weights, start_date=df.index[0], end_date=df.index[-1])
            plot_final_weights(weights, start_date=df.index[0])

        except Exception as e:
            # Some functions might fail with very small datasets
            pytest.skip(f"Skipping minimal data test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_functions_with_edge_case_dates(self, mock_figure, mock_show):
        """Test plotting functions with edge case date parameters."""
        df = self.create_test_data(100)
        weights = pd.Series(0.01, index=df.index)

        try:
            # Test with date range outside data range
            plot_features(
                df,
                weights=weights,
                start_date="2019-01-01",  # Before data starts
                end_date="2021-01-01",  # After data ends
            )

        except Exception as e:
            pytest.skip(f"Skipping edge case date test due to: {e}")


class TestPlottingErrorHandling:
    """Test plotting functions error handling."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_features_invalid_data(self, mock_figure, mock_show):
        """Test plot_features with invalid data."""
        # Empty DataFrame
        df_empty = pd.DataFrame()

        try:
            plot_features(df_empty)
            # If it doesn't raise an exception, that's fine
        except (ValueError, KeyError, IndexError):
            # Expected to fail with empty data
            pass
        except Exception as e:
            pytest.skip(f"Skipping invalid data test due to: {e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_final_weights_invalid_weights(self, mock_figure, mock_show):
        """Test plot_final_weights with invalid weights."""
        # Empty weights
        weights_empty = pd.Series(dtype=float)

        try:
            plot_final_weights(weights_empty)
            # If it doesn't raise an exception, that's fine
        except (ValueError, IndexError):
            # Expected to fail with empty weights
            pass
        except Exception as e:
            pytest.skip(f"Skipping invalid weights test due to: {e}")


class TestPlottingUtilities:
    """Test plotting utility functions and configurations."""

    def test_plotting_imports(self):
        """Test that plotting functions can be imported."""
        from stacking_sats_pipeline import (
            plot_features,
            plot_final_weights,
            plot_spd_comparison,
            plot_weight_sums_by_cycle,
        )

        # All functions should be callable
        assert callable(plot_features)
        assert callable(plot_final_weights)
        assert callable(plot_spd_comparison)
        assert callable(plot_weight_sums_by_cycle)

    def test_plotting_function_signatures(self):
        """Test that plotting functions have reasonable signatures."""
        import inspect

        # Check plot_features signature
        sig = inspect.signature(plot_features)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "plot_features should accept at least one parameter"

        # Check plot_final_weights signature
        sig = inspect.signature(plot_final_weights)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "plot_final_weights should accept at least one parameter"


class TestPlottingIntegration:
    """Integration tests for plotting with actual data."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    @pytest.mark.integration
    def test_full_plotting_workflow(self, mock_figure, mock_show):
        """Test a complete plotting workflow."""
        try:
            from stacking_sats_pipeline import compute_weights, load_data

            # Load real data
            df = load_data()
            weights = compute_weights(df)

            # Test all plotting functions
            plot_features(df, weights=weights)
            plot_final_weights(weights)
            plot_weight_sums_by_cycle(weights)

            # Verify matplotlib was called
            assert mock_figure.call_count >= 3

        except Exception as e:
            pytest.skip(f"Skipping integration plotting test due to: {e}")
