#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline strategy functionality
"""

import numpy as np
import pandas as pd

from stacking_sats_pipeline import compute_weights, construct_features


class TestStrategyFunctions:
    """Test strategy computation functions."""

    def create_test_data(self, num_days=100):
        """Create test Bitcoin price data."""
        dates = pd.date_range("2020-01-01", periods=num_days, freq="D")
        np.random.seed(42)  # For reproducibility
        prices = 30000 + np.cumsum(np.random.normal(0, 500, num_days))
        prices = np.maximum(prices, 1000)  # Ensure positive prices
        return pd.DataFrame({"PriceUSD": prices}, index=dates)

    def test_construct_features_basic(self):
        """Test construct_features function with basic data."""
        df = self.create_test_data(200)  # Need enough data for moving averages

        result = construct_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert isinstance(result.index, pd.DatetimeIndex)

        # Check that some expected features exist (this depends on implementation)
        # The exact features will depend on what construct_features actually does
        assert not result.empty

    def test_construct_features_short_data(self):
        """Test construct_features with minimal data."""
        df = self.create_test_data(10)

        # Should handle short data gracefully
        result = construct_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_compute_weights_basic(self):
        """Test compute_weights function returns valid weights."""
        df = self.create_test_data(200)

        weights = compute_weights(df)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(df)
        assert isinstance(weights.index, pd.DatetimeIndex)

        # Check weight constraints
        assert (weights >= 0).all(), "All weights should be non-negative"
        assert weights.sum() > 0, "Total weights should be positive"

        # Check for reasonable weight values
        assert weights.max() <= 1.0, "Individual weights should not exceed 1.0"

    def test_compute_weights_edge_cases(self):
        """Test compute_weights with edge cases."""
        # Very short data
        df_short = self.create_test_data(5)
        weights_short = compute_weights(df_short)
        assert isinstance(weights_short, pd.Series)
        assert len(weights_short) == len(df_short)

        # Data with extreme values
        df_extreme = self.create_test_data(50)
        df_extreme.loc[df_extreme.index[25], "PriceUSD"] = 1000000  # Very high price
        df_extreme.loc[df_extreme.index[30], "PriceUSD"] = 100  # Very low price

        weights_extreme = compute_weights(df_extreme)
        assert isinstance(weights_extreme, pd.Series)
        assert (weights_extreme >= 0).all()

    def test_compute_weights_missing_data(self):
        """Test compute_weights handles missing data appropriately."""
        df = self.create_test_data(100)

        # Introduce some NaN values
        df.loc[df.index[50], "PriceUSD"] = np.nan
        df.loc[df.index[75], "PriceUSD"] = np.nan

        try:
            weights = compute_weights(df)
            # If it succeeds, check that weights are still valid
            assert isinstance(weights, pd.Series)
            assert len(weights) == len(df)
        except (ValueError, TypeError):
            # It's acceptable for the function to fail with bad data
            pass


class TestStrategyValidation:
    """Test strategy validation logic."""

    def create_test_data(self, num_days=100):
        """Create test Bitcoin price data."""
        dates = pd.date_range("2020-01-01", periods=num_days, freq="D")
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.normal(0, 500, num_days))
        prices = np.maximum(prices, 1000)
        return pd.DataFrame({"PriceUSD": prices}, index=dates)

    def test_strategy_function_signature(self):
        """Test that compute_weights has the expected signature."""
        import inspect

        sig = inspect.signature(compute_weights)
        params = list(sig.parameters.keys())

        # Should accept a DataFrame as first parameter
        assert len(params) >= 1, "compute_weights should accept at least one parameter"
        assert "df" in params or "data" in params or len(params) >= 1

    def test_weights_mathematical_properties(self):
        """Test mathematical properties of generated weights."""
        df = self.create_test_data(100)
        weights = compute_weights(df)

        # Test non-negativity
        assert (weights >= 0).all(), "Weights should be non-negative"

        # Test finiteness
        assert weights.isna().sum() == 0, "Weights should not contain NaN"
        assert np.isfinite(weights).all(), "Weights should be finite"

        # Test reasonable magnitude
        assert weights.max() <= 10.0, "Individual weights should be reasonable"

        # Test that weights vary (not all identical)
        if len(weights) > 10:
            assert weights.std() > 0, "Weights should show some variation"


class TestCustomStrategy:
    """Test custom strategy patterns."""

    def test_uniform_strategy(self):
        """Test a simple uniform weighting strategy."""

        def uniform_strategy(df):
            """Equal weight for all periods."""
            return pd.Series(1.0 / len(df), index=df.index)

        # Create test data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [30000] * 50
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        weights = uniform_strategy(df)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(df)
        assert np.allclose(weights, 1.0 / len(df))
        assert np.isclose(weights.sum(), 1.0)

    def test_momentum_strategy(self):
        """Test a simple momentum-based strategy."""

        def momentum_strategy(df):
            """Buy more when price is increasing."""
            price_change = df["PriceUSD"].pct_change().fillna(0)

            # Base weight
            base_weight = 1.0 / len(df)
            weights = pd.Series(base_weight, index=df.index)

            # Increase weight when price is going up
            weights[price_change > 0] *= 1.2

            # Normalize to sum to 1
            return weights / weights.sum()

        # Create test data with some trend
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        prices = [30000 + i * 1000 for i in range(20)]  # Upward trend
        df = pd.DataFrame({"PriceUSD": prices}, index=dates)

        weights = momentum_strategy(df)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(df)
        assert np.isclose(weights.sum(), 1.0, rtol=1e-10)
        assert (weights >= 0).all()
