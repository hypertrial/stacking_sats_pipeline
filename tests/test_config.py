#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline configuration and metadata
"""

from datetime import datetime

import pytest

import stacking_sats_pipeline


class TestPackageMetadata:
    """Test package metadata and imports."""

    def test_version_exists(self):
        """Test that package has a version attribute."""
        assert hasattr(stacking_sats_pipeline, "__version__")
        assert isinstance(stacking_sats_pipeline.__version__, str)
        assert len(stacking_sats_pipeline.__version__) > 0

    def test_package_imports(self):
        """Test that all exported data engineering functions can be imported."""
        from stacking_sats_pipeline import (
            extract_all_data,
            load_data,
            validate_price_data,
        )

        # Test that all imports are not None
        assert extract_all_data is not None
        assert load_data is not None
        assert validate_price_data is not None


class TestConfigConstants:
    """Test configuration constants."""

    def test_config_constants_exist(self):
        """Test that all configuration constants exist and have expected types."""
        from stacking_sats_pipeline import (
            BACKTEST_END,
            BACKTEST_START,
        )

        # Test date constants
        assert BACKTEST_START is not None
        assert BACKTEST_END is not None
        assert isinstance(BACKTEST_START, str)
        assert isinstance(BACKTEST_END, str)

    def test_date_format(self):
        """Test that date constants are in valid format."""
        from stacking_sats_pipeline import BACKTEST_END, BACKTEST_START

        # Test that dates can be parsed
        try:
            start_date = datetime.fromisoformat(BACKTEST_START)
            end_date = datetime.fromisoformat(BACKTEST_END)
            assert start_date < end_date
        except ValueError:
            pytest.fail(
                f"Invalid date format in BACKTEST_START ({BACKTEST_START}) or "
                f"BACKTEST_END ({BACKTEST_END})"
            )
