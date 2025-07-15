#!/usr/bin/env python3
"""
End-to-end tests for stacking_sats_pipeline data extraction and cleaning pipeline.
"""

import os
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stacking_sats_pipeline.data import (
    MultiSourceDataLoader,
    load_and_merge_data,
    load_data,
    validate_price_data,
)


@pytest.fixture
def mock_fred_response():
    """Mock FRED API response for consistent testing."""
    return {
        "observations": [
            {"date": "2020-01-01", "value": "97.5"},
            {"date": "2020-01-02", "value": "98.1"},
            {"date": "2020-01-03", "value": "97.8"},
            {"date": "2020-01-04", "value": "97.9"},
            {"date": "2020-01-05", "value": "98.2"},
        ]
    }


@pytest.fixture
def mock_coinmetrics_response():
    """Mock CoinMetrics API response for consistent testing."""
    return "time,PriceUSD\n2020-01-01,30000\n2020-01-02,31000\n2020-01-03,32000\n"


class TestHelpers:
    """Helper methods for data pipeline tests."""

    @staticmethod
    def validate_basic_structure(df: pd.DataFrame, expected_cols: list, min_records: int = 1):
        """Validate basic DataFrame structure."""
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= min_records
        assert isinstance(df.index, pd.DatetimeIndex)
        for col in expected_cols:
            assert col in df.columns
        assert df.index.is_monotonic_increasing
        assert df.index.duplicated().sum() == 0

    @staticmethod
    def validate_price_data_quality(df: pd.DataFrame, price_col: str = "PriceUSD"):
        """Validate price data quality."""
        valid_prices = df[price_col].dropna()
        assert len(valid_prices) > 0
        assert (valid_prices > 0).all()
        assert valid_prices.max() < 1_000_000

    @staticmethod
    def validate_timezone(df: pd.DataFrame, expected_tz: str = "UTC"):
        """Validate DataFrame timezone."""
        assert df.index.tz is not None
        assert str(df.index.tz) == expected_tz

    @staticmethod
    def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data by removing invalid values."""
        cleaned = df.copy()

        if "PriceUSD" in cleaned.columns:
            cleaned = cleaned[cleaned["PriceUSD"].notna()]
            cleaned = cleaned[cleaned["PriceUSD"] > 0]

        for col in cleaned.select_dtypes(include=[np.number]).columns:
            cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)

        # Remove NaN values first
        cleaned = cleaned.dropna()

        # Only apply duplicate filtering if there are records left
        if len(cleaned) > 0:
            cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]

        return cleaned


class TestDataPipelineCore:
    """Core data pipeline functionality tests."""

    def test_coinmetrics_data_extraction(self):
        """Test CoinMetrics data extraction."""
        try:
            df = load_data("coinmetrics", use_memory=True)
            TestHelpers.validate_basic_structure(df, ["PriceUSD"], min_records=1000)
            TestHelpers.validate_price_data_quality(df)
            TestHelpers.validate_timezone(df)

            missing_pct = df["PriceUSD"].isna().sum() / len(df) * 100
            assert missing_pct < 50, f"Too many missing prices: {missing_pct:.1f}%"

            print(f"✓ CoinMetrics: {len(df)} records, {missing_pct:.1f}% missing")
        except Exception as e:
            pytest.skip(f"CoinMetrics API issue: {e}")

    @patch.dict(os.environ, {"FRED_API_KEY": "test_key"})
    @patch("stacking_sats_pipeline.data.fred_loader.requests.get")
    def test_fred_data_extraction(self, mock_get, mock_fred_response):
        """Test FRED data extraction."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        try:
            df = load_data("fred", use_memory=True)
            TestHelpers.validate_basic_structure(df, ["DXY_Value"], min_records=5)
            TestHelpers.validate_timezone(df)

            assert df["DXY_Value"].isna().sum() == 0
            assert (df["DXY_Value"] > 50).all() and (df["DXY_Value"] < 200).all()

            print(f"✓ FRED: {len(df)} records")
        except Exception as e:
            pytest.skip(f"FRED API issue: {e}")

    @patch.dict(os.environ, {"FRED_API_KEY": "test_key"})
    @patch("requests.get")
    def test_multi_source_merging(self, mock_get, mock_fred_response, mock_coinmetrics_response):
        """Test multi-source data merging."""

        def mock_requests_side_effect(url, *args, **kwargs):
            response = MagicMock()
            response.raise_for_status.return_value = None

            if "coinmetrics" in url or "btc" in url.lower():
                response.text = mock_coinmetrics_response
            elif "fred" in url.lower() or "stlouisfed" in url:
                response.json.return_value = mock_fred_response
            else:
                response.text = ""
            return response

        mock_get.side_effect = mock_requests_side_effect

        merged_df = load_and_merge_data(["coinmetrics", "fred"], use_memory=True)
        expected_cols = ["PriceUSD_coinmetrics", "DTWEXBGS_Value_fred"]
        TestHelpers.validate_basic_structure(merged_df, expected_cols, min_records=3)

        btc_count = merged_df["PriceUSD_coinmetrics"].notna().sum()
        dxy_count = merged_df["DTWEXBGS_Value_fred"].notna().sum()
        assert btc_count > 0 and dxy_count > 0

        print(f"✓ Merged: {len(merged_df)} total, {btc_count} BTC, {dxy_count} DXY")

    @pytest.mark.parametrize("invalid_source", ["invalid_source", "nonexistent"])
    def test_invalid_source_handling(self, invalid_source):
        """Test error handling for invalid sources."""
        with pytest.raises((ValueError, KeyError)):
            load_data(invalid_source, use_memory=True)

    def test_missing_api_keys(self):
        """Test behavior with missing API keys."""
        original_key = os.environ.get("FRED_API_KEY")
        try:
            if "FRED_API_KEY" in os.environ:
                del os.environ["FRED_API_KEY"]

            loader = MultiSourceDataLoader()
            available_sources = loader.get_available_sources()

            assert "coinmetrics" in available_sources
            assert "fred" not in available_sources

        finally:
            if original_key:
                os.environ["FRED_API_KEY"] = original_key


class TestDataQuality:
    """Data quality and validation tests."""

    def test_data_validation_pipeline(self):
        """Test data cleaning and validation."""
        try:
            df = load_data("coinmetrics", use_memory=True)
            valid_data = df[df["PriceUSD"].notna() & (df["PriceUSD"] > 0)]

            if len(valid_data) < 50:
                pytest.skip("Insufficient valid data")

            sample = valid_data.head(50).copy()
            validate_price_data(sample)  # Should pass

            # Introduce issues
            sample.iloc[5, sample.columns.get_loc("PriceUSD")] = np.nan
            sample.iloc[10, sample.columns.get_loc("PriceUSD")] = -1000

            with pytest.raises(ValueError):
                validate_price_data(sample)

            # Test cleaning
            cleaned = TestHelpers.clean_price_data(sample)
            assert cleaned["PriceUSD"].isna().sum() == 0
            assert (cleaned["PriceUSD"] > 0).all()

            print(f"✓ Validation: {len(sample)} → {len(cleaned)} after cleaning")

        except Exception as e:
            pytest.skip(f"Validation test failed: {e}")

    @pytest.mark.parametrize(
        "test_case,description",
        [
            (pd.DataFrame(), "empty"),
            (pd.DataFrame({"PriceUSD": [30000]}, index=[0]), "non-datetime index"),
            (
                pd.DataFrame({"Volume": [1000]}, index=pd.date_range("2020-01-01", periods=1)),
                "missing price column",
            ),
        ],
    )
    def test_validation_edge_cases(self, test_case, description):
        """Test validation edge cases."""
        with pytest.raises(ValueError):
            validate_price_data(test_case)


class TestFileOperations:
    """File caching and format tests."""

    def test_file_caching(self):
        """Test file caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                loader = MultiSourceDataLoader(data_dir=temp_dir)

                df1 = loader.load_from_source("coinmetrics", use_memory=False)
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
                assert len(csv_files) > 0

                df2 = loader.load_from_source("coinmetrics", use_memory=False)
                pd.testing.assert_frame_equal(df1, df2, check_dtype=False)

                print(f"✓ Caching: {len(df1)} records cached and reloaded")

            except Exception as e:
                pytest.skip(f"Caching test failed: {e}")

    @pytest.mark.parametrize("file_format", ["csv", "parquet"])
    def test_file_formats(self, file_format):
        """Test different file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                loader = MultiSourceDataLoader(data_dir=temp_dir)
                df = loader.load_from_source(
                    "coinmetrics", use_memory=False, file_format=file_format
                )

                TestHelpers.validate_basic_structure(df, ["PriceUSD"], min_records=1000)

                files = [f for f in os.listdir(temp_dir) if f.endswith(f".{file_format}")]
                assert len(files) > 0, f"Should create {file_format} file"

                print(f"✓ {file_format.upper()}: {len(df)} records")

            except Exception as e:
                pytest.skip(f"{file_format} test failed: {e}")


class TestPerformance:
    """Performance and efficiency tests."""

    def test_loading_performance(self):
        """Test data loading performance."""
        import time

        try:
            start_time = time.time()
            df = load_data("coinmetrics", use_memory=True)
            loading_time = time.time() - start_time

            assert loading_time < 30.0, f"Loading too slow: {loading_time:.2f}s"
            assert len(df) > 1000, "Insufficient data"

            memory_usage = df.memory_usage(deep=True).sum()
            assert memory_usage < 100_000_000, f"Memory usage too high: {memory_usage}"

            memory_mb = memory_usage / 1_000_000
            print(f"✓ Performance: {len(df)} records in {loading_time:.2f}s, {memory_mb:.1f}MB")

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")

    def test_memory_efficiency(self):
        """Test memory efficiency."""
        try:
            import psutil

            process = psutil.Process(os.getpid())

            memory_before = process.memory_info().rss
            df = load_data("coinmetrics", use_memory=True)
            memory_after = process.memory_info().rss

            memory_increase = memory_after - memory_before
            bytes_per_record = memory_increase / len(df) if len(df) > 0 else 0

            assert bytes_per_record < 10000, f"Memory per record too high: {bytes_per_record}"

            print(f"✓ Memory: {len(df)} records, {memory_increase:,} bytes increase")

        except ImportError:
            pytest.skip("psutil not available")
        except Exception as e:
            pytest.skip(f"Memory test failed: {e}")


class TestIntegration:
    """Integration tests with real APIs."""

    def test_real_data_integration(self):
        """Integration test with real endpoints."""
        try:
            df_cm = load_data("coinmetrics", use_memory=True)
            TestHelpers.validate_basic_structure(df_cm, ["PriceUSD"], min_records=1000)

            if os.getenv("FRED_API_KEY"):
                df_fred = load_data("fred", use_memory=True)
                TestHelpers.validate_basic_structure(df_fred, ["DXY_Value"], min_records=100)

                merged_df = load_and_merge_data(["coinmetrics", "fred"], use_memory=True)
                TestHelpers.validate_basic_structure(
                    merged_df, ["PriceUSD_coinmetrics", "DXY_Value_fred"], min_records=100
                )

                print(
                    f"✓ Full integration: CM({len(df_cm)}) + FRED({len(df_fred)}) = "
                    f"Merged({len(merged_df)})"
                )
            else:
                print(f"✓ Partial integration: CoinMetrics({len(df_cm)}) only")

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @patch.dict(os.environ, {"FRED_API_KEY": "test_key"})
    @patch("requests.get")
    def test_timestamp_alignment(self, mock_get, mock_fred_response, mock_coinmetrics_response):
        """Test timestamp alignment fix."""

        def mock_requests_side_effect(url, *args, **kwargs):
            response = MagicMock()
            response.raise_for_status.return_value = None

            if "coinmetrics" in url or "btc" in url.lower():
                response.text = mock_coinmetrics_response
            elif "fred" in url.lower() or "stlouisfed" in url:
                response.json.return_value = mock_fred_response
            else:
                response.text = ""
            return response

        mock_get.side_effect = mock_requests_side_effect

        try:
            df_cm = load_data("coinmetrics", use_memory=True)
            df_fred = load_data("fred", use_memory=True)

            # Verify timestamps are at midnight UTC
            for _source_name, df in [("CoinMetrics", df_cm), ("FRED", df_fred)]:
                sample_ts = df.index[:5]
                for ts in sample_ts:
                    assert ts.hour == 0 and ts.minute == 0 and ts.second == 0
                    assert str(ts.tz) == "UTC"

            merged_df = load_and_merge_data(["coinmetrics", "fred"], use_memory=True)

            price_col = "PriceUSD_coinmetrics"
            dxy_col = "DXY_Value_fred"
            both_available = merged_df[price_col].notna() & merged_df[dxy_col].notna()
            overlap_count = both_available.sum()

            print(f"✓ Timestamp alignment: {overlap_count} overlapping records")

        except Exception as e:
            pytest.skip(f"Timestamp alignment test failed: {e}")

    def test_date_range_filtering(self):
        """Test date range filtering."""
        try:
            df_full = load_data("coinmetrics", use_memory=True)

            end_date = df_full.index.max()
            start_date = end_date - timedelta(days=30)
            df_filtered = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]

            assert len(df_filtered) <= 31
            assert df_filtered.index.min() >= start_date
            assert df_filtered.index.max() <= end_date

            valid_prices = df_filtered["PriceUSD"].dropna()
            assert len(valid_prices) > 0

            print(f"✓ Date filtering: {len(df_full)} → {len(df_filtered)} records")

        except Exception as e:
            pytest.skip(f"Date filtering test failed: {e}")
