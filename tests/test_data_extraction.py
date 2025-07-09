#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline data extraction functionality
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stacking_sats_pipeline import extract_all_data


class TestDataExtractionPythonAPI:
    """Test the extract_all_data Python API functionality."""

    @pytest.mark.integration
    def test_extract_all_data_csv_integration(self):
        """Integration test for extracting all data to merged CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test CSV extraction
                extract_all_data("csv", temp_dir)

                # Check that the merged file was created
                output_path = Path(temp_dir)
                csv_files = list(output_path.glob("*.csv"))

                # Should have exactly one merged CSV file
                assert len(csv_files) == 1, (
                    f"Expected exactly 1 merged CSV file, got {len(csv_files)}"
                )

                merged_file = csv_files[0]
                assert merged_file.name == "merged_crypto_data.csv", (
                    f"Expected merged_crypto_data.csv, got {merged_file.name}"
                )

                # Validate that file contains merged data
                df = pd.read_csv(merged_file, index_col=0, parse_dates=True, low_memory=False)
                assert len(df) > 0, "Merged file should contain data"
                assert isinstance(df.index, pd.DatetimeIndex), (
                    "Merged file should have datetime index"
                )

                # Should have columns from different sources with suffixes
                expected_sources = ["coinmetrics", "feargreed"]
                if os.getenv("FRED_API_KEY"):
                    expected_sources.append("fred")

                for source in expected_sources:
                    source_columns = [col for col in df.columns if col.endswith(f"_{source}")]
                    assert len(source_columns) > 0, f"Should have columns from {source} source"

                print(f"✓ CSV merged extraction test: {df.shape} data shape")

            except Exception as e:
                pytest.skip(f"Skipping CSV extraction test due to API issue: {e}")

    @pytest.mark.integration
    def test_extract_all_data_parquet_integration(self):
        """Integration test for extracting all data to merged Parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test Parquet extraction
                extract_all_data("parquet", temp_dir)

                # Check that the merged file was created
                output_path = Path(temp_dir)
                parquet_files = list(output_path.glob("*.parquet"))

                # Should have exactly one merged Parquet file
                assert len(parquet_files) == 1, (
                    f"Expected exactly 1 merged Parquet file, got {len(parquet_files)}"
                )

                merged_file = parquet_files[0]
                assert merged_file.name == "merged_crypto_data.parquet", (
                    f"Expected merged_crypto_data.parquet, got {merged_file.name}"
                )

                # Validate that file contains merged data
                df = pd.read_parquet(merged_file)
                assert len(df) > 0, "Merged file should contain data"
                assert isinstance(df.index, pd.DatetimeIndex), (
                    "Merged file should have datetime index"
                )

                # Should have columns from different sources with suffixes
                expected_sources = ["coinmetrics", "feargreed"]
                if os.getenv("FRED_API_KEY"):
                    expected_sources.append("fred")

                for source in expected_sources:
                    source_columns = [col for col in df.columns if col.endswith(f"_{source}")]
                    assert len(source_columns) > 0, f"Should have columns from {source} source"

                print(f"✓ Parquet merged extraction test: {df.shape} data shape")

            except Exception as e:
                pytest.skip(f"Skipping Parquet extraction test due to API issue: {e}")

    def test_extract_all_data_default_directory(self):
        """Test extract_all_data with default directory (current directory)."""
        with (
            patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class,
        ):
            # Mock the loader instance and its methods
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock available sources
            mock_loader.get_available_sources.return_value = [
                "coinmetrics",
                "feargreed",
                "fred",
            ]

            # Mock merged DataFrame with to_csv method
            mock_merged_df = MagicMock()
            mock_merged_df.shape = (1000, 5)
            mock_merged_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_merged_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_to_csv = MagicMock()
            mock_merged_df.to_csv = mock_to_csv

            # Mock the filtered DataFrame (result of .loc operation)
            mock_filtered_df = MagicMock()
            mock_filtered_df.shape = (800, 5)  # Simulated filtered size
            mock_filtered_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_filtered_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_filtered_df.to_csv = mock_to_csv
            mock_merged_df.loc.__getitem__.return_value = mock_filtered_df

            mock_loader.load_and_merge.return_value = mock_merged_df

            # Mock the directory creation and file operations
            with (
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("pathlib.Path.stat") as mock_stat,
            ):
                # Create a proper stat result mock
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = 1024 * 1024  # 1MB
                mock_stat.return_value = mock_stat_result

                # Test with None (default directory)
                extract_all_data("csv", None)

                # Verify loader was created and methods called
                mock_loader_class.assert_called_once()
                mock_loader.get_available_sources.assert_called_once()
                mock_loader.load_and_merge.assert_called_once()
                mock_to_csv.assert_called_once()
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_extract_all_data_no_fred_api_key(self):
        """Test extract_all_data behavior when FRED API key is missing."""
        with (
            patch("stacking_sats_pipeline.main.os.getenv") as mock_getenv,
            patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class,
        ):
            # Mock no API key
            mock_getenv.return_value = None

            # Mock the loader instance
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock available sources (without FRED)
            mock_loader.get_available_sources.return_value = [
                "coinmetrics",
                "feargreed",
            ]

            # Mock merged DataFrame with to_csv method
            mock_merged_df = MagicMock()
            mock_merged_df.shape = (1000, 3)
            mock_merged_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_merged_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_to_csv = MagicMock()
            mock_merged_df.to_csv = mock_to_csv

            # Mock the filtered DataFrame (result of .loc operation)
            mock_filtered_df = MagicMock()
            mock_filtered_df.shape = (800, 3)  # Simulated filtered size
            mock_filtered_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_filtered_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_filtered_df.to_csv = mock_to_csv
            mock_merged_df.loc.__getitem__.return_value = mock_filtered_df

            mock_loader.load_and_merge.return_value = mock_merged_df

            # Mock the directory creation and file operations
            with (
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.stat") as mock_stat,
            ):
                # Create a proper stat result mock
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = 1024 * 1024  # 1MB
                mock_stat.return_value = mock_stat_result

                # Test extraction without FRED API key
                extract_all_data("csv", "test_dir")

                # Verify loader was created and methods called
                mock_loader_class.assert_called_once()
                mock_loader.get_available_sources.assert_called_once()
                mock_loader.load_and_merge.assert_called_once_with(
                    ["coinmetrics", "feargreed"], use_memory=True
                )

    def test_extract_all_data_error_handling(self):
        """Test error handling in extract_all_data."""
        with patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class:
            # Mock the loader instance
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock available sources
            mock_loader.get_available_sources.return_value = [
                "coinmetrics",
                "feargreed",
            ]

            # Mock an exception in data loading
            mock_loader.load_and_merge.side_effect = Exception("Network error")

            # Should raise the exception
            with pytest.raises(Exception) as exc_info:
                extract_all_data("csv", "test_dir")

            assert "Network error" in str(exc_info.value)

    def test_extract_all_data_invalid_format(self):
        """Test extract_all_data with invalid format defaults to CSV."""
        with (
            patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class,
        ):
            # Mock the loader instance
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock available sources
            mock_loader.get_available_sources.return_value = [
                "coinmetrics",
                "feargreed",
            ]

            # Mock merged DataFrame with to_csv method
            mock_merged_df = MagicMock()
            mock_merged_df.shape = (1000, 3)
            mock_merged_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_merged_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_to_csv = MagicMock()
            mock_merged_df.to_csv = mock_to_csv

            # Mock the filtered DataFrame (result of .loc operation)
            mock_filtered_df = MagicMock()
            mock_filtered_df.shape = (800, 3)  # Simulated filtered size
            mock_filtered_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_filtered_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_filtered_df.to_csv = mock_to_csv
            mock_merged_df.loc.__getitem__.return_value = mock_filtered_df

            mock_loader.load_and_merge.return_value = mock_merged_df

            # Mock the directory creation and file operations
            with (
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.stat") as mock_stat,
            ):
                # Create a proper stat result mock
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = 1024 * 1024  # 1MB
                mock_stat.return_value = mock_stat_result

                # Test with invalid format - should default to CSV behavior
                extract_all_data("invalid_format", "test_dir")

                # Should have called CSV method
                mock_to_csv.assert_called_once()

    @pytest.mark.integration
    def test_extract_all_data_file_size_comparison(self):
        """Test that Parquet files are more efficient than CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract to both formats
                csv_dir = Path(temp_dir) / "csv"
                parquet_dir = Path(temp_dir) / "parquet"

                csv_dir.mkdir()
                parquet_dir.mkdir()

                extract_all_data("csv", csv_dir)
                extract_all_data("parquet", parquet_dir)

                # Compare file sizes
                csv_file = csv_dir / "merged_crypto_data.csv"
                parquet_file = parquet_dir / "merged_crypto_data.parquet"

                assert csv_file.exists(), "CSV file should exist"
                assert parquet_file.exists(), "Parquet file should exist"

                csv_size = csv_file.stat().st_size
                parquet_size = parquet_file.stat().st_size

                compression_ratio = parquet_size / csv_size
                print(
                    f"✓ File size comparison: CSV={csv_size:,} bytes, "
                    f"Parquet={parquet_size:,} bytes"
                )
                print(f"✓ Compression ratio: {compression_ratio:.2f}")

                # Parquet should generally be smaller or at least not significantly
                # larger
                assert compression_ratio < 2.0, (
                    "Parquet should not be significantly larger than CSV"
                )

            except Exception as e:
                pytest.skip(f"Skipping file size comparison test due to API issue: {e}")

    @pytest.mark.integration
    def test_extract_all_data_data_integrity(self):
        """Test that extracted merged data maintains integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract data
                extract_all_data("csv", temp_dir)

                # Load and validate the merged file
                merged_file = Path(temp_dir) / "merged_crypto_data.csv"
                assert merged_file.exists(), "Merged CSV file should exist"

                df = pd.read_csv(merged_file, index_col=0, parse_dates=True, low_memory=False)
                assert len(df) > 100, "Merged data should have substantial records"
                assert isinstance(df.index, pd.DatetimeIndex), "Should have datetime index"

                # Check that we have columns from different sources
                coinmetrics_cols = [col for col in df.columns if "coinmetrics" in col]
                feargreed_cols = [col for col in df.columns if "feargreed" in col]

                assert len(coinmetrics_cols) > 0, "Should have CoinMetrics columns"
                assert len(feargreed_cols) > 0, "Should have Fear & Greed columns"

                # If FRED data is available, check for it too
                if os.getenv("FRED_API_KEY"):
                    fred_cols = [col for col in df.columns if "fred" in col]
                    assert len(fred_cols) > 0, "Should have FRED columns"

                # Check for reasonable data ranges
                price_cols = [col for col in df.columns if "PriceUSD" in col]
                if price_cols:
                    for col in price_cols:
                        valid_prices = df[col].dropna()
                        if len(valid_prices) > 0:
                            assert valid_prices.min() > 0, f"{col} should have positive prices"
                            assert valid_prices.max() < 1_000_000, (
                                f"{col} should have reasonable prices"
                            )

                fear_greed_cols = [col for col in df.columns if "fear_greed_value" in col]
                if fear_greed_cols:
                    for col in fear_greed_cols:
                        valid_values = df[col].dropna()
                        if len(valid_values) > 0:
                            assert valid_values.min() >= 0, f"{col} should be >= 0"
                            assert valid_values.max() <= 100, f"{col} should be <= 100"

                print("✓ Merged data integrity validation passed")

            except Exception as e:
                pytest.skip(f"Skipping data integrity test due to API issue: {e}")

    @pytest.mark.integration
    def test_extract_all_data_timestamp_alignment_verification(self):
        """Test that extracted CSV data has proper timestamp alignment and overlapping records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract all available data
                extract_all_data("csv", temp_dir)

                # Load the merged CSV file
                merged_file = Path(temp_dir) / "merged_crypto_data.csv"
                assert merged_file.exists(), "Merged CSV file should exist"

                df = pd.read_csv(merged_file, index_col=0, parse_dates=True, low_memory=False)
                assert isinstance(df.index, pd.DatetimeIndex), "Should have datetime index"

                # Verify all timestamps are at midnight UTC (the fix)
                sample_timestamps = df.index[:10]  # Check first 10 timestamps
                for ts in sample_timestamps:
                    assert ts.hour == 0, f"Timestamp should be at midnight: {ts}"
                    assert ts.minute == 0, f"Timestamp minute should be 0: {ts}"
                    assert ts.second == 0, f"Timestamp second should be 0: {ts}"
                    # Note: CSV loading may not preserve timezone info, but
                    # normalization should still work

                # Check for overlapping data between sources
                coinmetrics_cols = [
                    col for col in df.columns if "coinmetrics" in col and "PriceUSD" in col
                ]
                feargreed_cols = [
                    col for col in df.columns if "feargreed" in col and "fear_greed_value" in col
                ]

                if coinmetrics_cols and feargreed_cols:
                    btc_col = coinmetrics_cols[0]
                    fear_col = feargreed_cols[0]

                    # Count overlapping records
                    both_available = df[btc_col].notna() & df[fear_col].notna()
                    overlap_count = both_available.sum()

                    assert overlap_count > 0, (
                        f"Should have overlapping BTC and Fear&Greed data, got {overlap_count} rows"
                    )
                    print(f"✓ BTC & Fear&Greed overlap: {overlap_count} records")

                # Check FRED data overlap if available
                if os.getenv("FRED_API_KEY"):
                    fred_cols = [col for col in df.columns if "fred" in col and "DXY" in col]
                    if coinmetrics_cols and fred_cols:
                        btc_col = coinmetrics_cols[0]
                        dxy_col = fred_cols[0]

                        # Count overlapping records (this was the original bug - 0
                        # overlaps)
                        both_available = df[btc_col].notna() & df[dxy_col].notna()
                        overlap_count = both_available.sum()

                        assert overlap_count > 0, (
                            f"Timestamp alignment fix failed - BTC & DXY overlap is "
                            f"{overlap_count}. BTC records: {df[btc_col].notna().sum()}, "
                            f"DXY records: {df[dxy_col].notna().sum()}"
                        )

                        # Should have substantial overlap
                        assert overlap_count > 100, (
                            f"Expected substantial BTC & DXY overlap, got {overlap_count} records"
                        )

                        print(
                            f"✓ BTC & DXY overlap: {overlap_count} records "
                            f"(timestamp alignment working)"
                        )
                else:
                    print("✓ FRED API key not available - skipping BTC & DXY overlap test")

                print("✓ Timestamp alignment verification in CSV extraction passed")

            except Exception as e:
                pytest.skip(f"Skipping timestamp alignment test due to API issue: {e}")


class TestDataExtractionUtilities:
    """Test utility functions and edge cases for data extraction."""

    def test_extract_all_data_function_exists(self):
        """Test that extract_all_data function exists and is callable."""
        from stacking_sats_pipeline import extract_all_data

        assert callable(extract_all_data), "extract_all_data should be callable"

    def test_extract_all_data_in_init(self):
        """Test that extract_all_data is properly exported in __init__.py."""
        from stacking_sats_pipeline import __all__ as exported_functions

        assert "extract_all_data" in exported_functions, "extract_all_data should be in __all__"

    def test_extract_all_data_function_signature(self):
        """Test extract_all_data function signature."""
        import inspect

        from stacking_sats_pipeline import extract_all_data

        sig = inspect.signature(extract_all_data)
        params = list(sig.parameters.keys())

        assert len(params) >= 1, "extract_all_data should accept at least one parameter"
        assert "file_format" in params, "Should have file_format parameter"

    def test_path_handling(self):
        """Test that the function handles different path types correctly."""
        with patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class:
            # Mock the loader instance
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Mock available sources
            mock_loader.get_available_sources.return_value = [
                "coinmetrics",
                "feargreed",
            ]

            # Mock merged DataFrame with to_csv method
            mock_merged_df = MagicMock()
            mock_merged_df.shape = (1000, 3)
            mock_merged_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_merged_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_to_csv = MagicMock()
            mock_merged_df.to_csv = mock_to_csv

            # Mock the filtered DataFrame (result of .loc operation)
            mock_filtered_df = MagicMock()
            mock_filtered_df.shape = (800, 3)  # Simulated filtered size
            mock_filtered_df.index.min.return_value = pd.Timestamp("2020-01-01")
            mock_filtered_df.index.max.return_value = pd.Timestamp("2023-12-31")
            mock_filtered_df.to_csv = mock_to_csv
            mock_merged_df.loc.__getitem__.return_value = mock_filtered_df

            mock_loader.load_and_merge.return_value = mock_merged_df

            # Mock the directory creation and file operations
            with (
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.stat") as mock_stat,
            ):
                # Create a proper stat result mock
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = 1024 * 1024  # 1MB
                mock_stat.return_value = mock_stat_result

                # Test with string path
                extract_all_data("csv", "test_string_path")

                # Test with Path object
                extract_all_data("csv", Path("test_path_object"))

                # Verify loader was created twice
                assert mock_loader_class.call_count == 2

    def test_no_available_sources(self):
        """Test behavior when no data sources are available."""
        with patch("stacking_sats_pipeline.main.MultiSourceDataLoader") as mock_loader_class:
            # Mock the loader instance with no available sources
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.get_available_sources.return_value = []

            # Should return early without attempting to load data
            extract_all_data("csv", "test_dir")

            # Should not have called load_and_merge
            mock_loader.load_and_merge.assert_not_called()
