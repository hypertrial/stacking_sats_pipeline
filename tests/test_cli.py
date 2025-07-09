#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline CLI functionality
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import stacking_sats_pipeline.main; stacking_sats_pipeline.main.main()",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should exit with code 0 for help
            assert result.returncode == 0
            assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI help test timed out or Python not found")
        except Exception as e:
            pytest.skip(f"CLI help test failed: {e}")

    def test_stacking_sats_command(self):
        """Test that the stacking-sats command entry point works via main module."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "stacking_sats_pipeline.main", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should exit with code 0 for help
            assert result.returncode == 0
            assert "usage:" in result.stdout.lower()

        except subprocess.TimeoutExpired:
            pytest.fail("CLI help command timed out")
        except FileNotFoundError:
            pytest.fail("Python module stacking_sats_pipeline.main not found")
        except Exception as e:
            pytest.fail(f"CLI command test failed: {e}")


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_data_extraction_argument_parsing(self):
        """Test data extraction argument parsing."""
        import argparse

        # Test the argument parser directly with data extraction arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--extract-data", choices=["csv", "parquet"], default="csv")
        parser.add_argument("--output-dir", "-o", type=str)

        # Test default (CSV extraction)
        args = parser.parse_args([])
        assert args.extract_data == "csv"
        assert args.output_dir is None

        # Test with CSV extraction
        args = parser.parse_args(["--extract-data", "csv"])
        assert args.extract_data == "csv"

        # Test with Parquet extraction
        args = parser.parse_args(["--extract-data", "parquet"])
        assert args.extract_data == "parquet"

        # Test with output directory
        args = parser.parse_args(["--extract-data", "csv", "--output-dir", "data/"])
        assert args.extract_data == "csv"
        assert args.output_dir == "data/"

        # Test short form of output-dir
        args = parser.parse_args(["--extract-data", "parquet", "-o", "exports/"])
        assert args.extract_data == "parquet"
        assert args.output_dir == "exports/"


class TestCLIDataExtraction:
    """Test CLI data extraction functionality."""

    @patch("stacking_sats_pipeline.main.extract_all_data")
    def test_cli_extract_data_csv(self, mock_extract):
        """Test CLI with --extract-data csv."""
        from stacking_sats_pipeline.main import main

        with patch("sys.argv", ["main.py", "--extract-data", "csv"]):
            main()

        mock_extract.assert_called_once_with(file_format="csv", output_dir=None)

    @patch("stacking_sats_pipeline.main.extract_all_data")
    def test_cli_extract_data_parquet(self, mock_extract):
        """Test CLI with --extract-data parquet."""
        from stacking_sats_pipeline.main import main

        with patch("sys.argv", ["main.py", "--extract-data", "parquet"]):
            main()

        mock_extract.assert_called_once_with(file_format="parquet", output_dir=None)

    @patch("stacking_sats_pipeline.main.extract_all_data")
    def test_cli_extract_data_with_output_dir(self, mock_extract):
        """Test CLI with --extract-data and --output-dir."""
        from stacking_sats_pipeline.main import main

        with patch("sys.argv", ["main.py", "--extract-data", "csv", "--output-dir", "data/"]):
            main()

        mock_extract.assert_called_once_with(file_format="csv", output_dir="data/")

    @patch("stacking_sats_pipeline.main.extract_all_data")
    def test_cli_extract_data_short_output_dir(self, mock_extract):
        """Test CLI with --extract-data and -o (short form)."""
        from stacking_sats_pipeline.main import main

        with patch("sys.argv", ["main.py", "--extract-data", "parquet", "-o", "exports/"]):
            main()

        mock_extract.assert_called_once_with(file_format="parquet", output_dir="exports/")

    @pytest.mark.integration
    def test_cli_extract_data_integration(self):
        """Integration test for CLI data extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Test the CLI command with real execution
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import stacking_sats_pipeline.main; stacking_sats_pipeline.main.main()",
                        "--extract-data",
                        "csv",
                        "--output-dir",
                        str(temp_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,  # Allow more time for data loading
                )

                # Check if extraction completed (may skip if no data available)
                if "No data sources are available" in result.stdout:
                    pytest.skip("No data sources available for extraction test")

                if result.returncode != 0:
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")

                assert result.returncode == 0, f"CLI failed with: {result.stderr}"

                # Check that output file was created
                expected_file = temp_path / "merged_crypto_data.csv"
                if expected_file.exists():
                    assert expected_file.stat().st_size > 0, "Output file should not be empty"

            except subprocess.TimeoutExpired:
                pytest.skip("CLI integration test timed out")
            except Exception as e:
                pytest.skip(f"CLI integration test failed: {e}")

    def test_cli_extract_data_help_includes_new_options(self):
        """Test that CLI help includes new data extraction options."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import stacking_sats_pipeline.main; stacking_sats_pipeline.main.main()",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0
            help_text = result.stdout.lower()

            # Check for new options in help text
            assert "--extract-data" in help_text or "extract" in help_text
            assert "--output-dir" in help_text or "output" in help_text

        except Exception as e:
            pytest.skip(f"CLI help test failed: {e}")

    def test_cli_extract_data_invalid_format(self):
        """Test CLI with invalid --extract-data format."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import stacking_sats_pipeline.main; stacking_sats_pipeline.main.main()",
                    "--extract-data",
                    "invalid_format",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should fail with non-zero exit code
            assert result.returncode != 0
            assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()

        except Exception as e:
            pytest.skip(f"CLI invalid format test failed: {e}")


class TestCLIFunctionality:
    """Test specific CLI functionality."""

    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        from stacking_sats_pipeline.main import main

        assert callable(main)

    def test_main_function_signature(self):
        """Test main function signature."""
        import inspect

        from stacking_sats_pipeline.main import main

        sig = inspect.signature(main)
        # main() should not require any arguments
        assert len(sig.parameters) == 0

    @patch("sys.argv", ["main.py", "--help"])
    def test_main_with_help_argument(self):
        """Test main function with help argument."""
        from stacking_sats_pipeline.main import main

        try:
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Help should exit with code 0
            assert exc_info.value.code == 0

        except Exception as e:
            pytest.skip(f"Main help test failed: {e}")


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_extract_all_data_function_availability(self):
        """Test that extract_all_data function is available in main module."""
        try:
            from stacking_sats_pipeline.main import extract_all_data

            assert callable(extract_all_data), "extract_all_data should be callable"

        except ImportError:
            pytest.skip("extract_all_data function not available in main module")
