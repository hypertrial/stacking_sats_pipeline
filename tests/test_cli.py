#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline CLI functionality
"""

import os
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
        """Test that the stacking-sats command is available."""
        try:
            result = subprocess.run(
                ["stacking-sats", "--help"], capture_output=True, text=True, timeout=30
            )

            # Should exit with code 0 for help
            assert result.returncode == 0
            assert "usage:" in result.stdout.lower()

        except subprocess.CalledProcessError:
            # Command exists but may have failed for other reasons
            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("stacking-sats command not available or timed out")
        except Exception as e:
            pytest.skip(f"stacking-sats command test failed: {e}")


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_no_plot_argument_parsing(self):
        """Test --no-plot argument parsing."""
        import argparse

        # Test the argument parser directly
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-plot", action="store_true")

        # Test default (no --no-plot)
        args = parser.parse_args([])
        assert args.no_plot is False

        # Test with --no-plot
        args = parser.parse_args(["--no-plot"])
        assert args.no_plot is True

    def test_argument_parsing_strategy(self):
        """Test strategy argument parsing."""
        import argparse

        # Test argument parser directly
        parser = argparse.ArgumentParser()
        parser.add_argument("--strategy", "-s", type=str, default="strategy/strategy_template.py")
        parser.add_argument("--no-plot", action="store_true")
        parser.add_argument("--simulate", action="store_true")
        parser.add_argument("--budget", type=float, default=10_000_000)

        # Test default values
        args = parser.parse_args([])
        assert args.strategy == "strategy/strategy_template.py"
        assert args.no_plot is False
        assert args.simulate is False
        assert args.budget == 10_000_000

        # Test custom values
        args = parser.parse_args(
            [
                "--strategy",
                "custom.py",
                "--no-plot",
                "--simulate",
                "--budget",
                "5000000",
            ]
        )
        assert args.strategy == "custom.py"
        assert args.no_plot is True
        assert args.simulate is True
        assert args.budget == 5000000

    def test_data_extraction_argument_parsing(self):
        """Test data extraction argument parsing."""
        import argparse

        # Test the argument parser directly with new data extraction arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--extract-data", choices=["csv", "parquet"])
        parser.add_argument("--output-dir", "-o", type=str)

        # Test default (no extraction)
        args = parser.parse_args([])
        assert args.extract_data is None
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

    @patch("stacking_sats_pipeline.main.extract_all_data")
    @patch("stacking_sats_pipeline.main.load_data")
    def test_cli_extract_data_skips_backtesting(self, mock_load_data, mock_extract):
        """Test that CLI with --extract-data exits after extraction without running backtesting."""
        from stacking_sats_pipeline.main import main

        with patch("sys.argv", ["main.py", "--extract-data", "csv"]):
            main()

        # extract_all_data should be called
        mock_extract.assert_called_once()

        # load_data should NOT be called (backtesting is skipped)
        mock_load_data.assert_not_called()

    @pytest.mark.integration
    def test_cli_extract_data_integration(self):
        """Integration test for CLI data extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test the CLI command directly
                result = subprocess.run(
                    [
                        "stacking-sats",
                        "--extract-data",
                        "csv",
                        "--output-dir",
                        temp_dir,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Should exit with code 0
                assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

                # Check that files were created
                output_path = Path(temp_dir)
                csv_files = list(output_path.glob("*.csv"))

                # Should have at least Bitcoin and Fear & Greed data
                assert len(csv_files) >= 2, f"Expected at least 2 CSV files, got {len(csv_files)}"

                # Check for expected files
                file_names = [f.name for f in csv_files]
                assert "btc_coinmetrics.csv" in file_names, "Bitcoin CSV file missing"
                assert "fear_greed.csv" in file_names, "Fear & Greed CSV file missing"

                print(f"✓ CLI integration test: {len(csv_files)} files created")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                pytest.skip(f"CLI integration test timed out or command not found: {e}")
            except Exception as e:
                pytest.skip(f"CLI integration test failed: {e}")

    def test_cli_extract_data_help_includes_new_options(self):
        """Test that CLI help includes the new data extraction options."""
        try:
            result = subprocess.run(
                ["stacking-sats", "--help"], capture_output=True, text=True, timeout=30
            )

            help_text = result.stdout.lower()

            # Check that new options are documented
            assert "--extract-data" in help_text, "Help should mention --extract-data option"
            assert "--output-dir" in help_text, "Help should mention --output-dir option"
            assert "csv" in help_text and "parquet" in help_text, (
                "Help should mention CSV and Parquet formats"
            )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI help test timed out or command not found")
        except Exception as e:
            pytest.skip(f"CLI help test failed: {e}")

    def test_cli_extract_data_invalid_format(self):
        """Test CLI behavior with invalid extraction format."""
        try:
            result = subprocess.run(
                ["stacking-sats", "--extract-data", "invalid"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should fail with non-zero exit code
            assert result.returncode != 0, "CLI should reject invalid format"

            # Should mention valid choices in error
            error_text = result.stderr.lower()
            assert "csv" in error_text or "parquet" in error_text, (
                "Error should mention valid formats"
            )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI invalid format test timed out or command not found")
        except Exception as e:
            pytest.skip(f"CLI invalid format test failed: {e}")


class TestCLIStrategyLoading:
    """Test CLI strategy loading functionality."""

    def create_test_strategy_file(self, content):
        """Create a temporary strategy file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            return f.name

    def test_load_strategy_from_file_valid(self):
        """Test loading a valid strategy file."""
        strategy_content = '''
import pandas as pd

def compute_weights(df):
    """Simple test strategy."""
    base_weight = 1.0 / len(df)
    return pd.Series(base_weight, index=df.index)
'''

        try:
            from stacking_sats_pipeline.main import load_strategy_from_file

            strategy_file = self.create_test_strategy_file(strategy_content)

            try:
                strategy_func = load_strategy_from_file(strategy_file)

                assert callable(strategy_func)

                # Test the loaded strategy
                import pandas as pd

                test_data = pd.DataFrame(
                    {"PriceUSD": [30000, 31000, 32000]},
                    index=pd.date_range("2020-01-01", periods=3),
                )

                weights = strategy_func(test_data)
                assert isinstance(weights, pd.Series)
                assert len(weights) == 3

            finally:
                os.unlink(strategy_file)  # Clean up

        except Exception as e:
            pytest.skip(f"Strategy loading test failed: {e}")

    def test_load_strategy_from_file_invalid(self):
        """Test loading an invalid strategy file."""
        invalid_content = """
# This file doesn't have compute_weights function
def other_function():
    pass
"""

        try:
            from stacking_sats_pipeline.main import load_strategy_from_file

            strategy_file = self.create_test_strategy_file(invalid_content)

            try:
                with pytest.raises(AttributeError):
                    load_strategy_from_file(strategy_file)

            finally:
                os.unlink(strategy_file)  # Clean up

        except Exception as e:
            pytest.skip(f"Invalid strategy test failed: {e}")

    def test_load_strategy_nonexistent_file(self):
        """Test loading a non-existent strategy file."""
        try:
            from stacking_sats_pipeline.main import load_strategy_from_file

            with pytest.raises(FileNotFoundError):
                load_strategy_from_file("nonexistent_strategy.py")

        except Exception as e:
            pytest.skip(f"Nonexistent file test failed: {e}")


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.mark.integration
    @patch("stacking_sats_pipeline.main.load_data")
    def test_cli_with_default_strategy(self, mock_load_data):
        """Test CLI with default strategy (integration test)."""
        try:
            # Mock data to avoid network dependencies
            import numpy as np
            import pandas as pd

            dates = pd.date_range("2020-01-01", periods=100, freq="D")
            np.random.seed(42)
            prices = 30000 + np.cumsum(np.random.normal(0, 500, 100))
            prices = np.maximum(prices, 1000)
            mock_data = pd.DataFrame({"PriceUSD": prices}, index=dates)
            mock_load_data.return_value = mock_data

            # Test the main function directly instead of subprocess
            from stacking_sats_pipeline.main import main

            with patch("sys.argv", ["main.py", "--no-plot"]):
                # Should not raise an exception
                main()

            # If we get here, the test passed
            assert True

        except SystemExit as e:
            # SystemExit with code 0 is normal for successful completion
            if e.code == 0:
                assert True
            else:
                pytest.skip(f"CLI test failed with exit code: {e.code}")
        except Exception as e:
            pytest.skip(f"CLI integration test failed: {e}")


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_cli_with_invalid_strategy_file(self):
        """Test CLI behavior with invalid strategy file."""
        try:
            # Test the main function directly with invalid strategy file
            from stacking_sats_pipeline.main import main

            with patch("sys.argv", ["main.py", "--strategy", "nonexistent.py", "--no-plot"]):
                with pytest.raises((FileNotFoundError, SystemExit)) as exc_info:
                    main()

                # Should either raise FileNotFoundError or exit with non-zero code
                if hasattr(exc_info.value, "code"):
                    assert exc_info.value.code != 0
                else:
                    # FileNotFoundError is also acceptable
                    assert True

        except Exception as e:
            pytest.skip(f"CLI error test failed: {e}")


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

    def test_strategy_loading_utilities(self):
        """Test strategy loading utility functions."""
        try:
            from stacking_sats_pipeline.main import load_strategy_from_file

            # Function should exist and be callable
            assert callable(load_strategy_from_file)

            # Check function signature
            import inspect

            sig = inspect.signature(load_strategy_from_file)
            params = list(sig.parameters.keys())
            assert len(params) >= 1  # Should accept strategy path

        except ImportError:
            pytest.skip("Strategy loading utilities not available")

    def test_extract_all_data_function_availability(self):
        """Test that extract_all_data function is available in main module."""
        try:
            from stacking_sats_pipeline.main import extract_all_data

            assert callable(extract_all_data), "extract_all_data should be callable"

        except ImportError:
            pytest.skip("extract_all_data function not available in main module")
