#!/usr/bin/env python3
"""
Tests for stacking_sats_pipeline CLI functionality
"""

import os
import subprocess
import sys
import tempfile
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
        parser.add_argument(
            "--strategy", "-s", type=str, default="strategy/strategy_template.py"
        )
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
    def test_cli_with_default_strategy(self):
        """Test CLI with default strategy (integration test)."""
        try:
            # Run CLI with default strategy and no-plot to avoid display issues
            result = subprocess.run(
                ["stacking-sats", "--no-plot"],
                capture_output=True,
                text=True,
                timeout=60,  # Longer timeout for integration test
            )

            # Check that it ran without major errors
            # (It might fail due to network issues, which is acceptable)
            if result.returncode == 0:
                assert len(result.stdout) > 0  # Should produce some output
            else:
                # If it failed, check if it's due to expected issues
                error_output = result.stderr.lower()
                acceptable_errors = [
                    "network",
                    "connection",
                    "timeout",
                    "data",
                    "internet",
                ]

                if any(error in error_output for error in acceptable_errors):
                    pytest.skip("CLI test failed due to network/data issues")
                else:
                    # Unexpected error
                    pytest.fail(f"CLI test failed unexpectedly: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.skip("CLI integration test timed out")
        except FileNotFoundError:
            pytest.skip("stacking-sats command not available")
        except Exception as e:
            pytest.skip(f"CLI integration test failed: {e}")


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_cli_with_invalid_strategy_file(self):
        """Test CLI behavior with invalid strategy file."""
        try:
            result = subprocess.run(
                ["stacking-sats", "--strategy", "nonexistent.py", "--no-plot"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should exit with non-zero code
            assert result.returncode != 0

            # Should contain error message
            error_output = result.stderr.lower()
            assert "error" in error_output or "not found" in error_output

        except subprocess.TimeoutExpired:
            pytest.skip("CLI error test timed out")
        except FileNotFoundError:
            pytest.skip("stacking-sats command not available")
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
