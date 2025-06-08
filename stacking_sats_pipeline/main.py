# Import all necessary functions and constants
import argparse
import importlib.util
import sys
from pathlib import Path

from .backtest.checks import (
    backtest_dynamic_dca,
    validate_strategy_comprehensive,
)
from .backtest.simulation import run_full_simulation
from .config import BACKTEST_END, BACKTEST_START
from .data.data_loader import load_data, validate_price_data
from .plot.plotting import (
    plot_features,
    plot_final_weights,
    plot_spd_comparison,
    plot_weight_sums_by_cycle,
)


def load_strategy_from_file(strategy_path: str):
    """
    Dynamically load a strategy function from a Python file.

    Args:
        strategy_path: Path to the Python file containing the strategy

    Returns:
        The compute_weights function from the strategy file
    """
    strategy_path = Path(strategy_path)

    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    if not strategy_path.suffix == ".py":
        raise ValueError(f"Strategy file must be a Python file (.py): {strategy_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    strategy_module = importlib.util.module_from_spec(spec)

    # Add current modules to the module's globals so it can import from them if needed
    # strategy_module.__dict__["stacking_sats"] = sys.modules.get("stacking_sats")

    # Import commonly needed modules into the strategy module's namespace
    import numpy as np
    import pandas as pd

    strategy_module.__dict__["pd"] = pd
    strategy_module.__dict__["np"] = np

    spec.loader.exec_module(strategy_module)

    # Check if compute_weights function exists
    if not hasattr(strategy_module, "compute_weights"):
        raise AttributeError(
            f"Strategy file must contain a 'compute_weights' function: {strategy_path}"
        )

    return strategy_module.compute_weights


def main():
    """
    Main function to run the Bitcoin DCA strategy backtesting and visualization.
    """
    parser = argparse.ArgumentParser(
        description="Run Bitcoin DCA strategy backtesting and visualization"
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="strategy/strategy_template.py",
        help="Path to the strategy Python file (default: strategy/strategy_template.py)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots and only run backtesting",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run Bitcoin accumulation simulation with $10M annual budget",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=10_000_000,
        help="Annual budget for simulation in USD (default: 10,000,000)",
    )

    args = parser.parse_args()

    # Derive strategy name from file path
    strategy_path = Path(args.strategy)
    if args.strategy == "strategy/strategy_template.py":
        strategy_name = "200-Day MA Strategy"
    else:
        strategy_name = strategy_path.stem  # Gets filename without extension

    # Load strategy function
    try:
        if args.strategy == "strategy/strategy_template.py":
            # Default case - import from package
            from .strategy import strategy_template

            compute_weights = strategy_template.compute_weights
            print(f"Loaded strategy from package: {args.strategy}")
        elif args.strategy.startswith("strategy/") and not Path(args.strategy).exists():
            # Try importing from package if it's a relative path within the package
            module_name = args.strategy.replace("strategy/", "").replace(".py", "")
            try:
                strategy_module = importlib.import_module(f"strategy.{module_name}")
                compute_weights = strategy_module.compute_weights
                print(f"Loaded strategy from package: {args.strategy}")
            except ImportError:
                raise FileNotFoundError(f"Strategy module not found: {module_name}")
        else:
            # Load from file path
            compute_weights = load_strategy_from_file(args.strategy)
            print(f"Loaded strategy from file: {args.strategy}")

    except Exception as e:
        print(f"Error loading strategy: {e}")
        sys.exit(1)

    # Load and validate data
    btc_df = load_data()
    validate_price_data(btc_df)
    btc_df = btc_df.loc[BACKTEST_START:BACKTEST_END]

    # Compute strategy weights
    weights = compute_weights(btc_df)

    # Run comprehensive validation checks from checks.py
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE STRATEGY VALIDATION")
    print(f"{'=' * 60}")

    validation_results = validate_strategy_comprehensive(
        btc_df, compute_weights, return_details=True
    )

    # Print detailed validation results
    if validation_results["validation_passed"]:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("❌ VALIDATION ISSUES FOUND:")
        if validation_results["has_negative_weights"]:
            print("  • Strategy has negative or zero weights")
        if validation_results["has_below_min_weights"]:
            print("  • Strategy has weights below minimum threshold")
        if validation_results["weights_not_sum_to_one"]:
            print("  • Strategy weights don't sum to 1.0 per cycle")
        if validation_results["underperforms_uniform"]:
            print("  • Strategy underperforms uniform DCA")
        if validation_results["is_forward_looking"]:
            print("  • Strategy may be using forward-looking information")

        # Show cycle-specific issues
        if validation_results["cycle_issues"]:
            print("\nCycle-specific issues:")
            for cycle, issues in validation_results["cycle_issues"].items():
                if issues:  # Only show cycles with issues
                    issue_list = []
                    if issues.get("has_negative_weights"):
                        issue_list.append("negative weights")
                    if issues.get("has_below_min_weights"):
                        issue_list.append("below min weights")
                    if issues.get("weights_not_sum_to_one"):
                        issue_list.append(
                            f"sum={issues.get('weight_sum', 'unknown'):.6f}"
                        )
                    if issues.get("underperforms_uniform"):
                        issue_list.append(
                            f"performance: {issues.get('dynamic_pct', 0):.2f}% < {issues.get('uniform_pct', 0):.2f}%"
                        )

                    if issue_list:
                        print(f"  {cycle}: {', '.join(issue_list)}")

    # Run simulation if requested
    if args.simulate:
        run_full_simulation(
            btc_df,
            compute_weights,
            strategy_label=strategy_name,
            annual_budget_usd=args.budget,
        )

    # Generate plots (only if not disabled)
    if not args.no_plot:
        plot_features(
            btc_df, weights=weights, start_date=BACKTEST_START, end_date=BACKTEST_END
        )
        plot_final_weights(weights, start_date=BACKTEST_START)
        plot_weight_sums_by_cycle(weights)

    # Run backtesting
    df_spd = backtest_dynamic_dca(
        btc_df, strategy_fn=compute_weights, strategy_label=strategy_name
    )
    # check_strategy_submission_ready(btc_df, strategy_fn=compute_weights)

    # Generate comparison plot (only if not disabled)
    if not args.no_plot:
        plot_spd_comparison(df_spd, strategy_name=strategy_name)


if __name__ == "__main__":
    main()
