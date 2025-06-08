import numpy as np
import pandas as pd

from config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS, MIN_WEIGHT


def _make_cycle_label(start, end, cycle_years):
    """Return a pretty label for a cycle of arbitrary length."""
    if cycle_years == 1:
        return f"{start.year}"
    return f"{start.year}–{end.year}"


def _get_cycle_boundaries(df_backtest, cycle_years):
    """Generate cycle boundaries for backtesting."""
    cycle_length = pd.DateOffset(years=cycle_years)
    current = df_backtest.index.min()

    while current <= df_backtest.index.max():
        cycle_end = current + cycle_length - pd.Timedelta(days=1)
        end_date = min(cycle_end, df_backtest.index.max())
        cycle_mask = (df_backtest.index >= current) & (df_backtest.index <= end_date)
        cycle = df_backtest.loc[cycle_mask]

        if cycle.empty:
            break

        yield current, end_date, cycle_mask, cycle
        current += cycle_length


def compute_cycle_spd(df, strategy_fn, *, cycle_years: int = CYCLE_YEARS):
    """Compute SPD stats per cycle of length `cycle_years`."""
    df_backtest = df.loc[BACKTEST_START:BACKTEST_END]
    full_weights = strategy_fn(df).fillna(0).clip(lower=0)
    inverted_prices = (1 / df_backtest["PriceUSD"]) * 1e8  # satoshi-per-dollar

    rows = []
    for current, end_date, cycle_mask, cycle in _get_cycle_boundaries(
        df_backtest, cycle_years
    ):
        label = _make_cycle_label(current, end_date, cycle_years)

        # Compute SPD metrics
        prices = cycle["PriceUSD"].values
        high, low = np.max(prices), np.min(prices)
        min_spd, max_spd = (1 / high) * 1e8, (1 / low) * 1e8

        cycle_inv = inverted_prices.loc[cycle.index]
        w_slice = full_weights.loc[cycle.index]

        dynamic_spd = (w_slice * cycle_inv).sum()
        uniform_spd = cycle_inv.mean()

        # Calculate static_dca SPD (30% percentile baseline)
        spd_range = max_spd - min_spd
        static_dca_spd = min_spd + (0.30 * spd_range)  # Exactly 30% percentile

        uniform_pct = (uniform_spd - min_spd) / spd_range * 100
        dynamic_pct = (dynamic_spd - min_spd) / spd_range * 100
        static_dca_pct = 30.0  # Always exactly 30%

        excess_pct = dynamic_pct - uniform_pct
        excess_vs_static_dca = dynamic_pct - static_dca_pct

        rows.append(
            {
                "cycle": label,
                "min_spd": min_spd,
                "max_spd": max_spd,
                "uniform_spd": uniform_spd,
                "dynamic_spd": dynamic_spd,
                "static_dca_spd": static_dca_spd,
                "uniform_pct": uniform_pct,
                "dynamic_pct": dynamic_pct,
                "static_dca_pct": static_dca_pct,
                "excess_pct": excess_pct,
                "excess_vs_static_dca": excess_vs_static_dca,
            }
        )

    return pd.DataFrame(rows).set_index("cycle")


def backtest_dynamic_dca(
    df,
    strategy_fn,
    *,
    strategy_label: str = "your_strategy",
    cycle_years: int = CYCLE_YEARS,
):
    """Convenience wrapper: print aggregate stats and return the cycle table."""
    res = compute_cycle_spd(df, strategy_fn, cycle_years=cycle_years)

    dyn_spd = res["dynamic_spd"]
    dyn_pct = res["dynamic_pct"]

    print(f"\nAggregated Metrics for {strategy_label}:")
    print("Dynamic SPD:")
    for k in ["min", "max", "mean", "median"]:
        print(f"  {k}: {getattr(dyn_spd, k)():.2f}")
    print("Dynamic SPD Percentile:")
    for k in ["min", "max", "mean", "median"]:
        print(f"  {k}: {getattr(dyn_pct, k)():.2f}")

    mean_excess = res["excess_pct"].mean()
    mean_excess_vs_static = res["excess_vs_static_dca"].mean()
    print(
        f"\nMean Excess SPD Percentile Difference (Dynamic – Uniform): {mean_excess:.2f}%"
    )
    print(
        f"Mean Excess SPD Percentile Difference (Dynamic – Static DCA 30%): {mean_excess_vs_static:.2f}%"
    )

    # Add detailed SPD comparison table (actual values)
    print(
        f"\nSPD Comparison (Uniform vs Static DCA vs {strategy_label}) - Sats per Dollar:"
    )
    print(
        f"{'Cycle':<8} {'Uniform SPD':<12} {'Static DCA SPD':<13} {f'{strategy_label} SPD':<15} {'vs Uniform':<12} {'vs Static':<12}"
    )
    print("-" * 85)
    for cycle, row in res.iterrows():
        uniform_diff = row["dynamic_spd"] - row["uniform_spd"]
        static_diff = row["dynamic_spd"] - row["static_dca_spd"]
        print(
            f"{cycle:<8} {row['uniform_spd']:<12.2f} {row['static_dca_spd']:<13.2f} {row['dynamic_spd']:<15.2f} {uniform_diff:<12.2f} {static_diff:<12.2f}"
        )

    # Add detailed SPD percentile comparison table
    print(f"\nSPD Percentile Comparison (Uniform vs Static DCA vs {strategy_label}):")
    print(
        f"{'Cycle':<8} {'Uniform %':<12} {'Static DCA %':<12} {f'{strategy_label} %':<15} {'vs Uniform':<12} {'vs Static':<12}"
    )
    print("-" * 80)
    for cycle, row in res.iterrows():
        print(
            f"{cycle:<8} {row['uniform_pct']:<12.2f} {row['static_dca_pct']:<12.2f} {row['dynamic_pct']:<15.2f} {row['excess_pct']:<12.2f} {row['excess_vs_static_dca']:<12.2f}"
        )

    print("\nExcess SPD Percentile Difference (Dynamic – Uniform) per Cycle:")
    for cycle, row in res.iterrows():
        print(f"  {cycle}: {row['excess_pct']:.2f}%")

    print("\nExcess SPD Percentile Difference (Dynamic – Static DCA 30%) per Cycle:")
    for cycle, row in res.iterrows():
        print(f"  {cycle}: {row['excess_vs_static_dca']:.2f}%")

    return res


def _validate_cycle_constraints(df_backtest, full_weights, cycle_years):
    """Validate weight constraints per cycle."""
    passed = True
    cycle_issues = {}
    validation_messages = []

    has_negative_weights = False
    has_below_min_weights = False
    weights_not_sum_to_one = False

    for current, end_date, cycle_mask, cycle in _get_cycle_boundaries(
        df_backtest, cycle_years
    ):
        w_slice = full_weights.loc[cycle.index]
        cycle_label = _make_cycle_label(current, end_date, cycle_years)
        cycle_issues[cycle_label] = {}

        # Criterion 1: strictly positive
        if (w_slice <= 0).any():
            validation_messages.append(
                f"[{cycle_label}] Some weights are zero or negative."
            )
            passed = False
            has_negative_weights = True
            cycle_issues[cycle_label]["has_negative_weights"] = True

        # Criterion 2: above minimum threshold
        if (w_slice < MIN_WEIGHT).any():
            validation_messages.append(
                f"[{cycle_label}] Some weights are below MIN_WEIGHT = {MIN_WEIGHT}."
            )
            passed = False
            has_below_min_weights = True
            cycle_issues[cycle_label]["has_below_min_weights"] = True

        # Criterion 3: weights must sum to 1 over the entire cycle
        total_weight = (
            w_slice.sum().sum() if isinstance(w_slice, pd.DataFrame) else w_slice.sum()
        )
        if not np.isclose(total_weight, 1.0, rtol=1e-5, atol=1e-8):
            validation_messages.append(
                f"[{cycle_label}] Total weights across the cycle do not sum to 1 (sum = {total_weight:.6f})."
            )
            passed = False
            weights_not_sum_to_one = True
            cycle_issues[cycle_label]["weights_not_sum_to_one"] = True
            cycle_issues[cycle_label]["weight_sum"] = float(total_weight)

    return (
        passed,
        cycle_issues,
        validation_messages,
        {
            "has_negative_weights": has_negative_weights,
            "has_below_min_weights": has_below_min_weights,
            "weights_not_sum_to_one": weights_not_sum_to_one,
        },
    )


def _validate_performance(df, strategy_fn, cycle_years):
    """Validate that strategy performance meets or exceeds uniform DCA."""
    passed = True
    cycle_issues = {}
    validation_messages = []
    underperforms_uniform = False

    spd_results = compute_cycle_spd(df, strategy_fn, cycle_years=cycle_years)
    for cycle, row in spd_results.iterrows():
        if cycle not in cycle_issues:
            cycle_issues[cycle] = {}

        if row["dynamic_pct"] < row["uniform_pct"]:
            validation_messages.append(
                f"[{cycle}] Strategy performance ({row['dynamic_pct']:.2f}%) is below threshold."
            )
            passed = False
            underperforms_uniform = True
            cycle_issues[cycle]["underperforms_uniform"] = True
            cycle_issues[cycle]["dynamic_pct"] = float(row["dynamic_pct"])
            cycle_issues[cycle]["uniform_pct"] = float(row["uniform_pct"])

    return (
        passed,
        cycle_issues,
        validation_messages,
        {"underperforms_uniform": underperforms_uniform},
    )


def _test_causality(strategy_fn, df, cycle_years):
    """Test if strategy is causal (not forward-looking)."""
    validation_messages = []
    is_forward_looking = False

    def perturb_func(df_future):
        """Replace future values with random noise."""
        np.random.seed(42)  # For reproducibility
        perturb_factor = np.random.uniform(1.5, 2.5)
        return df_future * perturb_factor

    def is_causal(
        construct_features_func,
        df_test,
        test_indices,
        perturb_func,
        rtol=1e-5,
        atol=1e-8,
    ):
        """Test if feature construction is causal."""
        features_original = construct_features_func(df_test)

        for t in test_indices:
            if t >= len(df_test):
                continue

            df_perturbed = df_test.copy()
            if t + 1 < len(df_perturbed):
                future_data = df_perturbed.iloc[t + 1 :].copy()
                df_perturbed.iloc[t + 1 :] = perturb_func(future_data)

            features_perturbed = construct_features_func(df_perturbed)
            original_val = features_original.iloc[t].fillna(0)
            perturbed_val = features_perturbed.iloc[t].fillna(0)

            if not np.allclose(original_val, perturbed_val, rtol=rtol, atol=atol):
                validation_messages.append(
                    f"Features at time index {t} change when future data is perturbed."
                )
                return False

        return True

    def is_weight_causal(
        strategy_func,
        df_test,
        test_indices,
        perturb_func,
        cycle_years,
        rtol=1e-5,
        atol=1e-8,
    ):
        """Test if weight computation is causal."""
        weights_original = strategy_func(df_test)

        for t in test_indices:
            if t >= len(df_test):
                continue

            test_date = df_test.index[t]
            start_year = pd.to_datetime(df_test.index.min()).year
            cycle_id = (test_date.year - start_year) // cycle_years

            cycle_start_year = start_year + cycle_id * cycle_years
            cycle_end_year = cycle_start_year + cycle_years - 1
            cycle_start = pd.Timestamp(f"{cycle_start_year}-01-01")
            cycle_end = pd.Timestamp(f"{cycle_end_year}-12-31")

            cycle_mask = (df_test.index >= cycle_start) & (df_test.index <= cycle_end)
            cycle_indices = df_test.index[cycle_mask]

            if len(cycle_indices) == 0 or test_date not in cycle_indices:
                continue

            future_in_cycle = cycle_indices[cycle_indices > test_date]
            if len(future_in_cycle) == 0:
                continue

            df_perturbed = df_test.copy()
            future_data = df_perturbed.loc[future_in_cycle].copy()
            df_perturbed.loc[future_in_cycle] = perturb_func(future_data)

            weights_perturbed = strategy_func(df_perturbed)

            if (
                test_date in weights_original.index
                and test_date in weights_perturbed.index
            ):
                original_weight = weights_original.loc[test_date]
                perturbed_weight = weights_perturbed.loc[test_date]

                if pd.isna(original_weight):
                    original_weight = 0
                if pd.isna(perturbed_weight):
                    perturbed_weight = 0

                if not np.allclose(
                    [original_weight], [perturbed_weight], rtol=rtol, atol=atol
                ):
                    validation_messages.append(
                        f"Weight at time {test_date.strftime('%Y-%m-%d')} changes when future data in the same cycle is perturbed. "
                        f"This indicates forward-looking bias in weight computation (e.g., cycle-wide normalization)."
                    )
                    return False

        return True

    try:
        # Set up test indices
        warm_up = 500
        data_len = len(df)
        num_test_points = 10
        test_step = (data_len - warm_up) // (num_test_points + 1)
        test_indices = [warm_up + i * test_step for i in range(1, num_test_points + 1)]

        # Try to find construct_features function
        construct_features = None
        if hasattr(strategy_fn, "__module__") and hasattr(strategy_fn, "__name__"):
            try:
                import sys

                module_name = strategy_fn.__module__
                if module_name in sys.modules:
                    strategy_module = sys.modules[module_name]
                    construct_features = getattr(
                        strategy_module, "construct_features", None
                    )
            except Exception:
                pass

        if construct_features is None and hasattr(strategy_fn, "__globals__"):
            try:
                construct_features = strategy_fn.__globals__.get(
                    "construct_features", None
                )
            except Exception:
                pass

        # Test feature causality if possible
        if construct_features is not None:
            is_feature_causal = is_causal(
                construct_features, df, test_indices, perturb_func
            )
            if not is_feature_causal:
                validation_messages.append(
                    "Strategy features may be forward-looking: they use information from future data."
                )
                is_forward_looking = True

        # Test weight causality
        is_weight_causal_result = is_weight_causal(
            strategy_fn, df, test_indices, perturb_func, cycle_years
        )
        if not is_weight_causal_result:
            is_forward_looking = True

        causality_check_error = (
            None
            if construct_features is not None
            else "No construct_features function found - only weight causality tested"
        )

    except Exception as e:
        validation_messages.append(f"Error in validation checks: {str(e)}")
        causality_check_error = str(e)

    return is_forward_looking, validation_messages, causality_check_error


def validate_strategy_comprehensive(
    df, strategy_fn, *, cycle_years: int = CYCLE_YEARS, return_details=False
):
    """Comprehensive validation of a strategy for submission readiness."""
    df_backtest = df.loc[BACKTEST_START:BACKTEST_END]
    full_weights = strategy_fn(df).fillna(0)

    passed = True
    all_validation_messages = []
    validation_results = {
        "validation_passed": True,
        "has_negative_weights": False,
        "has_below_min_weights": False,
        "weights_not_sum_to_one": False,
        "underperforms_uniform": False,
        "is_forward_looking": False,
    }

    # Validate cycle constraints
    constraints_passed, cycle_issues, constraint_messages, constraint_flags = (
        _validate_cycle_constraints(df_backtest, full_weights, cycle_years)
    )
    passed &= constraints_passed
    all_validation_messages.extend(constraint_messages)
    validation_results.update(constraint_flags)

    # Validate performance
    performance_passed, perf_cycle_issues, perf_messages, perf_flags = (
        _validate_performance(df, strategy_fn, cycle_years)
    )
    passed &= performance_passed
    all_validation_messages.extend(perf_messages)
    validation_results.update(perf_flags)

    # Merge cycle issues
    for cycle, issues in perf_cycle_issues.items():
        if cycle in cycle_issues:
            cycle_issues[cycle].update(issues)
        else:
            cycle_issues[cycle] = issues

    # Test causality
    is_forward_looking, causality_messages, causality_check_error = _test_causality(
        strategy_fn, df, cycle_years
    )
    validation_results["is_forward_looking"] = is_forward_looking
    if is_forward_looking:
        passed = False
    all_validation_messages.extend(causality_messages)
    if causality_check_error:
        validation_results["causality_check_error"] = causality_check_error

    validation_results["cycle_issues"] = cycle_issues
    validation_results["validation_passed"] = passed

    # Print results
    if passed:
        print(f"✅ Strategy passed all validation checks ({cycle_years}-year cycles).")
        print(f"   • Checked {len(cycle_issues)} cycles for weight constraints")
        print("   • Verified weights are positive and above minimum threshold")
        print("   • Confirmed weights sum to 1.0 per cycle")
        print("   • Validated strategy performance vs uniform DCA")

        if causality_check_error:
            print(f"   • Causality check: {causality_check_error}")
        else:
            print("   • ✅ Causality check: Strategy appears to be causal")
    else:
        print(f"❌ Strategy failed validation ({cycle_years}-year cycles):")
        for message in all_validation_messages:
            print(f"   {message}")

        if causality_check_error:
            print(
                f"   Note: Causality check could not be performed - {causality_check_error}"
            )
        elif not is_forward_looking:
            print("   • ✅ Causality check: Strategy appears to be causal")

    return validation_results if return_details else passed


def check_strategy_submission_ready(df, strategy_fn, *, cycle_years: int = CYCLE_YEARS):
    """Check if strategy is ready for submission using comprehensive validation."""
    print(f"Running submission readiness check with {cycle_years}-year cycles...")

    validation_results = validate_strategy_comprehensive(
        df, strategy_fn, cycle_years=cycle_years, return_details=True
    )

    passed = validation_results["validation_passed"]

    if passed:
        print("✅ Strategy is ready for submission.")
    else:
        print("⚠️ Fix issues above before submission.")

        # Print summary of issues
        issues_summary = []
        if validation_results["has_negative_weights"]:
            issues_summary.append("negative/zero weights")
        if validation_results["has_below_min_weights"]:
            issues_summary.append("weights below minimum threshold")
        if validation_results["weights_not_sum_to_one"]:
            issues_summary.append("weights don't sum to 1")
        if validation_results["underperforms_uniform"]:
            issues_summary.append("underperforms uniform DCA")
        if validation_results["is_forward_looking"]:
            issues_summary.append("forward-looking features")

        if issues_summary:
            print(f"Issues found: {', '.join(issues_summary)}")

    return passed
