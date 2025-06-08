import numpy as np
import pandas as pd

# Import configuration constants
from config import BACKTEST_END, BACKTEST_START, CYCLE_YEARS, MIN_WEIGHT


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct technical indicators used for the strategy.
    Uses only past data for calculations to avoid look-ahead bias.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()

    # Extract relevant features (using only price here)
    df = df[["PriceUSD"]]

    # Shift the PriceUSD column by one to use only past data for our calculations
    past_price = df["PriceUSD"].shift(1)
    # Calculate 200-day moving average
    df["ma200"] = past_price.rolling(window=200, min_periods=1).mean()
    # Calculate 200-day standard deviation
    df["std200"] = past_price.rolling(window=200, min_periods=1).std()
    return df


# @register_strategy(ETH_WALLET_ADDRESS)
def compute_weights(df: pd.DataFrame, *, cycle_years: int = CYCLE_YEARS) -> pd.Series:
    """
    Dynamic-DCA weights based on 200-day MA distance.
    Sums to 1 **within each `cycle_years` block**.
    """
    # ── params ──────────────────────────────────────────────────────────
    REBALANCE_WINDOW = max(int(cycle_years * 365) // 2, 1)  # half the cycle
    ALPHA = 1.25
    # ── features ────────────────────────────────────────────────────────
    df_feat = construct_features(df).loc[BACKTEST_START:BACKTEST_END]
    weights = pd.Series(index=df_feat.index, dtype=float)

    start_year = pd.to_datetime(BACKTEST_START).year
    cycle_labels = df_feat.index.to_series().apply(
        lambda ts: (ts.year - start_year) // cycle_years
    )

    # ── loop over cycles ────────────────────────────────────────────────
    for _, cycle_df in df_feat.groupby(cycle_labels):
        N = len(cycle_df)
        base_wt = 1.0 / N
        temp_wt = np.full(N, base_wt)

        for day in range(N):
            price, ma, sd = (
                cycle_df["PriceUSD"].iat[day],
                cycle_df["ma200"].iat[day],
                cycle_df["std200"].iat[day],
            )
            if pd.isna(ma) or pd.isna(sd) or sd == 0 or price >= ma:
                continue

            z = (ma - price) / sd
            boosted = temp_wt[day] * (1 + ALPHA * z)
            excess = boosted - temp_wt[day]

            # redistribute from the **last part of the SAME cycle**
            start_r = max(N - REBALANCE_WINDOW, day + 1)
            idx_r = np.arange(start_r, N)
            if len(idx_r) == 0:
                continue
            reduction = excess / len(idx_r)
            if np.all(temp_wt[idx_r] - reduction >= MIN_WEIGHT):
                temp_wt[day] = boosted
                temp_wt[idx_r] -= reduction

        weights.loc[cycle_df.index] = temp_wt

    return weights
