# Stacking Sats Pipeline

A Bitcoin DCA strategy backtesting framework for testing strategies against historical price data.

## Quick Start

### Library Interface

```python
from stacking_sats_pipeline import backtest, strategy

# Simple function approach
def my_strategy(df):
    """Calculate weights based on price data"""
    # Your strategy logic here
    return weights

results = backtest(my_strategy)
results.summary()
results.plot()

# Or use decorator approach
@strategy(name="My Strategy", auto_backtest=True)
def my_strategy(df):
    return weights
```

> **Note**: Data is now loaded directly into memory from CoinMetrics (no CSV files needed). For legacy file-based loading, use `load_data(use_memory=False)`.

### Interactive Tutorial

```bash
pip install marimo
marimo edit tutorials/examples.py
```

### Command Line

```bash
pip install -r requirements.txt
python main.py --strategy path/to/your_strategy.py
```

## Usage Examples

### Basic Strategy

```python
def simple_ma_strategy(df):
    """Buy more when price is below 200-day moving average"""
    df = df.copy()
    past_price = df["PriceUSD"].shift(1)
    df["ma200"] = past_price.rolling(window=200, min_periods=1).mean()
    
    base_weight = 1.0 / len(df)
    weights = pd.Series(base_weight, index=df.index)
    
    # Buy 50% more when below MA
    below_ma = df["PriceUSD"] < df["ma200"]
    weights[below_ma] *= 1.5
    
    return weights / weights.sum()

results = backtest(simple_ma_strategy)
```

### Quick Comparison

```python
strategy1_perf = quick_backtest(strategy1)
strategy2_perf = quick_backtest(strategy2)
```

### Custom Parameters

```python
results = backtest(
    my_strategy,
    start_date="2021-01-01",
    end_date="2023-12-31",
    cycle_years=2
)
```

## Strategy Requirements

Your strategy function must:

```python
def your_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Args:
        df: DataFrame with 'PriceUSD' column and datetime index
        
    Returns:
        pd.Series with weights that sum to 1.0 per cycle
    """
    # Your logic here
    return weights
```

**Validation Rules:**
- Weights sum to 1.0 within each cycle
- All weights positive (≥ 1e-5)
- No forward-looking data
- Return pandas Series indexed by date

## Testing

The project includes a comprehensive test suite covering all major functionality:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest -m integration        # Run only integration tests

# Run tests with coverage
pytest --cov=stacking_sats_pipeline

# Run specific test files
pytest tests/test_backtest.py
pytest tests/test_strategy.py
```

For detailed testing documentation, see [TESTS.md](TESTS.md).

## Command Line Options

```bash
# Basic usage
python main.py --strategy your_strategy.py

# Skip plots
python main.py --strategy your_strategy.py --no-plot

# Run simulation
python main.py --strategy your_strategy.py --simulate --budget 1000000

# Historical weight calculator (coinmetrics data only)
python -m weights.weight_calculator 1000 2020-01-01 2023-12-31 --save
```

## Project Structure

```
├── main.py              # Pipeline orchestrator
├── tutorials/examples.py # Interactive notebook
├── backtest/            # Validation & simulation
├── data/                # Price data pipeline (in-memory loading)
├── plot/                # Visualization
├── strategy/            # Strategy templates
└── weights/             # Historical allocation calculator
```

## Output

The pipeline provides:
- **Validation Report**: Strategy compliance
- **Performance Metrics**: SPD (Sats Per Dollar) statistics
- **Comparative Analysis**: vs Uniform DCA and Static DCA
- **Visualizations**: Weight distribution plots

### Example Output
```
============================================================
COMPREHENSIVE STRATEGY VALIDATION
============================================================
✅ ALL VALIDATION CHECKS PASSED

Your Strategy Performance:
Dynamic SPD: mean=4510.21, median=2804.03
Dynamic SPD Percentile: mean=39.35%, median=43.80%

Mean Excess vs Uniform DCA: -0.40%
Mean Excess vs Static DCA: 9.35%
```