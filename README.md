# Stacking Sats Pipeline

A Bitcoin DCA strategy backtesting framework for testing strategies against historical price data.

## Requirements

- Python 3.11 or 3.12
- pip

## Installation

```bash
pip install stacking-sats-pipeline
```

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

### Data Extraction

Extract all data sources to local files for offline analysis:

#### CLI Usage
```bash
# Extract all data to CSV format
stacking-sats --extract-data csv

# Extract all data to Parquet format (smaller files, better compression)
stacking-sats --extract-data parquet

# Extract to specific directory
stacking-sats --extract-data csv --output-dir data/
stacking-sats --extract-data parquet -o exports/
```

#### Python API
```python
from stacking_sats_pipeline import extract_all_data

# Extract all data to CSV in current directory
extract_all_data("csv")

# Extract all data to Parquet in specific directory
extract_all_data("parquet", "data/exports/")
```

**What gets extracted:**
- 📈 **Bitcoin Price Data** (CoinMetrics) → `btc_coinmetrics.csv/parquet`
- 😨 **Fear & Greed Index** (Alternative.me) → `fear_greed.csv/parquet`  
- 💵 **U.S. Dollar Index** (FRED) → `dxy_fred.csv/parquet`*

*\*Requires `FRED_API_KEY` environment variable. Get a free key at [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)*

**File Format Benefits:**
- **CSV**: Human-readable, universally compatible
- **Parquet**: ~50% smaller files, faster loading, preserves data types

### Interactive Tutorial

```bash
pip install marimo
marimo edit tutorials/examples.py
```

### Command Line

```bash
stacking-sats --strategy path/to/your_strategy.py
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

## Development

For development and testing:

**Requirements**: Python 3.11 or 3.12

```bash
# Clone the repository
git clone https://github.com/hypertrial/stacking_sats_pipeline.git
cd stacking_sats_pipeline

# Set up development environment (installs dependencies + pre-commit hooks)
make setup-dev

# OR manually:
pip install -e ".[dev]"
pre-commit install

# Run tests
make test
# OR: pytest

# Code quality (MANDATORY - CI will fail if not clean)
make lint          # Fix linting issues
make format        # Format code
make check         # Check without fixing (CI-style)

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest -m integration        # Run only integration tests
```

### Code Quality Standards

**⚠️ MANDATORY**: All code must pass ruff linting and formatting checks.

- **Linting/Formatting**: We use [ruff](https://docs.astral.sh/ruff/) for both linting and code formatting
- **Pre-commit hooks**: Automatically run on every commit to catch issues early
- **CI enforcement**: Pull requests will fail if code doesn't meet standards

**Quick commands:**
```bash
make help          # Show all available commands
make lint          # Fix ALL issues (autopep8 + ruff + format)
make autopep8      # Fix line length issues specifically  
make format        # Format code with ruff only
make format-all    # Comprehensive formatting (autopep8 + ruff)
make check         # Check code quality (what CI runs)
```

For detailed testing documentation, see [TESTS.md](tests/TESTS.md).

### Contributing Data Sources

The data loading system is designed to be modular and extensible. To add new data sources (exchanges, APIs, etc.), see the [Data Loader Contribution Guide](stacking_sats_pipeline/data/CONTRIBUTE.md) which provides step-by-step instructions for implementing new data loaders.

## Command Line Options

```bash
# Basic usage
stacking-sats --strategy your_strategy.py

# Skip plots
stacking-sats --strategy your_strategy.py --no-plot

# Run simulation
stacking-sats --strategy your_strategy.py --simulate --budget 1000000

# Extract data
stacking-sats --extract-data csv --output-dir data/
stacking-sats --extract-data parquet -o exports/

# Show help
stacking-sats --help
```

## Project Structure

```
├── stacking_sats_pipeline/
│   ├── main.py          # Pipeline orchestrator
│   ├── backtest/        # Validation & simulation
│   ├── data/            # Modular data loading system
│   │   ├── coinmetrics_loader.py  # CoinMetrics data source
│   │   ├── data_loader.py         # Multi-source data loader
│   │   └── CONTRIBUTE.md          # Guide for adding data sources
│   ├── plot/            # Visualization
│   ├── strategy/        # Strategy templates
│   └── weights/         # Historical allocation calculator
├── tutorials/examples.py # Interactive notebook
└── tests/               # Comprehensive test suite
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