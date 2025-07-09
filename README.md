# Stacking Sats Pipeline

A data engineering pipeline for extracting, loading, and merging cryptocurrency and financial data from multiple sources.

## Requirements

- Python 3.11 or 3.12
- pip

## Installation

```bash
pip install stacking-sats-pipeline
```

## Quick Start

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

### Data Loading

```python
from stacking_sats_pipeline import load_data

# Load Bitcoin price data
df = load_data()

# Load specific data source
from stacking_sats_pipeline.data import CoinMetricsLoader
loader = CoinMetricsLoader()
btc_data = loader.load_from_web()
```

**What gets extracted:**

- 📈 **Bitcoin Price Data** (CoinMetrics) → `btc_coinmetrics.csv/parquet`
- 😨 **Fear & Greed Index** (Alternative.me) → `fear_greed.csv/parquet`
- 💵 **U.S. Dollar Index** (FRED) → `dxy_fred.csv/parquet`\*

_\*Requires `FRED_API_KEY` environment variable. Get a free key at [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)_

**File Format Benefits:**

- **CSV**: Human-readable, universally compatible
- **Parquet**: ~50% smaller files, faster loading, preserves data types

### Multi-Source Data Loading

```python
from stacking_sats_pipeline.data import MultiSourceDataLoader

# Load and merge data from all available sources
loader = MultiSourceDataLoader()
available_sources = loader.get_available_sources()
merged_df = loader.load_and_merge(available_sources)

# Available sources: coinmetrics, feargreed, fred (if API key available)
print(f"Available data sources: {available_sources}")
print(f"Merged data shape: {merged_df.shape}")
```

## Data Sources

### CoinMetrics (Bitcoin Price Data)

```python
from stacking_sats_pipeline.data import CoinMetricsLoader

loader = CoinMetricsLoader(data_dir="data/")
df = loader.load_from_web()  # Fetch latest data
df = loader.load_from_file()  # Load cached data (fetches if missing)

# Extract to files
csv_path = loader.extract_to_csv()
parquet_path = loader.extract_to_parquet()
```

### Fear & Greed Index

```python
from stacking_sats_pipeline.data import FearGreedLoader

loader = FearGreedLoader(data_dir="data/")
df = loader.load_from_web()
```

### FRED (Federal Reserve Economic Data)

```python
import os
os.environ['FRED_API_KEY'] = 'your_api_key_here'

from stacking_sats_pipeline.data import FREDLoader

loader = FREDLoader(data_dir="data/")
df = loader.load_from_web()  # DXY (Dollar Index) data
```

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
# Extract data
stacking-sats --extract-data csv --output-dir data/
stacking-sats --extract-data parquet -o exports/

# Show help
stacking-sats --help
```

## Project Structure

```
├── stacking_sats_pipeline/
│   ├── main.py                    # Pipeline orchestrator and CLI
│   ├── config.py                  # Configuration constants
│   ├── data/                      # Modular data loading system
│   │   ├── coinmetrics_loader.py  # CoinMetrics data source
│   │   ├── fear_greed_loader.py   # Fear & Greed Index data source
│   │   ├── fred_loader.py         # FRED economic data source
│   │   ├── data_loader.py         # Multi-source data loader
│   │   └── CONTRIBUTE.md          # Guide for adding data sources
│   └── __init__.py                # Package exports
├── tutorials/examples.py          # Interactive examples
└── tests/                         # Comprehensive test suite
```

## API Reference

### Core Functions

```python
from stacking_sats_pipeline import (
    extract_all_data,           # Extract all data sources to files
    load_data,                  # Load Bitcoin price data
    validate_price_data,        # Validate price data quality
    extract_btc_data_to_csv,    # Extract Bitcoin data to CSV
    extract_btc_data_to_parquet # Extract Bitcoin data to Parquet
)
```

### Configuration Constants

```python
from stacking_sats_pipeline import (
    BACKTEST_START,    # Default start date for data range
    BACKTEST_END,      # Default end date for data range
    CYCLE_YEARS,       # Default cycle period
    MIN_WEIGHT,        # Minimum weight threshold
    PURCHASE_FREQ      # Default purchase frequency
)
```

## Data Validation

All data sources include built-in validation:

```python
from stacking_sats_pipeline import validate_price_data

# Validate Bitcoin price data
df = load_data()
is_valid = validate_price_data(df)

# Custom validation with specific requirements
requirements = {
    'required_columns': ['PriceUSD', 'Volume'],
    'min_price': 100,
    'max_price': 1000000
}
is_valid = validate_price_data(df, **requirements)
```

## File Format Support

The pipeline supports both CSV and Parquet formats:

- **CSV**: Universal compatibility, human-readable
- **Parquet**: Better compression (~50% smaller), faster loading, preserves data types

```python
# CSV format
extract_all_data("csv", "output_dir/")

# Parquet format
extract_all_data("parquet", "output_dir/")
```

## Timestamp Handling

All data sources normalize timestamps to midnight UTC for consistent merging:

```python
loader = MultiSourceDataLoader()
merged_df = loader.load_and_merge(['coinmetrics', 'fred'])

# All timestamps are normalized to 00:00:00 UTC
print(merged_df.index.tz)  # UTC
print(merged_df.index.time[0])  # 00:00:00
```

## Error Handling

The pipeline includes comprehensive error handling:

```python
try:
    df = extract_all_data("csv")
except Exception as e:
    print(f"Data extraction failed: {e}")
    # Partial extraction may have succeeded
```

Individual data sources fail gracefully - if one source is unavailable, others will still be extracted.
