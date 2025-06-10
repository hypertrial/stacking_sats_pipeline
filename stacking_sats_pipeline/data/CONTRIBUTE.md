# Contributing New Data Loaders

This guide explains how to add new data sources to the `stacking_sats_pipeline` data loading system.

## Overview

The data loading system is designed to be modular and extensible. Each data source has its own loader class that implements the `DataLoader` protocol, and all loaders are managed by the `MultiSourceDataLoader` class.

## Quick Start

To add a new data source, you need to:

1. **Create a new loader file** (e.g., `my_source_loader.py`)
2. **Implement the DataLoader protocol**
3. **Register your loader** with the `MultiSourceDataLoader`
4. **Update the imports** in `__init__.py`

## Step-by-Step Guide

### 1. Create Your Loader Class

Create a new file in the `stacking_sats_pipeline/data/` directory:

```python
# my_source_loader.py
"""
My custom data source loader for BTC price data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests  # or whatever library you need

class MySourceLoader:
    """Loader for My Data Source."""
    
    BASE_URL = "https://api.mysource.com/btc"
    DEFAULT_FILENAME = "btc_mysource.csv"
    
    def __init__(self, data_dir: str | Path | None = None):
        """Initialize the loader."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
    
    def load(self, use_memory: bool = True, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load data from your source.
        
        This is the REQUIRED method that implements the DataLoader protocol.
        """
        if use_memory:
            return self.load_from_web()
        else:
            return self.load_from_file(path)
    
    def load_from_web(self) -> pd.DataFrame:
        """Download data directly from your API/source."""
        # Your implementation here
        resp = requests.get(self.BASE_URL)
        # Process the response into a DataFrame
        df = pd.DataFrame()  # Replace with actual processing
        
        # Ensure consistent format: DatetimeIndex with price columns
        df.index = pd.to_datetime(df.index)
        self._validate_data(df)
        return df
    
    def load_from_file(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load data from a local file."""
        if path is None:
            path = self.data_dir / self.DEFAULT_FILENAME
        
        # Your file loading logic here
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        self._validate_data(df)
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate your data format."""
        # Add validation specific to your data source
        if df.empty:
            raise ValueError("Data is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        # Add other validations as needed
```

### 2. Register Your Loader

There are two ways to register your loader:

#### Option A: Add to MultiSourceDataLoader initialization

Edit `data_loader.py` and add your loader to the `__init__` method:

```python
# In data_loader.py, MultiSourceDataLoader.__init__
self.loaders: Dict[str, DataLoader] = {
    'coinmetrics': CoinMetricsLoader(self.data_dir),
    'mysource': MySourceLoader(self.data_dir),  # Add this line
}
```

#### Option B: Add dynamically (recommended for external contributions)

```python
from stacking_sats_pipeline.data import MultiSourceDataLoader
from .my_source_loader import MySourceLoader

# Create loader instance
loader = MultiSourceDataLoader()

# Add your custom loader
loader.add_loader('mysource', MySourceLoader())

# Now you can use it
data = loader.load_from_source('mysource')
```

### 3. Update Imports (Optional)

If you want your loader to be available at the package level, add it to `__init__.py`:

```python
# In __init__.py
from .my_source_loader import MySourceLoader

__all__ = [
    # ... existing exports ...
    "MySourceLoader",
]
```

## Data Format Requirements

### Index
- Must be a `pd.DatetimeIndex`
- Should be timezone-naive or consistently timezone-aware
- Should be sorted chronologically

### Columns
- Must contain at least one price column
- Price columns should contain "Price" in the name (e.g., "PriceUSD", "PriceBTC")
- Column names should be descriptive and avoid conflicts with other sources

### Example DataFrame Structure
```python
                PriceUSD  Volume24h   MarketCap
time
2020-01-01      7200.17   12345678   131000000
2020-01-02      6985.47   23456789   127000000
...
```

## Testing Your Loader

Create tests for your loader in `tests/test_data.py`:

```python
class TestMySourceLoader:
    """Tests for MySourceLoader."""
    
    def test_load_from_source(self):
        """Test loading from your source."""
        loader = MultiSourceDataLoader()
        # Add your loader to the test
        loader.add_loader('mysource', MySourceLoader())
        
        # Test loading (you may want to mock the API calls)
        df = loader.load_from_source('mysource')
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
```

## Best Practices

1. **Error Handling**: Wrap API calls in try/catch blocks with informative error messages
2. **Rate Limiting**: Respect API rate limits and add appropriate delays if needed
3. **Caching**: Consider implementing local file caching for expensive API calls
4. **Validation**: Always validate data format and content
5. **Logging**: Use the logging module to provide helpful debug information
6. **Documentation**: Document your data source, API limitations, and data format

## Example Usage

Once your loader is registered, users can:

```python
from stacking_sats_pipeline.data import load_data, load_and_merge_data

# Load from your source
data = load_data('mysource')

# Merge multiple sources
merged = load_and_merge_data(['coinmetrics', 'mysource'])
```

## Need Help?

- Check the `coinmetrics_loader.py` for a complete example
- Look at existing tests in `tests/test_data.py`
- Ensure your loader implements the `DataLoader` protocol correctly
- Test with both memory and file-based loading 