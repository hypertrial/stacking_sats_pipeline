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
5. **Handle API keys** (if required) via environment variables

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
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Load environment variables if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

class MySourceLoader:
    """Loader for My Data Source."""
    
    BASE_URL = "https://api.mysource.com/btc"
    DEFAULT_FILENAME = "btc_mysource.csv"
    
    def __init__(self, data_dir: str | Path | None = None, api_key: Optional[str] = None):
        """Initialize the loader."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('MY_SOURCE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Please set MY_SOURCE_API_KEY in your .env file or "
                "pass api_key parameter. Get a free API key at: https://mysource.com/api"
            )
    
    def load(self, use_memory: bool = True, path: str | Path | None = None, file_format: str = "csv") -> pd.DataFrame:
        """
        Load data from your source.
        
        This is the REQUIRED method that implements the DataLoader protocol.
        """
        if use_memory:
            return self.load_from_web()
        else:
            return self.load_from_file(path, file_format=file_format)
    
    def load_from_web(self) -> pd.DataFrame:
        """Download data directly from your API/source."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'symbol': 'BTC',
            'format': 'json'
        }
        
        try:
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            
            data = resp.json()
            # Process the response into a DataFrame
            df = pd.DataFrame(data.get('prices', []))
            
            # Ensure consistent format: DatetimeIndex with price columns
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'price': 'PriceUSD'}, inplace=True)
            
            # Remove duplicates and sort
            df = df.loc[~df.index.duplicated(keep="last")].sort_index()
            
            self._validate_data(df)
            return df
            
        except requests.exceptions.RequestException as e:
            logging.error("Failed to download data from MySource: %s", e)
            raise
        except Exception as e:
            logging.error("Failed to process MySource data: %s", e)
            raise
    
    def load_from_file(self, path: str | Path | None = None, file_format: str = "csv") -> pd.DataFrame:
        """Load data from a local file."""
        if path is None:
            if file_format == "parquet":
                path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
            else:
                path = self.data_dir / self.DEFAULT_FILENAME
        
        if not Path(path).exists():
            logging.info("File not found, downloading from API...")
            df = self.load_from_web()
            if file_format == "parquet":
                df.to_parquet(path)
            else:
                df.to_csv(path)
            return df
        
        if file_format == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
        self._validate_data(df)
        return df
    
    def extract_to_parquet(self, path: str | Path | None = None) -> None:
        """Extract data from the web API and save to Parquet format."""
        if path is None:
            path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
        
        df = self.load_from_web()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logging.info("Data extracted to Parquet: %s", path)
    
    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load data from a Parquet file."""
        if path is None:
            path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
        
        if not Path(path).exists():
            logging.info("Parquet file not found, extracting from API...")
            self.extract_to_parquet(path)
        
        df = pd.read_parquet(path)
        self._validate_data(df)
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate your data format."""
        if df.empty:
            raise ValueError("Data is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        if "PriceUSD" not in df.columns:
            raise ValueError("PriceUSD column is required")

### 2. Working with API Keys

#### Environment Variable Setup
Create a `.env` file in your project root:

```bash
# .env file
MY_SOURCE_API_KEY=your_actual_api_key_here
FRED_API_KEY=your_fred_api_key_here
COINMARKETCAP_API_KEY=your_cmc_api_key_here
```

#### API Key Best Practices

1. **Never hardcode API keys** in your source code
2. **Use environment variables** to store sensitive credentials
3. **Provide clear error messages** when API keys are missing
4. **Support multiple ways** to pass API keys (parameter or environment)
5. **Include documentation** about where to get API keys

#### Example with API Key Handling

```python
class APIKeyLoader:
    """Example loader showing proper API key handling."""
    
    def __init__(self, data_dir: str | Path | None = None, api_key: Optional[str] = None):
        # Try multiple sources for API key
        self.api_key = (
            api_key or                           # Passed directly
            os.getenv('MY_API_KEY') or          # Environment variable
            os.getenv('MY_API_TOKEN')           # Alternative env var name
        )
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Please:\n"
                "1. Set MY_API_KEY in your .env file, or\n"
                "2. Pass api_key parameter to the constructor\n"
                "3. Get a free API key at: https://example.com/api"
            )
    
    def load_from_web(self) -> pd.DataFrame:
        """Load data with proper error handling for API issues."""
        headers = {'X-API-Key': self.api_key}
        
        try:
            response = requests.get(self.BASE_URL, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Handle rate limiting
            if response.status_code == 429:
                raise ValueError(
                    "Rate limit exceeded. Please wait before making more requests."
                )
                
            # Handle authentication errors
            if response.status_code == 401:
                raise ValueError(
                    "Invalid API key. Please check your API key and try again."
                )
                
            return self._process_response(response)
            
        except requests.exceptions.RequestException as e:
            logging.error("API request failed: %s", e)
            raise
```

### 3. Register Your Loader

#### Option A: Add to MultiSourceDataLoader initialization

Edit `data_loader.py` and add your loader to the `__init__` method:

```python
# In data_loader.py, MultiSourceDataLoader.__init__
self.loaders: Dict[str, DataLoader] = {
    'coinmetrics': CoinMetricsLoader(self.data_dir),
    'fred': FREDLoader(self.data_dir),
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

### 4. Update Imports (Optional)

If you want your loader to be available at the package level, add it to `__init__.py`:

```python
# In __init__.py
from .my_source_loader import MySourceLoader

__all__ = [
    # ... existing exports ...
    "MySourceLoader",
]
```

## Parquet Support Requirements

### Dependencies
To support Parquet format in your data loader, ensure that `pyarrow` is available:

```python
# At the top of your loader file
try:
    import pyarrow  # Required for Parquet support
except ImportError:
    raise ImportError(
        "pyarrow is required for Parquet support. Install with: pip install pyarrow>=10.0.0"
    )
```

### Required Methods for Parquet Support
Your loader should implement these methods to fully support Parquet format:

1. **`load()`** - Updated to accept `file_format` parameter
2. **`load_from_file()`** - Updated to handle both CSV and Parquet
3. **`extract_to_parquet()`** - Extract data directly to Parquet format
4. **`load_from_parquet()`** - Load data from Parquet files

### Parquet Benefits
- **Smaller file sizes**: Typically 40-60% of CSV size due to efficient compression
- **Faster loading**: Binary format with column-oriented storage
- **Type preservation**: Maintains data types including DatetimeIndex
- **Schema validation**: Built-in schema validation and consistency checking

### Implementation Pattern

```python
class YourParquetEnabledLoader:
    DEFAULT_FILENAME = "data.csv"
    
    def load(self, use_memory: bool = True, path: str | Path | None = None, file_format: str = "csv") -> pd.DataFrame:
        """Load with file format support."""
        if use_memory:
            return self.load_from_web()
        else:
            return self.load_from_file(path, file_format=file_format)
    
    def load_from_file(self, path: str | Path | None = None, file_format: str = "csv") -> pd.DataFrame:
        """Load from file with format selection."""
        if path is None:
            if file_format == "parquet":
                path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
            else:
                path = self.data_dir / self.DEFAULT_FILENAME
        
        if not Path(path).exists():
            df = self.load_from_web()
            if file_format == "parquet":
                df.to_parquet(path)
            else:
                df.to_csv(path)
            return df
        
        if file_format == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        return df
    
    def extract_to_parquet(self, path: str | Path | None = None) -> None:
        """Dedicated Parquet extraction method."""
        if path is None:
            path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
        
        df = self.load_from_web()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    
    def load_from_parquet(self, path: str | Path | None = None) -> pd.DataFrame:
        """Dedicated Parquet loading method."""
        if path is None:
            path = self.data_dir / f"{self.DEFAULT_FILENAME.replace('.csv', '.parquet')}"
        
        if not Path(path).exists():
            self.extract_to_parquet(path)
        
        return pd.read_parquet(path)
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

## API Key Security Guidelines

### Do's ✅
- Store API keys in `.env` files
- Add `.env` to your `.gitignore` (already done in this project)
- Use environment variables in production
- Validate API keys before making requests
- Provide clear error messages for authentication failures
- Document where users can obtain API keys
- Handle rate limiting gracefully

### Don'ts ❌
- Never commit API keys to version control
- Don't hardcode API keys in source code
- Don't share API keys in documentation or examples
- Don't log API keys (even in debug mode)
- Don't use the same API key across multiple projects if avoidable

### Example .env Template
Create a `.env.example` file for users:

```bash
# API Keys for Data Sources
# =========================
# Copy this file to .env and add your actual API keys

# Federal Reserve Economic Data (FRED)
# Get your key at: https://fred.stlouisfed.org/docs/api/api_key.html
# FRED_API_KEY=your_fred_api_key_here

# My Data Source
# Get your key at: https://mysource.com/api
# MY_SOURCE_API_KEY=your_api_key_here
```

## Testing Your Loader

Create tests for your loader in `tests/test_data.py`:

```python
class TestMySourceLoader:
    """Tests for MySourceLoader."""
    
    @patch("stacking_sats_pipeline.data.my_source_loader.requests.get")
    def test_load_from_web_mocked(self, mock_get):
        """Test loading with mocked API response."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'prices': [
                {'timestamp': 1577836800, 'price': 7200.17},
                {'timestamp': 1577923200, 'price': 6985.47}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test with API key
        loader = MySourceLoader(api_key="test_api_key")
        df = loader.load_from_web()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "PriceUSD" in df.columns
        
        # Verify API key was used in headers
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'headers' in call_args.kwargs
        assert 'Authorization' in call_args.kwargs['headers']
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises appropriate error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                MySourceLoader()
            
            assert "API key is required" in str(exc_info.value)
    
    @patch("stacking_sats_pipeline.data.my_source_loader.requests.get")
    def test_api_authentication_error(self, mock_get):
        """Test handling of API authentication errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
        mock_get.return_value = mock_response
        
        loader = MySourceLoader(api_key="invalid_key")
        
        with pytest.raises(requests.exceptions.HTTPError):
            loader.load_from_web()
    
    @patch("stacking_sats_pipeline.data.my_source_loader.requests.get")
    def test_rate_limiting_handling(self, mock_get):
        """Test handling of rate limiting."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429")
        mock_get.return_value = mock_response
        
        loader = MySourceLoader(api_key="test_key")
        
        with pytest.raises(requests.exceptions.HTTPError):
            loader.load_from_web()
    
    def test_integration_with_multisource_loader(self):
        """Test integration with MultiSourceDataLoader."""
        loader = MultiSourceDataLoader()
        
        # Add custom loader (with mocked API key)
        with patch.dict(os.environ, {'MY_SOURCE_API_KEY': 'test_key'}):
            custom_loader = MySourceLoader()
            loader.add_loader('mysource', custom_loader)
        
        assert 'mysource' in loader.get_available_sources()
    
    @patch("stacking_sats_pipeline.data.my_source_loader.requests.get")
    def test_parquet_functionality(self, mock_get):
        """Test Parquet extraction and loading."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'prices': [
                {'timestamp': 1577836800, 'price': 7200.17},
                {'timestamp': 1577923200, 'price': 6985.47}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        loader = MySourceLoader(api_key="test_key")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "test_data.parquet"
            
            # Test extraction to Parquet
            loader.extract_to_parquet(parquet_path)
            assert parquet_path.exists()
            
            # Test loading from Parquet
            df_parquet = loader.load_from_parquet(parquet_path)
            assert isinstance(df_parquet, pd.DataFrame)
            assert len(df_parquet) == 2
            assert "PriceUSD" in df_parquet.columns
            assert isinstance(df_parquet.index, pd.DatetimeIndex)
            
            # Test file format parameter in load method
            df_format = loader.load(use_memory=False, path=parquet_path, file_format="parquet")
            pd.testing.assert_frame_equal(df_parquet, df_format)
    
    def test_parquet_file_size_efficiency(self):
        """Test that Parquet files are smaller than CSV files."""
        # This test would compare file sizes and validate compression efficiency
        # Implementation depends on your specific data structure
        pass
```

## Advanced API Integration Patterns

### Rate Limiting and Retries

```python
import time
from functools import wraps

def rate_limited(max_calls=60, period=60):
    """Decorator to implement rate limiting."""
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than the period
            calls[:] = [call_time for call_time in calls if now - call_time < period]
            
            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

class RateLimitedLoader:
    @rate_limited(max_calls=100, period=3600)  # 100 calls per hour
    def load_from_web(self):
        # Your API call here
        pass
```

### Caching API Responses

```python
import hashlib
import json
from pathlib import Path

class CachedAPILoader:
    def __init__(self, cache_dir: Path = None, cache_ttl: int = 3600):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'stacking_sats'
        self.cache_ttl = cache_ttl  # Time to live in seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, url: str, params: dict) -> str:
        """Generate cache key from URL and parameters."""
        cache_data = {'url': url, 'params': sorted(params.items())}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        age = time.time() - cache_file.stat().st_mtime
        return age < self.cache_ttl
    
    def load_from_web(self) -> pd.DataFrame:
        cache_key = self._get_cache_key(self.BASE_URL, self.params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Try to load from cache first
        if self._is_cache_valid(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        # Fetch from API and cache the result
        df = self._fetch_from_api()
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(df.to_dict('records'), f)
        
        return df
```

## Real-World Examples

### FRED Loader (Already Implemented)
The project includes a complete example with the FRED (Federal Reserve Economic Data) loader:
- File: `fred_loader.py`
- API: Federal Reserve Economic Data API
- Authentication: API key via environment variable
- Data: U.S. Dollar Index (DTWEXBGS)

### CoinMetrics Loader (No API Key Required)
- File: `coinmetrics_loader.py`
- Data Source: GitHub CSV file
- No authentication required
- Good example of simple data loading

## Best Practices Summary

1. **Error Handling**: Wrap API calls in try/catch blocks with informative error messages
2. **Rate Limiting**: Respect API rate limits and add appropriate delays if needed
3. **Caching**: Consider implementing local file caching for expensive API calls
4. **Validation**: Always validate data format and content
5. **Logging**: Use the logging module to provide helpful debug information
6. **Documentation**: Document your data source, API limitations, and data format
7. **Security**: Never commit API keys; use environment variables
8. **Testing**: Write comprehensive tests including mocked API responses and Parquet functionality
9. **Authentication**: Handle auth errors gracefully with clear error messages
10. **Backwards Compatibility**: Provide convenience functions for common use cases
11. **Parquet Support**: Implement full Parquet support for better performance and file sizes
12. **File Format Flexibility**: Support both CSV and Parquet formats with proper parameter handling
13. **Type Preservation**: Ensure Parquet files maintain proper data types and index structures
14. **Storage Efficiency**: Leverage Parquet's compression benefits for large datasets

## Example Usage

Once your loader is registered with full Parquet support, users can:

```python
from stacking_sats_pipeline.data import load_data, load_and_merge_data

# Load from your source (API key from .env file)
data = load_data('mysource')

# Load with Parquet format for better performance
data_parquet = load_data('mysource', file_format='parquet')

# Merge multiple sources including yours
merged = load_and_merge_data(['coinmetrics', 'fred', 'mysource'])

# Direct usage with API key parameter
from stacking_sats_pipeline.data import MySourceLoader
loader = MySourceLoader(api_key="your_key_here")

# Load from web API
data = loader.load(use_memory=True)

# Load from CSV file
data_csv = loader.load(use_memory=False, file_format='csv')

# Load from Parquet file (faster, smaller)
data_parquet = loader.load(use_memory=False, file_format='parquet')

# Extract data directly to Parquet format
loader.extract_to_parquet('my_btc_data.parquet')

# Load specifically from Parquet
data_from_parquet = loader.load_from_parquet('my_btc_data.parquet')
```

### Parquet Integration with Backtest Results

Your data can also be used with the enhanced backtest system that supports Parquet export:

```python
from stacking_sats_pipeline import backtest
from stacking_sats_pipeline.strategy import uniform_strategy

# Run a backtest with your data
data = load_data('mysource')
results = backtest(uniform_strategy, data)

# Export results to Parquet (smaller file size)
results.save_weights_to_parquet('backtest_results.parquet')

# Or use the generic save method with format selection
results.save_weights('backtest_results', format='parquet')
```

## Need Help?

- Check the `fred_loader.py` for a complete API key example with Parquet support
- Check the `coinmetrics_loader.py` for a simple no-auth example with Parquet support
- Look at existing tests in `tests/test_data.py` and `tests/test_data_pipeline_e2e.py`
- Review the comprehensive end-to-end tests for Parquet functionality examples
- Ensure your loader implements the `DataLoader` protocol correctly
- Test with both memory and file-based loading (CSV and Parquet)
- Always test error conditions (invalid API keys, rate limits, etc.)
- Test Parquet extraction, loading, and file size efficiency
- Validate data type preservation between CSV and Parquet formats
- Ensure your Parquet implementation works with the full pipeline including backtest exports 