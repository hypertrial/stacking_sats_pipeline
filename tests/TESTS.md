# Testing Documentation

This document provides comprehensive information about the test suite for the Stacking Sats Pipeline data engineering project.

## Requirements

- Python 3.11 or 3.12
- pytest and development dependencies

## Overview

The test suite is built using pytest and covers all major components of the data engineering pipeline:

- **Data loading and validation** (`test_data.py`)
- **Data extraction and export** (`test_data_extraction.py`)
- **End-to-end data pipeline including Parquet support** (`test_data_pipeline_e2e.py`)
- **Command-line interface** (`test_cli.py`)
- **Configuration and metadata** (`test_config.py`)

## Quick Start

### Installation

**Requirements**: Python 3.11 or 3.12

```bash
# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data.py

# Run specific test class
pytest tests/test_data.py::TestDataLoading

# Run specific test method
pytest tests/test_data.py::TestDataLoading::test_load_data_integration
```

### Test Categories

Tests are organized into two main categories:

#### Unit Tests (Default)

Fast tests that don't require external dependencies or network access:

```bash
pytest -m "not integration"
```

#### Integration Tests

Tests that may require network access or external data:

```bash
pytest -m integration
```

## Test Structure

### Test Files Overview

| File                        | Purpose                     | Key Features                                                                                                            |
| --------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `test_data.py`              | Data loading and validation | CoinMetrics integration, CSV handling, data validation, **timestamp alignment**                                         |
| `test_data_extraction.py`   | Data extraction and export  | Multi-format export (CSV/Parquet), CLI integration, file validation, **timestamp alignment verification**               |
| `test_data_pipeline_e2e.py` | End-to-end data pipeline    | Real API testing, Parquet support, file format conversion, data quality validation, **timestamp alignment integration** |
| `test_cli.py`               | Command-line interface      | Argument parsing, error handling, data extraction commands                                                              |
| `test_config.py`            | Configuration and metadata  | Package metadata, constants validation                                                                                  |

## Detailed Test Documentation

### Data Loading Tests (`test_data.py`)

The data loading system has been refactored into a modular architecture with separate loaders for different data sources. See `stacking_sats_pipeline/data/CONTRIBUTE.md` for information on adding new data sources.

#### TestDataLoading

Tests data loading functionality with the new modular data loading system:

- **Integration Data Loading**: `test_load_data_integration()`

  - Tests `load_data()` with real CoinMetrics API via the new modular system
  - Validates DataFrame structure and data quality
  - Checks price data reasonableness

- **Web Data Loading**: `test_load_btc_data_from_web_integration()`

  - Tests direct web API calls through backward compatibility functions
  - Validates data format and structure

- **Data Validation**: Multiple validation tests with flexible validation system
  - `test_validate_price_data_valid()`: Valid data handling
  - `test_validate_price_data_missing_column()`: Missing price column detection (flexible - looks for any "Price" columns)
  - `test_validate_price_data_specific_columns()`: Tests specific price column validation with custom requirements
  - `test_validate_price_data_negative_prices()`: Price validation
  - `test_validate_price_data_nan_values()`: NaN handling
  - `test_validate_price_data_empty_dataframe()`: Empty data handling

#### TestFREDLoader

Tests for FRED data loader with API key functionality and comprehensive error handling:

- **API Key Management**: Tests initialization with and without API keys
- **Data Loading**: Tests successful data loading from FRED API
- **Error Handling**: Tests HTTP errors, timeouts, invalid responses
- **Data Validation**: Tests data structure and quality validation
- **File Operations**: Tests CSV and Parquet extraction and loading

#### TestFREDLoaderTimestampAlignment ⭐ **NEW**

Tests for FRED loader timestamp alignment and normalization (addresses timestamp misalignment bug):

- **Midnight UTC Normalization**: `test_fred_timestamps_normalized_to_midnight_utc()`

  - Verifies all FRED timestamps are normalized to exactly midnight UTC (00:00:00+00:00)
  - Ensures no hour/minute/second/microsecond components remain
  - Critical for fixing the original 6-hour timestamp offset issue

- **Cross-Source Consistency**: `test_fred_timestamp_consistency_with_other_sources()`

  - Tests that FRED timestamps match CoinMetrics timestamp format exactly
  - Verifies timestamps can be properly aligned for merging
  - Ensures no timezone conversion artifacts

- **DST and Edge Case Handling**: `test_fred_no_time_zone_conversion_artifacts()`
  - Tests dates affected by Daylight Saving Time transitions
  - Verifies year boundaries and special dates are handled correctly
  - Ensures no timezone conversion creates unexpected time offsets

#### TestDataSourceTimestampConsistency ⭐ **NEW**

Tests for timestamp consistency across all data sources:

- **Unified Midnight UTC**: `test_all_sources_use_midnight_utc()`

  - Verifies all data sources (CoinMetrics, FRED, Fear&Greed) use midnight UTC
  - Tests that timestamp formats are identical across sources
  - Ensures consistent timezone handling

- **Proper Data Merging**: `test_merged_data_has_proper_overlaps()`
  - **Critical test**: Verifies the original timestamp alignment bug is fixed
  - Tests that merged data has actual overlapping records (was 0 before fix)
  - Validates that overlapping data contains correct values from both sources
  - Ensures multi-source strategies can access all indicators

#### TestDataUtilities

Tests utility functions:

- **CSV Extraction**: `test_extract_btc_data_to_csv_integration()`

#### TestDataMocking

Tests with mocked network responses:

- **Mocked Web Loading**: `test_load_btc_data_from_web_mocked()`
  - Uses unittest.mock to simulate API responses via the CoinMetrics loader
  - Tests CSV parsing and data transformation through the new modular system
  - Mocks `stacking_sats_pipeline.data.coinmetrics_loader.requests.get` for proper isolation

### Data Extraction Tests (`test_data_extraction.py`)

#### TestDataExtractionPythonAPI

Tests the `extract_all_data()` Python API functionality:

- **CSV Integration Test**: `test_extract_all_data_csv_integration()`

  - Tests extracting all data sources to CSV format
  - Validates file creation and data integrity
  - Checks for Bitcoin, Fear & Greed, and FRED data files
  - Verifies file content and structure

- **Parquet Integration Test**: `test_extract_all_data_parquet_integration()`

  - Tests extracting all data sources to Parquet format
  - Validates compression efficiency and data preservation
  - Ensures proper datetime indexing is maintained

- **Timestamp Alignment Verification**: `test_extract_all_data_timestamp_alignment_verification()` ⭐ **NEW**

  - **Critical integration test**: Verifies timestamp alignment fix works in CSV extraction
  - Tests that all timestamps in extracted CSV are at midnight (normalized correctly)
  - Validates that merged CSV files have substantial overlapping records between data sources
  - Specifically tests BTC & DXY overlap (the original 0-overlap bug)
  - Ensures the fix works end-to-end in the data extraction pipeline

- **Data Integrity Validation**: `test_extract_all_data_data_integrity()`

  - Validates extracted data maintains original quality
  - Tests price ranges and data type preservation
  - Ensures timezone consistency across formats

- **Error Handling**: `test_extract_all_data_error_handling()`

  - Tests resilience when individual data sources fail
  - Ensures partial extraction continues even with failures
  - Validates error logging and reporting

- **File Size Comparison**: `test_extract_all_data_file_size_comparison()`
  - Compares CSV vs Parquet file sizes
  - Validates compression efficiency (typically 40-60% reduction)
  - Tests storage optimization benefits

#### TestDataExtractionUtilities

Tests utility functions and edge cases:

- **Function Availability**: Tests that `extract_all_data` is properly exported
- **Function Signature**: Validates parameter structure and types
- **Path Handling**: Tests string vs Path object handling
- **Format Validation**: Tests invalid format handling (defaults to CSV)

### CLI Data Extraction Tests (`test_cli.py`)

#### TestCLIDataExtraction

Tests command-line interface for data extraction:

- **CSV Extraction**: `test_cli_extract_data_csv()`

  - Tests `stacking-sats --extract-data csv`
  - Validates proper function calling with correct parameters

- **Parquet Extraction**: `test_cli_extract_data_parquet()`

  - Tests `stacking-sats --extract-data parquet`
  - Ensures format parameter is passed correctly

- **Output Directory Options**:

  - `test_cli_extract_data_with_output_dir()`: Tests `--output-dir` flag
  - `test_cli_extract_data_short_output_dir()`: Tests `-o` short form

- **Integration Testing**: `test_cli_extract_data_integration()`

  - End-to-end test of CLI data extraction with real commands
  - Validates file creation and content in temporary directory
  - Tests timeout handling and error reporting

- **Help Documentation**: `test_cli_extract_data_help_includes_new_options()`

  - Ensures new CLI options are documented in help text
  - Validates format choices are clearly explained

- **Error Handling**: `test_cli_extract_data_invalid_format()`
  - Tests rejection of invalid formats
  - Ensures clear error messages with valid options

#### TestCLIArguments

Tests argument parsing for data extraction:

- **Data Extraction Arguments**: `test_data_extraction_argument_parsing()`
  - Tests `--extract-data {csv,parquet}` choices validation
  - Tests `--output-dir` and `-o` parameter handling
  - Validates argument combinations and defaults

#### TestCLIBasic

Tests basic CLI functionality:

- **Help Command**: `test_cli_help()`

  - Tests `--help` flag functionality
  - Validates help text output

- **Command Availability**: `test_stacking_sats_command()`
  - Tests that `stacking-sats` command is properly installed
  - Validates command registration

#### TestCLIUtilities

Tests CLI utility functions:

- **Function Availability**: `test_extract_all_data_function_availability()`
  - Tests that CLI can access main extraction function
  - Validates function integration

### Configuration Tests (`test_config.py`)

#### TestPackageMetadata

Tests package configuration:

- **Version Information**: `test_version_exists()`

  - Validates package version attribute
  - Ensures version string format

- **Import Validation**: `test_package_imports()`
  - Tests that all exported data engineering functions are importable
  - Validates public API availability

#### TestConfigConstants

Tests configuration constants:

- **Constant Existence**: `test_config_constants_exist()`

  - Validates all required constants are defined
  - Checks data types and reasonable values

- **Date Format Validation**: `test_date_format()`

  - Tests date string parsing
  - Validates date range consistency

- **Purchase Frequency**: `test_purchase_freq_valid()`
  - Validates frequency string format

### End-to-End Data Pipeline Tests (`test_data_pipeline_e2e.py`)

The end-to-end test suite provides comprehensive testing of the entire data pipeline using real API calls and covers the complete data lifecycle including the new Parquet support.

#### TestDataPipelineEndToEnd

Tests the complete data pipeline with real API endpoints:

- **Timestamp Alignment Integration**: `test_timestamp_alignment_integration()` ⭐ **NEW**

  - **Comprehensive integration test**: Verifies the complete timestamp alignment fix with real APIs
  - Tests that both CoinMetrics and FRED data use midnight UTC timestamps
  - Validates that merged real data has substantial overlapping records (>100 days)
  - **Critical validation**: Ensures the original 0-overlap bug is completely resolved
  - Tests with actual API data to verify real-world fix effectiveness
  - Validates overlapping data has reasonable values (price ranges, DXY ranges)

- **Single Source Extraction**: `test_single_source_data_extraction_coinmetrics()`, `test_single_source_data_extraction_fred()`

  - Tests data extraction from CoinMetrics and FRED APIs with real data
  - Validates data structure, quality, and consistency
  - Handles real-world data quality issues (missing values, etc.)

- **Multi-Source Data Merging**: `test_multi_source_data_merging_and_cleaning()`

  - Tests merging data from multiple sources (CoinMetrics + FRED)
  - Validates proper column suffixes and data alignment
  - Tests overlapping date ranges and data consistency

- **Data Cleaning Pipeline**: `test_data_cleaning_and_validation_pipeline()`

  - Tests comprehensive data cleaning with real data samples
  - Validates handling of missing values, negative prices, and infinite values
  - Tests validation functions with problematic data

- **File Caching**: `test_data_pipeline_with_file_caching()`

  - Tests file-based caching functionality
  - Validates that cached data matches fresh API data
  - Tests cache creation and reuse

- **Performance Characteristics**: `test_data_pipeline_performance_characteristics()`

  - Tests loading time and memory usage constraints
  - Validates performance expectations for production use

- **Date Range Filtering**: `test_data_pipeline_date_range_filtering()`

  - Tests filtering data to specific date ranges
  - Validates boundary conditions and data availability

- **API Key Handling**: `test_data_pipeline_with_missing_api_keys()`

  - Tests graceful handling when API keys are missing
  - Validates source availability based on authentication

- **Integration Testing**: `test_real_data_pipeline_integration()`

  - Complete integration test with all available data sources
  - Tests full pipeline from raw API data to clean, merged DataFrames

- **Timezone Validation**: `test_data_timezone_is_utc()`
  - Tests that all DataFrames have UTC timezone
  - Validates CoinMetrics, FRED, and merged data timezone consistency
  - Ensures timezone-aware datetime indexes across all data sources

#### TestDataQualityValidation

Tests comprehensive data quality validation with real data:

- **Comprehensive Validation**: `test_price_data_validation_comprehensive()`

  - Tests validation pipeline with real data samples
  - Validates error detection for various data quality issues
  - Tests edge cases with empty DataFrames and invalid structures

- **Data Consistency**: `test_data_consistency_validation()`
  - Tests consistency validation across multiple data sources
  - Validates date alignment and data range checks
  - Tests cross-source data quality metrics

#### TestDataPipelineMemoryManagement

Tests memory efficiency and resource management:

- **Memory Efficient Loading**: `test_memory_efficient_data_loading()`
  - Tests memory usage during data loading operations
  - Validates memory per record stays within reasonable bounds
  - Tests memory cleanup and garbage collection

#### TestParquetDataPipeline

Tests comprehensive Parquet file support throughout the data pipeline:

- **CoinMetrics Parquet Operations**: `test_coinmetrics_parquet_extraction_and_loading()`

  - Tests extraction to Parquet format
  - Validates Parquet file creation and loading
  - Compares Parquet data with original web data for accuracy

- **FRED Parquet Operations**: `test_fred_parquet_extraction_and_loading()`

  - Tests FRED data extraction and loading in Parquet format
  - Validates data integrity between web and Parquet sources
  - Tests with real FRED API data (requires API key)

- **Multi-Source Parquet Support**: `test_data_loader_parquet_support()`

  - Tests main data loader with Parquet format parameter
  - Validates file format selection functionality
  - Tests Parquet file creation in temporary directories

- **Main Function Parquet Support**: `test_load_data_function_parquet_support()`

  - Tests `load_data()` function with `file_format="parquet"` parameter
  - Validates round-trip consistency (web → parquet → load)
  - Tests file path handling and format selection

- **File Size Efficiency**: `test_parquet_file_size_efficiency()`

  - Compares file sizes between CSV and Parquet formats
  - Validates compression efficiency (Parquet typically 40-60% of CSV size)
  - Tests storage optimization benefits

- **Data Type Preservation**: `test_parquet_data_types_preservation()`
  - Tests that Parquet preserves DatetimeIndex and numeric types
  - Validates schema preservation across save/load cycles
  - Tests data type consistency compared to original data

### Key Features of End-to-End Tests

#### Real API Integration

- Uses actual CoinMetrics and FRED APIs (not mocked)
- Tests with real-world data quality issues
- Validates API authentication and error handling
- Tests network resilience and timeout handling

#### Comprehensive Parquet Support

- Tests all data loaders with Parquet format
- Validates file format conversion accuracy
- Tests performance and storage efficiency
- Validates data type preservation and schema consistency

#### Data Quality Validation

- Tests with real data that may have missing values
- Validates handling of edge cases in historical data
- Tests data cleaning and validation pipelines
- Handles real-world Bitcoin price data quirks

#### Performance and Resource Testing

- Tests memory usage during data operations
- Validates loading time constraints
- Tests file size efficiency
- Validates resource cleanup and management

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_default_fixture_loop_scope = "function"
markers = [
    "integration: marks tests as integration tests (may require network/data)",
]
```

### Test Markers

- `@pytest.mark.integration`: Tests requiring external resources
- Tests without markers: Fast unit tests

## Running Specific Test Categories

### Unit Tests Only

```bash
pytest -m "not integration"
```

### Integration Tests Only

```bash
pytest -m integration
```

### Specific Components

```bash
# Data loading
pytest tests/test_data.py

# Data extraction and export
pytest tests/test_data_extraction.py

# End-to-end data pipeline (includes Parquet tests)
pytest tests/test_data_pipeline_e2e.py

# CLI functionality
pytest tests/test_cli.py

# Configuration
pytest tests/test_config.py
```

### Test Output Options

```bash
# Verbose output
pytest -v

# Very verbose with print statements
pytest -vv -s

# Show test coverage
pytest --cov=stacking_sats_pipeline

# Generate HTML coverage report
pytest --cov=stacking_sats_pipeline --cov-report=html
```

## Test Data and Mocking

### Test Data Creation

Most tests create synthetic Bitcoin price data:

```python
def create_test_data(self, num_days=100):
    """Create test Bitcoin price data."""
    dates = pd.date_range("2020-01-01", periods=num_days, freq="D")
    np.random.seed(42)  # For reproducibility
    prices = 30000 + np.cumsum(np.random.normal(0, 500, num_days))
    prices = np.maximum(prices, 1000)  # Ensure positive prices
    return pd.DataFrame({"PriceUSD": prices}, index=dates)
```

### Mocking Strategy

Tests use `unittest.mock` to avoid external dependencies:

- **Data Loading**: Mock CoinMetrics API responses
- **File Operations**: Mock file system interactions

## Continuous Integration

The test suite is designed to work in CI environments:

- **Python Version Support**: Tests run on Python 3.11 and 3.12
- **Network Independence**: Unit tests don't require internet
- **Integration Test Isolation**: Marked separately for optional execution

## Troubleshooting

### Common Issues

1. **Integration Test Failures**

   - May indicate network connectivity issues
   - CoinMetrics API availability
   - Skip with: `pytest -m "not integration"`

2. **Data Extraction Test Failures**
   - May indicate network connectivity or API issues
   - Check FRED_API_KEY environment variable for FRED tests
   - Verify file permissions for output directory tests

### Test Development

When adding new tests:

1. **Follow Naming Convention**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use Appropriate Markers**: Mark integration tests with `@pytest.mark.integration`
3. **Mock External Dependencies**: Use `unittest.mock` for external services
4. **Create Synthetic Data**: Use reproducible random seeds
5. **Test Edge Cases**: Include tests for boundary conditions and error scenarios
6. **Test Both Formats**: For data extraction, test both CSV and Parquet formats
7. **Validate File Creation**: Ensure tests clean up temporary files

## Test Coverage

The test suite aims for comprehensive coverage of:

- ✅ Data loading and validation
- ✅ Data extraction and export (CSV/Parquet)
- ✅ Command-line interface including data extraction commands
- ✅ End-to-end data pipeline with real API integration
- ✅ Parquet file format support throughout the pipeline
- ✅ CLI argument parsing and execution
- ✅ Configuration and metadata
- ✅ Error handling and edge cases
- ✅ File format conversion and efficiency
- ✅ Data type preservation and schema validation
- ✅ Multi-format data export functionality
- ✅ Timestamp alignment and normalization
- ✅ Multi-source data merging and cleaning

Run coverage analysis:

```bash
pytest --cov=stacking_sats_pipeline --cov-report=term-missing
```
