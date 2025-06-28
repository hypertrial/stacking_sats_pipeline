# Testing Documentation

This document provides comprehensive information about the test suite for the Stacking Sats Pipeline project.

## Requirements

- Python 3.11 or 3.12
- pytest and development dependencies

## Overview

The test suite is built using pytest and covers all major components of the pipeline:

- **Backtesting functionality** (`test_backtest.py`)
- **Data loading and validation** (`test_data.py`) 
- **End-to-end data pipeline including Parquet support** (`test_data_pipeline_e2e.py`)
- **Command-line interface** (`test_cli.py`)
- **Configuration and metadata** (`test_config.py`)
- **Plotting and visualization** (`test_plot.py`)
- **Strategy computation** (`test_strategy.py`)
- **Weight calculation and historical data constraints** (`test_weight_calculator.py`)

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
pytest tests/test_backtest.py

# Run specific test class
pytest tests/test_backtest.py::TestBacktestingCore

# Run specific test method
pytest tests/test_backtest.py::TestBacktestingCore::test_quick_backtest_integration
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

| File | Purpose | Key Features |
|------|---------|--------------|
| `test_backtest.py` | Backtesting functionality | Strategy validation, performance metrics, result analysis |
| `test_data.py` | Data loading and validation | CoinMetrics integration, CSV handling, data validation |
| `test_data_pipeline_e2e.py` | End-to-end data pipeline | Real API testing, Parquet support, file format conversion, data quality validation |
| `test_cli.py` | Command-line interface | Argument parsing, strategy loading, error handling |
| `test_config.py` | Configuration and metadata | Package metadata, constants validation |
| `test_plot.py` | Plotting and visualization | Chart generation, matplotlib integration |
| `test_strategy.py` | Strategy computation | Weight calculation, feature construction |
| `test_weight_calculator.py` | Weight calculator functionality | Historical data constraints, date validation, CSV export |

## Detailed Test Documentation

### Backtesting Tests (`test_backtest.py`)

#### TestBacktestingCore
Tests core backtesting functionality including:

- **Quick Backtesting**: `test_quick_backtest_integration()`
  - Tests the `quick_backtest()` function with real data
  - Validates performance metric calculation
  - Marked as integration test due to data loading

- **Strategy Decorator**: `test_strategy_decorator_basic()`
  - Tests the `@strategy` decorator functionality
  - Validates metadata attachment and function preservation
  - Ensures decorated strategies work correctly

- **Mocked Backtesting**: `test_backtest_mocked()`
  - Tests `backtest()` function with mocked data
  - Uses unittest.mock to avoid external dependencies
  - Validates BacktestResults object creation

#### TestBacktestResults
Tests BacktestResults class functionality:

- **Object Creation**: `test_backtest_results_creation()`
- **Method Validation**: `test_backtest_results_methods()`

#### TestStrategyValidation
Tests strategy validation logic:

- **Valid Strategy Validation**: `test_check_strategy_submission_ready_valid()`
  - Tests strategies that should pass all validation checks
  - Ensures proper weight normalization and constraints

- **Invalid Strategy Detection**: `test_check_strategy_submission_ready_invalid()`
  - Tests strategies with issues (negative weights, etc.)
  - Validates error detection and reporting

#### TestLegacyBacktesting
Tests legacy backtesting functions:

- **Cycle SPD Computation**: `test_compute_cycle_spd_basic()`
- **Dynamic DCA Backtesting**: `test_backtest_dynamic_dca_basic()`

#### TestStrategyPatterns
Tests common strategy patterns:

- **Constant Strategy**: `test_constant_strategy()`
- **Varying Strategy**: `test_varying_strategy()`

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

#### TestDataUtilities
Tests utility functions:

- **CSV Extraction**: `test_extract_btc_data_to_csv_integration()`

#### TestDataMocking
Tests with mocked network responses:

- **Mocked Web Loading**: `test_load_btc_data_from_web_mocked()`
  - Uses unittest.mock to simulate API responses via the CoinMetrics loader
  - Tests CSV parsing and data transformation through the new modular system
  - Mocks `stacking_sats_pipeline.data.coinmetrics_loader.requests.get` for proper isolation

### CLI Tests (`test_cli.py`)

#### TestCLIBasic
Tests basic CLI functionality:

- **Help Command**: `test_cli_help()`
  - Tests `--help` flag functionality
  - Validates help text output

- **Command Availability**: `test_stacking_sats_command()`
  - Tests that `stacking-sats` command is properly installed
  - Validates command registration

#### TestCLIArguments
Tests argument parsing:

- **No-Plot Argument**: `test_no_plot_argument_parsing()`
- **Strategy Argument**: `test_argument_parsing_strategy()`
- Tests default values and custom argument handling

#### TestCLIStrategyLoading
Tests strategy file loading:

- **Valid Strategy Loading**: `test_load_strategy_from_file_valid()`
  - Creates temporary strategy files for testing
  - Validates strategy function extraction
  - Tests strategy execution

- **Invalid Strategy Handling**: `test_load_strategy_from_file_invalid()`
  - Tests error handling for malformed strategy files
  - Validates proper error reporting

- **Nonexistent File Handling**: `test_load_strategy_nonexistent_file()`

#### TestCLIErrorHandling
Tests error scenarios:

- **Invalid Strategy Files**: `test_cli_with_invalid_strategy_file()`

#### TestCLIFunctionality
Tests CLI function signatures and availability:

- **Main Function**: `test_main_function_exists()`
- **Function Signatures**: `test_main_function_signature()`

### Configuration Tests (`test_config.py`)

#### TestPackageMetadata
Tests package configuration:

- **Version Information**: `test_version_exists()`
  - Validates package version attribute
  - Ensures version string format

- **Import Validation**: `test_package_imports()`
  - Tests that all exported functions are importable
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

### Plotting Tests (`test_plot.py`)

#### TestPlottingFunctions
Tests plotting functionality with mocked matplotlib:

- **Feature Plotting**: `test_plot_features_basic()`
  - Tests `plot_features()` function
  - Validates matplotlib integration
  - Tests with and without weights

- **Weight Plotting**: `test_plot_final_weights()`
  - Tests weight distribution visualization
  - Validates chart generation

- **Cycle Analysis**: `test_plot_weight_sums_by_cycle()`
  - Tests cycle-based weight analysis charts

- **SPD Comparison**: `test_plot_spd_comparison()`
  - Tests performance comparison charts
  - Validates multi-strategy comparison

#### TestPlottingInputValidation
Tests plotting with various input scenarios:

- **Minimal Data**: `test_plot_functions_with_minimal_data()`
- **Edge Case Dates**: `test_plot_functions_with_edge_case_dates()`

#### TestPlottingErrorHandling
Tests error handling in plotting functions:

- **Invalid Data**: `test_plot_features_invalid_data()`
- **Invalid Weights**: `test_plot_final_weights_invalid_weights()`

### Strategy Tests (`test_strategy.py`)

#### TestStrategyFunctions
Tests strategy computation functions:

- **Feature Construction**: `test_construct_features_basic()`
  - Tests `construct_features()` function
  - Validates feature engineering pipeline
  - Tests with various data lengths

- **Weight Computation**: `test_compute_weights_basic()`
  - Tests `compute_weights()` function
  - Validates weight constraints and properties
  - Tests mathematical properties

- **Edge Cases**: Multiple edge case tests
  - Short data handling
  - Extreme values
  - Missing data scenarios

#### TestStrategyValidation
Tests strategy validation logic:

- **Function Signatures**: `test_strategy_function_signature()`
- **Mathematical Properties**: `test_weights_mathematical_properties()`
  - Non-negativity
  - Finiteness
  - Reasonable magnitude
  - Variation checks

#### TestCustomStrategy
Tests custom strategy patterns:

- **Uniform Strategy**: `test_uniform_strategy()`
  - Tests equal-weight strategy implementation
  - Validates weight normalization

- **Momentum Strategy**: `test_momentum_strategy()`
  - Tests trend-following strategy
  - Validates price change calculations

### Weight Calculator Tests (`test_weight_calculator.py`)

#### TestDateValidation
Tests date range validation for historical data limits:

- **Valid Date Ranges**: `test_validate_date_range_valid_dates()`
  - Tests `validate_date_range()` with dates within available data
  - Validates multiple valid date range scenarios

- **Start Date Before Data**: `test_validate_date_range_start_before_data()`
  - Tests rejection of start dates before historical data begins
  - Validates proper error messages

- **End Date After Data**: `test_validate_date_range_end_after_data()`
  - Tests rejection of end dates after historical data ends
  - Ensures future dates are properly rejected

- **Both Dates Outside Data**: `test_validate_date_range_both_outside_data()`
  - Tests scenarios where both dates are outside available range

- **Edge Cases**: `test_validate_date_range_edge_cases()`
  - Tests exact boundary dates
  - Single day requests at data boundaries

#### TestHistoricalDataLoading
Tests historical data loading with validation:

- **Mocked Data Loading**: `test_get_historical_btc_data_mocked()`
  - Tests `get_historical_btc_data()` with mocked data
  - Validates data structure and loading process

- **Valid Period Loading**: `test_get_historical_data_for_period_valid()`
  - Tests loading data for valid date ranges
  - Validates data filtering and validation pipeline

- **Invalid Period Handling**: `test_get_historical_data_for_period_invalid()`
  - Tests proper error handling for invalid date ranges

#### TestWeightCalculation
Tests weight calculation functions with historical data constraints:

- **Valid Range Calculation**: `test_get_weights_for_period_valid_range()`
  - Tests `get_weights_for_period()` with valid dates
  - Validates weight calculation pipeline

- **Invalid Range Rejection**: `test_get_weights_for_period_invalid_range()`
  - Tests proper error handling for invalid date ranges
  - Ensures historical constraints are enforced

- **Display Function Constraints**: `test_display_weights_historical_constraint()`
  - Tests that `display_weights()` respects historical limits

#### TestHistoricalDataConstraints
Tests enforcement of historical data constraints across functions:

- **Future Date Rejection**: `test_future_dates_rejected()`
  - Tests that future dates are properly rejected
  - Validates error messages for out-of-range dates

- **Very Old Date Rejection**: `test_very_old_dates_rejected()`
  - Tests rejection of dates before Bitcoin existed
  - Validates historical data boundaries

- **CSV Export Constraints**: `test_csv_export_historical_constraint()`
  - Tests that CSV export respects historical data limits
  - Validates file creation and parameter passing

#### TestErrorHandling
Tests error handling in weight calculator functions:

- **Malformed Date Strings**: `test_malformed_date_strings()`
  - Tests handling of invalid date formats
  - Ensures system doesn't crash on bad input

- **Reversed Date Range**: `test_reversed_date_range()`
  - Tests handling when start_date > end_date

- **Empty DataFrame Handling**: `test_empty_dataframe_handling()`
  - Tests graceful handling of empty data scenarios

#### TestIntegrationWithRealConstraints
Integration tests with actual data constraints:

- **Real Data Constraints**: `test_real_data_constraints()` (integration test)
  - Tests with actual historical data loading
  - Validates real data boundaries and constraints
  - Tests boundary conditions with actual data ranges

- **Current Date Boundary**: `test_current_date_boundary()` (integration test)
  - Tests that current/future dates are properly handled
  - Validates real-time constraint enforcement

### End-to-End Data Pipeline Tests (`test_data_pipeline_e2e.py`)

The end-to-end test suite provides comprehensive testing of the entire data pipeline using real API calls and covers the complete data lifecycle including the new Parquet support.

#### TestDataPipelineEndToEnd
Tests the complete data pipeline with real API endpoints:

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

#### TestBacktestParquetExport
Tests Parquet export functionality in backtest results and weight calculation:

- **Backtest Results Parquet Export**: `test_backtest_results_parquet_export()`
  - Tests `save_weights_to_parquet()` method in BacktestResults
  - Validates Parquet file structure and content
  - Compares CSV vs Parquet export consistency
  - Tests generic `save_weights()` method with format selection

- **Weight Calculator Parquet Export**: `test_weight_calculator_parquet_export()`
  - Tests Parquet export functionality in weight calculator module
  - Validates `save_weights_to_parquet()` and `save_weights()` functions
  - Tests file creation and data integrity

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
# Backtest functionality
pytest tests/test_backtest.py

# Data loading
pytest tests/test_data.py

# End-to-end data pipeline (includes Parquet tests)
pytest tests/test_data_pipeline_e2e.py

# CLI functionality  
pytest tests/test_cli.py

# Plotting (with mocked matplotlib)
pytest tests/test_plot.py

# Strategy computation
pytest tests/test_strategy.py

# Weight calculator functionality
pytest tests/test_weight_calculator.py

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
- **Plotting**: Mock matplotlib to avoid GUI dependencies
- **File Operations**: Mock file system interactions

## Continuous Integration

The test suite is designed to work in CI environments:

- **Python Version Support**: Tests run on Python 3.11 and 3.12
- **No GUI Dependencies**: Plotting tests mock matplotlib
- **Network Independence**: Unit tests don't require internet
- **Integration Test Isolation**: Marked separately for optional execution

## Troubleshooting

### Common Issues

1. **Integration Test Failures**
   - May indicate network connectivity issues
   - CoinMetrics API availability
   - Skip with: `pytest -m "not integration"`

2. **Plotting Test Failures**
   - Usually indicates matplotlib import issues
   - Tests are mocked to avoid GUI requirements

3. **Strategy Test Failures**
   - Check that strategy functions return proper pandas Series
   - Ensure weights sum to 1.0 and are non-negative

### Test Development

When adding new tests:

1. **Follow Naming Convention**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use Appropriate Markers**: Mark integration tests with `@pytest.mark.integration`
3. **Mock External Dependencies**: Use `unittest.mock` for external services
4. **Create Synthetic Data**: Use reproducible random seeds
5. **Test Edge Cases**: Include tests for boundary conditions and error scenarios

## Test Coverage

The test suite aims for comprehensive coverage of:

- ✅ Core backtesting functionality
- ✅ Data loading and validation
- ✅ End-to-end data pipeline with real API integration
- ✅ Parquet file format support throughout the pipeline
- ✅ CLI argument parsing and execution
- ✅ Strategy computation and validation
- ✅ Weight calculator and historical data constraints
- ✅ Plotting and visualization
- ✅ Configuration and metadata
- ✅ Error handling and edge cases
- ✅ File format conversion and efficiency
- ✅ Data type preservation and schema validation

Run coverage analysis:
```bash
pytest --cov=stacking_sats_pipeline --cov-report=term-missing
``` 