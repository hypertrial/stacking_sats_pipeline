# Stacking Sats Pipeline

A Bitcoin DCA strategy backtesting pipeline that validates and tests strategies against historical price data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline with default strategy
python main.py

# Test your custom strategy
python main.py --strategy path/to/your_strategy.py
```

## Pipeline Commands

### Basic Backtesting
```bash
# Run full pipeline with validation and plots
python main.py --strategy strategy/strategy_template.py

# Skip visualization for faster execution
python main.py --strategy your_strategy.py --no-plot

# Run with custom cycle length
python main.py --strategy your_strategy.py --cycle-years 2
```

### Simulation Mode
```bash
# Run accumulation simulation
python main.py --strategy your_strategy.py --simulate

# Custom budget simulation
python main.py --strategy your_strategy.py --simulate --budget 1000000
```

### Weight Allocation Calculator
```bash
# Calculate daily allocations for a specific period and budget
python -m weights.weight_calculator 1000 2024-01-01 2024-01-31

# Save allocation data to CSV
python -m weights.weight_calculator 5000 2024-03-01 2024-03-31 --save

# Custom filename for CSV export
python -m weights.weight_calculator 1000 2024-06-01 2024-06-30 --save --filename my_allocation.csv
```

## Pipeline Architecture

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Configuration constants
├── backtest/
│   ├── checks.py             # Strategy validation & SPD calculation
│   └── simulation.py         # Bitcoin accumulation simulation
├── data/
│   └── data_loader.py        # BTC price data pipeline
├── plot/
│   └── plotting.py           # Visualization pipeline
├── strategy/
│   └── strategy_template.py   # Reference strategy implementation
└── weights/
    └── weight_calculator.py   # Weight allocation calculator
```

## Strategy Requirements

Your strategy file must implement:

```python
from config import BACKTEST_START, BACKTEST_END, CYCLE_YEARS

def compute_weights(df: pd.DataFrame, *, cycle_years: int = CYCLE_YEARS) -> pd.Series:
    """Return daily investment weights that sum to 1.0 per cycle."""
    # Your strategy logic here
    return weights_series
```

**Pipeline Validation Rules:**
- Weights must sum to 1.0 within each cycle
- All weights must be positive (≥ 1e-5)
- No forward-looking data usage
- Must return pandas Series indexed by date

## Pipeline Output

The pipeline generates:
- **Validation Report**: Strategy constraint compliance
- **Performance Metrics**: SPD (Sats Per Dollar) statistics per cycle
- **Comparative Analysis**: vs Uniform DCA and Static DCA (30th percentile)
- **Visualizations**: Weight distribution and feature plots (unless `--no-plot`)

### Example Output
```
============================================================
COMPREHENSIVE STRATEGY VALIDATION
============================================================
✅ ALL VALIDATION CHECKS PASSED

Aggregated Metrics for Your Strategy:
Dynamic SPD:
  mean: 4510.21
  median: 2804.03
Dynamic SPD Percentile:
  mean: 39.35%
  median: 43.80%

Mean Excess vs Uniform DCA: -0.40%
Mean Excess vs Static DCA: 9.35%
```

## Weight Allocation Calculator

The weights module provides a standalone calculator for determining daily Bitcoin allocation amounts based on your strategy and a specific budget.

### Features
- **Flexible Date Ranges**: Calculate allocations for any period within the available data
- **Budget-Based Allocation**: Specify total USD budget to allocate across the period
- **Detailed Breakdown**: Shows daily weights, USD amounts, BTC prices, and BTC quantities
- **CSV Export**: Save allocation data for further analysis or implementation

### Example Output
```
=== Allocation Weights from 2024-01-01 to 2024-01-31 ===
Total budget: $1,000.00
Total days: 31
Total weight: 0.0847
Average daily weight: 0.0027
Average daily allocation: $32.26

Daily Breakdown:
Date         Weight   Weight %   USD Amount   BTC Price    BTC Amount  
2024-01-01   0.0027   0.27      % $2.73        $44049.47    0.000062    
2024-01-02   0.0027   0.27      % $2.73        $44941.16    0.000061    
...
TOTAL        0.0847   8.47      % $1000.00                  0.001978
```

### Weight Calculator Options

| Argument | Description |
|----------|-------------|
| `budget` | Total USD budget to allocate across the period |
| `start_date` | Start date in YYYY-MM-DD format |
| `end_date` | End date in YYYY-MM-DD format |
| `--save, -s` | Save allocation data to CSV file |
| `--filename, -f` | Custom CSV filename (optional) |

## Command Options

| Flag | Description |
|------|-------------|
| `--strategy` | Path to strategy Python file |
| `--no-plot` | Skip plot generation |
| `--simulate` | Run accumulation simulation |
| `--budget` | Annual budget for simulation (default: $10M) |