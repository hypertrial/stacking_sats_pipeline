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
└── strategy/
    └── strategy_template.py   # Reference strategy implementation
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

## Command Options

| Flag | Description |
|------|-------------|
| `--strategy` | Path to strategy Python file |
| `--no-plot` | Skip plot generation |
| `--simulate` | Run accumulation simulation |
| `--budget` | Annual budget for simulation (default: $10M) |