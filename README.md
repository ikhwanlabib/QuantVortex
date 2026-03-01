# QuantVortex

A production-grade quantitative trading engine combining statistical arbitrage, momentum, mean reversion, ML alpha (XGBoost + LSTM), volatility surface arbitrage, and reinforcement learning (PPO) strategies.

## Features

- **6 Trading Strategies**: StatArb, MomentumFactor, MeanReversion, VolSurfaceArb, MLAlpha (XGBoost + LSTM), RLTrader (PPO)
- **Event-Driven Backtesting Engine** with realistic transaction cost modelling
- **Monte Carlo Simulation** for risk scenario analysis
- **Performance Analytics**: Sharpe ratio, max drawdown, VaR, and more
- **Interactive Dash Dashboard** for visualising results
- **YAML-Based Configuration** — no code changes needed to adjust parameters

## Project Structure

```
QuantVortex/
├── strategies/       # 6 trading strategy implementations
├── backtesting/      # Event-driven engine, performance analytics, Monte Carlo simulation
├── config/           # YAML configuration files (settings.yaml, strategies.yaml)
├── quantvortex/      # CLI entry point
├── portfolio/        # Portfolio management
├── models/           # ML model definitions
├── data/             # Data loading and caching
├── visualization/    # Interactive Dash dashboard
└── tests/            # Test suite
```

## Prerequisites

- Python >= 3.10
- pip

## Installation (Quick Start)

1. **Clone the repository**

   ```bash
   git clone https://github.com/ikhwanlabib/QuantVortex.git
   cd QuantVortex
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - Windows:
     ```powershell
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the package with development dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Verify the installation**

   ```bash
   python -c "import quantvortex; print(quantvortex.__version__)"
   ```

## Troubleshooting

If `pip install -e .` fails and you get `ModuleNotFoundError: No module named 'quantvortex'`, set `PYTHONPATH` manually:

- **Windows (PowerShell)**:
  ```powershell
  $env:PYTHONPATH = "."
  python -m quantvortex.cli
  ```
- **macOS / Linux**:
  ```bash
  PYTHONPATH=. python -m quantvortex.cli
  ```

Alternatively, install in non-editable mode:

```bash
pip install ".[dev]"
```

## Usage

```bash
# Show help
quantvortex

# Run backtesting
quantvortex backtest --config config/settings.yaml

# Launch the interactive dashboard
quantvortex dashboard --port 8050 --debug
```

You can also invoke via the module directly:

```bash
python -m quantvortex.cli backtest --config config/settings.yaml
```

## Configuration

| File | Description |
|------|-------------|
| `config/settings.yaml` | Data universe (20 tickers), backtesting parameters (initial capital $10 M, commission 5 bps, slippage 10 bps), risk limits (max position 10 %, max drawdown 15 %, 99 % VaR) |
| `config/strategies.yaml` | Strategy allocations — StatArb 25 %, Momentum 20 %, MeanReversion 15 %, MLAlpha 25 %, RLTrader 15 % |

## Strategies

| Strategy | Description |
|----------|-------------|
| **StatArb** | Statistical arbitrage using cointegration and pairs trading |
| **MomentumFactor** | Cross-sectional momentum factor model |
| **MeanReversion** | Mean-reversion signals with Ornstein–Uhlenbeck process |
| **VolSurfaceArb** | Volatility surface arbitrage across options strikes/expiries |
| **MLAlpha** | Machine-learning alpha combining XGBoost and LSTM predictions |
| **RLTrader** | Reinforcement learning trader trained with PPO via Stable-Baselines3 |

## Running Tests

```bash
pytest tests/
```

## Tech Stack / Dependencies

| Category | Libraries |
|----------|-----------|
| Numerics | numpy, pandas, scipy |
| Machine Learning | scikit-learn, xgboost, torch |
| Reinforcement Learning | gymnasium, stable-baselines3 |
| Explainability | shap |
| Finance / Statistics | yfinance, statsmodels, arch, hmmlearn |
| Visualisation | matplotlib, plotly, dash |
| Configuration | pyyaml |

## License

MIT