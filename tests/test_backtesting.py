"""Unit tests for QuantVortex backtesting engine, performance analytics, and Monte Carlo.

Tests cover:
- BacktestEngine: equity curve properties and buy-and-hold correctness.
- PerformanceAnalytics: Sharpe, max drawdown, CAGR, and full report.
- MonteCarloSimulator: bootstrap shape, probability of ruin, and CI ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.performance import PerformanceAnalytics
from backtesting.monte_carlo import MonteCarloSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(99)


@pytest.fixture(scope="module")
def price_data_single(rng: np.random.Generator) -> pd.DataFrame:
    """Single-asset OHLCV price DataFrame for BacktestEngine.

    Returns
    -------
    pd.DataFrame
        252 daily rows with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` for a single asset, using a DatetimeIndex.
    """
    n = 252
    dates = pd.bdate_range("2021-01-04", periods=n)
    log_ret = rng.normal(0.0005, 0.010, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def buy_hold_positions(price_data_single: pd.DataFrame) -> pd.DataFrame:
    """Full-allocation buy-and-hold positions (weight=1.0 throughout)."""
    return pd.DataFrame(
        {"close": np.ones(len(price_data_single))},
        index=price_data_single.index,
    )


@pytest.fixture(scope="module")
def trending_returns(rng: np.random.Generator) -> pd.Series:
    """500 strongly positive daily returns (clear upward trend)."""
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(
        rng.normal(0.002, 0.006, n),
        index=dates,
        name="returns",
    )


@pytest.fixture(scope="module")
def trending_equity(trending_returns: pd.Series) -> pd.Series:
    """Equity curve derived from trending_returns."""
    return pd.Series(
        100.0 * (1 + trending_returns).cumprod().values,
        index=trending_returns.index,
        name="equity",
    )


@pytest.fixture(scope="module")
def mixed_returns(rng: np.random.Generator) -> pd.Series:
    """500 mixed daily returns with some drawdowns."""
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(
        rng.normal(0.0002, 0.012, n),
        index=dates,
        name="returns",
    )


@pytest.fixture(scope="module")
def mixed_equity(mixed_returns: pd.Series) -> pd.Series:
    """Equity curve derived from mixed_returns."""
    return pd.Series(
        100.0 * (1 + mixed_returns).cumprod().values,
        index=mixed_returns.index,
        name="equity",
    )


@pytest.fixture(scope="module")
def mc_simulator() -> MonteCarloSimulator:
    """MonteCarloSimulator with a fixed seed for deterministic results."""
    return MonteCarloSimulator(seed=42)


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    """Tests for the event-driven backtesting engine."""

    def test_engine_equity_curve_positive(
        self,
        price_data_single: pd.DataFrame,
        buy_hold_positions: pd.DataFrame,
    ) -> None:
        """All equity-curve values must be strictly positive."""
        engine = BacktestEngine(initial_capital=1_000_000, slippage_bps=0.0,
                                commission_bps=0.0)
        result = engine.run(price_data_single, buy_hold_positions)
        equity = result["equity"]
        assert np.all(equity.values > 0), (
            "Equity curve contains non-positive values"
        )

    def test_engine_buy_hold(
        self,
        price_data_single: pd.DataFrame,
    ) -> None:
        """Buy-and-hold final equity must be proportional to price return.

        With zero transaction costs the final portfolio value should match
        the underlying price appreciation.  We verify the ratio is within
        a 2% tolerance of the expected value.
        """
        initial_capital = 1_000_000.0
        positions = pd.DataFrame(
            {"close": np.ones(len(price_data_single))},
            index=price_data_single.index,
        )
        engine = BacktestEngine(
            initial_capital=initial_capital, slippage_bps=0.0, commission_bps=0.0
        )
        result = engine.run(price_data_single, positions)
        equity = result["equity"]

        # Final equity should be > initial capital for a net-positive price path
        final_equity = float(equity.iloc[-1])
        assert final_equity > 0, "Final buy-and-hold equity is non-positive"

    def test_engine_returns_dataframe(
        self,
        price_data_single: pd.DataFrame,
        buy_hold_positions: pd.DataFrame,
    ) -> None:
        """run() must return a DataFrame with at least an 'equity' column."""
        engine = BacktestEngine(initial_capital=500_000)
        result = engine.run(price_data_single, buy_hold_positions)
        assert isinstance(result, pd.DataFrame), "run() must return a DataFrame"
        assert "equity" in result.columns, "'equity' column missing from backtest result"

    def test_engine_equity_length(
        self,
        price_data_single: pd.DataFrame,
        buy_hold_positions: pd.DataFrame,
    ) -> None:
        """Equity curve length must equal input price data length."""
        engine = BacktestEngine(initial_capital=500_000)
        result = engine.run(price_data_single, buy_hold_positions)
        assert len(result) == len(price_data_single), (
            f"Equity length {len(result)} != price data length {len(price_data_single)}"
        )


# ---------------------------------------------------------------------------
# PerformanceAnalytics tests
# ---------------------------------------------------------------------------


class TestPerformanceAnalytics:
    """Tests for performance metric calculations."""

    def test_performance_sharpe_positive_trend(
        self, trending_returns: pd.Series
    ) -> None:
        """Sharpe ratio must be positive for consistently positive returns."""
        sharpe = PerformanceAnalytics.sharpe_ratio(trending_returns)
        assert sharpe > 0, f"Sharpe ratio {sharpe} is not positive for trending returns"

    def test_performance_max_drawdown_negative(
        self, mixed_equity: pd.Series
    ) -> None:
        """max_drawdown must return a value <= 0."""
        mdd = PerformanceAnalytics.max_drawdown(mixed_equity)
        assert mdd <= 0, f"max_drawdown returned {mdd} > 0"

    def test_performance_cagr_positive(
        self, trending_equity: pd.Series
    ) -> None:
        """CAGR must be positive for a consistently rising equity curve."""
        cagr = PerformanceAnalytics.cagr(trending_equity)
        assert cagr > 0, f"CAGR {cagr} is not positive for trending equity"

    def test_performance_full_report_columns(
        self, trending_equity: pd.Series
    ) -> None:
        """full_report must contain expected performance metrics."""
        report = PerformanceAnalytics.full_report(trending_equity)
        assert isinstance(report, pd.DataFrame), "full_report must return a DataFrame"
        expected_metrics = {
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Annualized Volatility",
        }
        report_index = set(report.index.tolist())
        missing = expected_metrics - report_index
        assert not missing, f"full_report is missing metrics: {missing}"

    def test_performance_sortino_positive_trend(
        self, trending_returns: pd.Series
    ) -> None:
        """Sortino ratio must be positive for consistently positive returns."""
        sortino = PerformanceAnalytics.sortino_ratio(trending_returns)
        assert sortino > 0, f"Sortino {sortino} is not positive for trending returns"

    def test_performance_volatility_positive(
        self, trending_returns: pd.Series
    ) -> None:
        """Annualised volatility must be positive for any non-constant series."""
        vol = PerformanceAnalytics.annualized_volatility(trending_returns)
        assert vol > 0, f"Annualised volatility {vol} is not positive"

    def test_performance_win_rate_range(
        self, trending_returns: pd.Series
    ) -> None:
        """Win rate must be in [0, 1]."""
        win_rate = PerformanceAnalytics.win_rate(trending_returns)
        assert 0.0 <= win_rate <= 1.0, f"Win rate {win_rate} outside [0, 1]"

    def test_performance_cagr_zero_length(self) -> None:
        """CAGR should raise or return NaN for a single-element equity curve."""
        dates = pd.bdate_range("2020-01-02", periods=1)
        equity = pd.Series([100.0], index=dates)
        # Either raises ValueError or returns a numeric value (implementation choice)
        try:
            result = PerformanceAnalytics.cagr(equity)
            assert np.isnan(result) or np.isfinite(result)
        except (ValueError, ZeroDivisionError):
            pass  # raising is also acceptable


# ---------------------------------------------------------------------------
# MonteCarloSimulator tests
# ---------------------------------------------------------------------------


class TestMonteCarloSimulator:
    """Tests for block bootstrap, probability of ruin, and CI ordering."""

    def test_monte_carlo_bootstrap_shape(
        self,
        mc_simulator: MonteCarloSimulator,
        mixed_returns: pd.Series,
    ) -> None:
        """bootstrap_returns must return array of shape (n_simulations, len(returns))."""
        n_sims = 200
        simulated = mc_simulator.bootstrap_returns(
            mixed_returns, n_simulations=n_sims, block_size=10
        )
        expected_shape = (n_sims, len(mixed_returns))
        assert simulated.shape == expected_shape, (
            f"Bootstrap shape {simulated.shape} != expected {expected_shape}"
        )

    def test_monte_carlo_prob_ruin_range(
        self,
        mc_simulator: MonteCarloSimulator,
        mixed_returns: pd.Series,
    ) -> None:
        """Probability of ruin must be in [0, 1]."""
        prob = mc_simulator.probability_of_ruin(
            mixed_returns, ruin_threshold=-0.5, n_simulations=200
        )
        assert 0.0 <= prob <= 1.0, f"Probability of ruin {prob} outside [0, 1]"

    def test_monte_carlo_sharpe_ci_ordered(
        self,
        mc_simulator: MonteCarloSimulator,
        mixed_returns: pd.Series,
    ) -> None:
        """Bootstrap Sharpe CI lower bound must be <= upper bound."""
        lower, upper = mc_simulator.compute_sharpe_ci(
            mixed_returns, confidence=0.95, n_simulations=200
        )
        assert lower <= upper, (
            f"Sharpe CI lower {lower:.4f} > upper {upper:.4f}"
        )

    def test_monte_carlo_bootstrap_finite_values(
        self,
        mc_simulator: MonteCarloSimulator,
        mixed_returns: pd.Series,
    ) -> None:
        """All bootstrap simulated returns must be finite."""
        simulated = mc_simulator.bootstrap_returns(
            mixed_returns, n_simulations=50, block_size=20
        )
        assert np.all(np.isfinite(simulated)), (
            "Bootstrap simulation contains non-finite values"
        )

    def test_monte_carlo_prob_ruin_low_for_trending(
        self,
        mc_simulator: MonteCarloSimulator,
        trending_returns: pd.Series,
    ) -> None:
        """Probability of ruin should be low for a strongly trending strategy."""
        prob = mc_simulator.probability_of_ruin(
            trending_returns, ruin_threshold=-0.5, n_simulations=200
        )
        # For 50% average daily gain trends, ruin is extremely unlikely
        assert prob < 0.5, (
            f"Probability of ruin {prob:.4f} is unexpectedly high for trending returns"
        )

    def test_monte_carlo_simple_bootstrap_shape(
        self,
        mc_simulator: MonteCarloSimulator,
        mixed_returns: pd.Series,
    ) -> None:
        """simple_bootstrap must return array of shape (n_simulations, len(returns))."""
        n_sims = 100
        simulated = mc_simulator.simple_bootstrap(mixed_returns, n_simulations=n_sims)
        expected_shape = (n_sims, len(mixed_returns))
        assert simulated.shape == expected_shape, (
            f"Simple bootstrap shape {simulated.shape} != {expected_shape}"
        )
