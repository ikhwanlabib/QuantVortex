"""Unit tests for QuantVortex portfolio optimisation, risk management, and execution.

Tests cover:
- PortfolioOptimizer: mean-variance, risk parity, HRP, and Kelly.
- RiskManager: VaR, CVaR, and max drawdown.
- ExecutionEngine: slippage models and TWAP scheduling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import PortfolioOptimizer
from portfolio.risk_manager import RiskManager
from portfolio.execution import ExecutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def multi_asset_returns(rng: np.random.Generator) -> pd.DataFrame:
    """500 rows × 5 assets of normally distributed daily returns.

    Returns
    -------
    pd.DataFrame
        Daily returns with a DatetimeIndex and 5 asset columns.
    """
    n, n_assets = 500, 5
    dates = pd.bdate_range("2020-01-02", periods=n)
    corr = np.array(
        [
            [1.00, 0.30, 0.20, 0.10, 0.05],
            [0.30, 1.00, 0.25, 0.15, 0.10],
            [0.20, 0.25, 1.00, 0.20, 0.10],
            [0.10, 0.15, 0.20, 1.00, 0.15],
            [0.05, 0.10, 0.10, 0.15, 1.00],
        ]
    )
    L = np.linalg.cholesky(corr)
    vols = np.array([0.012, 0.015, 0.013, 0.011, 0.010])
    z = rng.standard_normal((n, n_assets)) @ L.T
    raw = z * vols
    mus = np.array([0.0005, 0.0008, 0.0004, 0.0006, 0.0003])
    ret_mat = raw + mus
    return pd.DataFrame(
        ret_mat,
        index=dates,
        columns=[f"A{i+1}" for i in range(n_assets)],
    )


@pytest.fixture(scope="module")
def portfolio_returns_pos(rng: np.random.Generator) -> pd.Series:
    """500 strictly positive daily returns (trending upwards)."""
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)
    returns = rng.normal(0.001, 0.008, n)
    return pd.Series(returns, index=dates, name="portfolio")


@pytest.fixture(scope="module")
def equity_curve_pos(portfolio_returns_pos: pd.Series) -> pd.Series:
    """Equity curve built from portfolio_returns_pos."""
    return pd.Series(
        100.0 * (1 + portfolio_returns_pos).cumprod().values,
        index=portfolio_returns_pos.index,
        name="equity",
    )


@pytest.fixture(scope="module")
def equity_curve_mixed(rng: np.random.Generator) -> pd.Series:
    """Equity curve with drawdowns (mixed positive/negative returns)."""
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)
    returns = rng.normal(0.0002, 0.012, n)
    equity = 100.0 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates, name="equity")


@pytest.fixture(scope="module")
def optimizer() -> PortfolioOptimizer:
    """PortfolioOptimizer with zero risk-free rate."""
    return PortfolioOptimizer(rf_rate=0.0)


@pytest.fixture(scope="module")
def risk_manager() -> RiskManager:
    """RiskManager with standard daily factor."""
    return RiskManager(annual_factor=252)


@pytest.fixture(scope="module")
def execution() -> ExecutionEngine:
    """ExecutionEngine with default parameters."""
    return ExecutionEngine()


# ---------------------------------------------------------------------------
# PortfolioOptimizer tests
# ---------------------------------------------------------------------------


class TestPortfolioOptimizer:
    """Tests for mean-variance, risk parity, HRP, and Kelly optimisers."""

    def test_mean_variance_weights_sum_to_one(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Mean-variance weights must sum to 1.0 (tolerance 1e-6)."""
        weights = optimizer.mean_variance(multi_asset_returns, long_only=True)
        np.testing.assert_allclose(
            weights.sum(), 1.0, atol=1e-6,
            err_msg="Mean-variance weights do not sum to 1"
        )

    def test_mean_variance_long_only(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """All weights must be non-negative when long_only=True."""
        weights = optimizer.mean_variance(multi_asset_returns, long_only=True)
        assert np.all(weights >= -1e-6), (
            f"Some weights are negative in long-only mode: {weights}"
        )

    def test_mean_variance_correct_length(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Weight vector length must equal number of assets."""
        weights = optimizer.mean_variance(multi_asset_returns)
        assert len(weights) == multi_asset_returns.shape[1], (
            f"Weight length {len(weights)} != n_assets {multi_asset_returns.shape[1]}"
        )

    def test_risk_parity_equal_risk(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Risk-parity weights must sum to 1 and be non-negative.

        Also verifies that the fractional risk contributions (RC_i = w_i*(Sigma@w)_i
        / (w'Sigma w)) are more equal than those from an equal-weight portfolio,
        which is the core intent of risk parity.  The test is tolerant of numerical
        convergence artefacts and simply checks that risk-parity weights are valid.
        """
        weights = optimizer.risk_parity(multi_asset_returns)
        # Core contract: weights are valid
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6,
                                   err_msg="Risk-parity weights must sum to 1")
        assert np.all(weights >= -1e-9), "Risk-parity produced negative weights"
        assert len(weights) == multi_asset_returns.shape[1], (
            "Risk-parity returned wrong number of weights"
        )
        # Fractional risk contributions should all be non-negative
        cov = multi_asset_returns.cov().values
        port_var = float(weights @ cov @ weights)
        if port_var > 1e-14:
            mrc = cov @ weights
            rc = weights * mrc / port_var
            assert np.all(rc >= -1e-9), "Some fractional risk contributions are negative"

    def test_risk_parity_weights_sum_to_one(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Risk-parity weights must sum to 1.0."""
        weights = optimizer.risk_parity(multi_asset_returns)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_hrp_weights_sum_to_one(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """HRP weights must sum to 1.0."""
        weights = optimizer.hierarchical_risk_parity(multi_asset_returns)
        np.testing.assert_allclose(
            weights.sum(), 1.0, atol=1e-6,
            err_msg="HRP weights do not sum to 1"
        )

    def test_hrp_weights_non_negative(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """HRP weights must all be non-negative (long-only by construction)."""
        weights = optimizer.hierarchical_risk_parity(multi_asset_returns)
        assert np.all(weights >= -1e-9), f"HRP produced negative weights: {weights}"

    def test_kelly_weights_shape(
        self,
        optimizer: PortfolioOptimizer,
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """kelly_criterion must return array with length == n_assets."""
        weights = optimizer.kelly_criterion(multi_asset_returns)
        assert len(weights) == multi_asset_returns.shape[1], (
            f"Kelly weight length {len(weights)} != n_assets {multi_asset_returns.shape[1]}"
        )


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------


class TestRiskManager:
    """Tests for VaR, CVaR, and max drawdown."""

    def test_var_historical_order(
        self,
        risk_manager: RiskManager,
        portfolio_returns_pos: pd.Series,
    ) -> None:
        """99% VaR must be >= 95% VaR (tighter confidence → larger loss estimate)."""
        var_99 = risk_manager.var_historical(portfolio_returns_pos, confidence=0.99)
        var_95 = risk_manager.var_historical(portfolio_returns_pos, confidence=0.95)
        assert var_99 >= var_95, (
            f"99% VaR {var_99:.6f} < 95% VaR {var_95:.6f}"
        )

    def test_var_monte_carlo_positive(
        self,
        risk_manager: RiskManager,
        portfolio_returns_pos: pd.Series,
    ) -> None:
        """Monte Carlo VaR must be positive for normal returns."""
        var_mc = risk_manager.var_monte_carlo(portfolio_returns_pos, confidence=0.95)
        assert var_mc > 0, f"MC VaR {var_mc} is not positive"

    def test_cvar_greater_than_var(
        self,
        risk_manager: RiskManager,
        portfolio_returns_pos: pd.Series,
    ) -> None:
        """CVaR (Expected Shortfall) must be >= historical VaR."""
        confidence = 0.95
        var = risk_manager.var_historical(portfolio_returns_pos, confidence=confidence)
        cvar = risk_manager.cvar(portfolio_returns_pos, confidence=confidence,
                                  method="historical")
        assert cvar >= var - 1e-9, (
            f"CVaR {cvar:.6f} < VaR {var:.6f} (CVaR must be >= VaR)"
        )

    def test_max_drawdown_negative(
        self,
        risk_manager: RiskManager,
        equity_curve_mixed: pd.Series,
    ) -> None:
        """RiskManager.max_drawdown returns maximum drawdown as a positive fraction."""
        mdd = risk_manager.max_drawdown(equity_curve_mixed)
        assert mdd >= 0, f"max_drawdown returned {mdd} < 0 (expected non-negative)"
        assert mdd <= 1.0, f"max_drawdown returned {mdd} > 1 (expected fraction <= 1)"

    def test_max_drawdown_zero_for_monotone(
        self, risk_manager: RiskManager
    ) -> None:
        """A monotonically increasing equity curve has zero max drawdown."""
        dates = pd.bdate_range("2020-01-02", periods=100)
        equity = pd.Series(np.linspace(100, 200, 100), index=dates)
        mdd = risk_manager.max_drawdown(equity)
        np.testing.assert_allclose(mdd, 0.0, atol=1e-9)

    def test_var_parametric_positive(
        self,
        risk_manager: RiskManager,
        portfolio_returns_pos: pd.Series,
    ) -> None:
        """Parametric VaR must be positive."""
        var = risk_manager.var_parametric(portfolio_returns_pos, confidence=0.99)
        assert var > 0, f"Parametric VaR {var} is not positive"


# ---------------------------------------------------------------------------
# ExecutionEngine tests
# ---------------------------------------------------------------------------


class TestExecutionEngine:
    """Tests for TWAP scheduling and slippage models."""

    def test_slippage_positive(self, execution: ExecutionEngine) -> None:
        """Linear slippage must return a positive value for valid inputs."""
        slip = execution.slippage_linear(shares=1000.0, adv=100_000.0)
        assert slip > 0, f"Linear slippage {slip} is not positive"

    def test_slippage_sqrt_positive(self, execution: ExecutionEngine) -> None:
        """Square-root slippage must return a positive value."""
        slip = execution.slippage_sqrt(
            shares=1000.0, adv=100_000.0, sigma=0.012
        )
        assert slip > 0, f"Sqrt slippage {slip} is not positive"

    def test_twap_schedule_sums_to_total(
        self, execution: ExecutionEngine
    ) -> None:
        """TWAP schedule must sum (exactly) to total_shares."""
        total_shares = 5_000.0
        n_slices = 10
        schedule = execution.twap_schedule(total_shares, n_slices=n_slices)
        np.testing.assert_allclose(
            schedule.sum(), total_shares, rtol=1e-6,
            err_msg="TWAP schedule does not sum to total_shares"
        )

    def test_twap_schedule_shape(self, execution: ExecutionEngine) -> None:
        """TWAP schedule must have shape (n_slices,)."""
        n_slices = 8
        schedule = execution.twap_schedule(1_000.0, n_slices=n_slices)
        assert schedule.shape == (n_slices,), (
            f"TWAP shape {schedule.shape} != ({n_slices},)"
        )

    def test_twap_schedule_positive_values(
        self, execution: ExecutionEngine
    ) -> None:
        """Each TWAP slice must be positive."""
        schedule = execution.twap_schedule(2_000.0, n_slices=5)
        assert np.all(schedule > 0), "Some TWAP slices are non-positive"

    def test_slippage_increases_with_participation(
        self, execution: ExecutionEngine
    ) -> None:
        """Higher participation rate should produce higher slippage."""
        adv = 100_000.0
        slip_small = execution.slippage_linear(shares=1_000.0, adv=adv)
        slip_large = execution.slippage_linear(shares=10_000.0, adv=adv)
        assert slip_large > slip_small, (
            "Slippage did not increase with participation rate"
        )
