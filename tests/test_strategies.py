"""Unit tests for QuantVortex trading strategies.

Tests cover:
- Statistical arbitrage (StatArb): pair-signal shape and half-life.
- Momentum factor: score neutrality and position shape.
- Mean reversion (MeanReversion): OU parameter estimation and half-life.
- ML alpha (MLAlpha): feature engineering and signal shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.stat_arb import StatArb
from strategies.momentum_factor import MomentumFactor
from strategies.mean_reversion import MeanReversion
from strategies.ml_alpha import MLAlpha


# ---------------------------------------------------------------------------
# Shared price fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def gbm_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Geometric Brownian Motion prices for 5 correlated assets.

    Returns
    -------
    pd.DataFrame
        Shape (300, 5) price DataFrame with DatetimeIndex.
    """
    n, n_assets = 300, 5
    dates = pd.bdate_range("2020-01-02", periods=n)
    mu = 0.0002
    sigma = 0.015
    corr = np.array(
        [
            [1.00, 0.80, 0.10, 0.10, 0.05],
            [0.80, 1.00, 0.10, 0.10, 0.05],
            [0.10, 0.10, 1.00, 0.70, 0.05],
            [0.10, 0.10, 0.70, 1.00, 0.05],
            [0.05, 0.05, 0.05, 0.05, 1.00],
        ]
    )
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, n_assets)) @ L.T
    log_ret = mu - 0.5 * sigma ** 2 + sigma * z
    price_mat = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(
        price_mat,
        index=dates,
        columns=[f"ASSET{i+1}" for i in range(n_assets)],
    )


@pytest.fixture(scope="module")
def single_price_series(rng: np.random.Generator) -> pd.DataFrame:
    """Single-asset price series with a 'close' column.

    Returns
    -------
    pd.DataFrame
        Shape (400, 1) with column ``"close"`` and DatetimeIndex.
    """
    n = 400
    dates = pd.bdate_range("2020-01-02", periods=n)
    log_ret = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture(scope="module")
def mean_reverting_spread(rng: np.random.Generator) -> pd.Series:
    """Ornstein-Uhlenbeck mean-reverting spread (100 obs).

    Returns
    -------
    pd.Series
        OU path with mu=0, kappa=0.5, sigma=0.5.
    """
    n = 100
    dates = pd.bdate_range("2020-01-02", periods=n)
    mu, kappa, sigma, dt = 0.0, 0.5, 0.5, 1.0
    x = np.zeros(n)
    x[0] = 0.0
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * eps[t]
    return pd.Series(x, index=dates, name="spread")


# ---------------------------------------------------------------------------
# StatArb tests
# ---------------------------------------------------------------------------


class TestStatArb:
    """Tests for statistical arbitrage strategy."""

    def test_stat_arb_signals_shape(self, gbm_prices: pd.DataFrame) -> None:
        """generate_signals must return a DataFrame with rows matching input.

        The first two assets are highly correlated, so at least one pair
        should be found.
        """
        strat = StatArb(
            entry_z=2.0,
            exit_z=0.5,
            coint_pvalue=0.10,
            zscore_window=30,
        )
        signals = strat.generate_signals(gbm_prices)
        assert isinstance(signals, pd.DataFrame), "generate_signals must return DataFrame"
        assert len(signals) == len(gbm_prices), (
            f"Signal rows {len(signals)} != price rows {len(gbm_prices)}"
        )

    def test_stat_arb_half_life_positive(self, gbm_prices: pd.DataFrame) -> None:
        """compute_half_life must return a positive scalar for a real spread."""
        strat = StatArb()
        # Build a cointegrated spread manually from correlated assets
        spread = gbm_prices["ASSET1"] - 0.95 * gbm_prices["ASSET2"]
        half_life = StatArb.compute_half_life(spread)
        assert half_life > 0, f"Half-life {half_life} is not positive"

    def test_stat_arb_positions_binary(self, gbm_prices: pd.DataFrame) -> None:
        """Position values must be in {-1, 0, +1} after discretisation."""
        strat = StatArb(
            entry_z=2.0,
            exit_z=0.5,
            coint_pvalue=0.10,
            zscore_window=30,
        )
        signals = strat.generate_signals(gbm_prices)
        if signals.empty:
            pytest.skip("No cointegrated pairs found with test data.")
        positions = strat.compute_positions(signals)
        unique_vals = set(np.unique(positions.values.round(6)))
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected position values: {unique_vals}"
        )


# ---------------------------------------------------------------------------
# MomentumFactor tests
# ---------------------------------------------------------------------------


class TestMomentumFactor:
    """Tests for momentum factor strategy."""

    def test_momentum_scores_sum_to_zero(self, gbm_prices: pd.DataFrame) -> None:
        """Long/short portfolio weights should sum to approximately 0.

        Each date: sum of weights = 1/long_n * long_n - 1/short_n * short_n = 0.
        """
        strat = MomentumFactor(lookback=60, skip_last=5, long_n=2, short_n=2)
        signals = strat.generate_signals(gbm_prices)
        positions = strat.compute_positions(signals)
        # Check rows that are non-zero
        non_zero_rows = positions[(positions != 0).any(axis=1)]
        if non_zero_rows.empty:
            pytest.skip("No non-zero position rows generated.")
        row_sums = non_zero_rows.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values, np.zeros(len(row_sums)), atol=1e-9,
            err_msg="Long/short momentum weights do not sum to 0"
        )

    def test_momentum_positions_shape(self, gbm_prices: pd.DataFrame) -> None:
        """compute_positions must return DataFrame with same shape as signals."""
        strat = MomentumFactor(lookback=60, skip_last=5, long_n=2, short_n=2)
        signals = strat.generate_signals(gbm_prices)
        positions = strat.compute_positions(signals)
        assert positions.shape == signals.shape, (
            f"Positions shape {positions.shape} != signals shape {signals.shape}"
        )

    def test_momentum_scores_finite(self, gbm_prices: pd.DataFrame) -> None:
        """All non-NaN momentum scores must be finite."""
        strat = MomentumFactor(lookback=60, skip_last=5)
        signals = strat.generate_signals(gbm_prices)
        non_nan = signals.values[~np.isnan(signals.values)]
        assert np.all(np.isfinite(non_nan)), "Non-finite values in momentum scores"


# ---------------------------------------------------------------------------
# MeanReversion tests
# ---------------------------------------------------------------------------


class TestMeanReversion:
    """Tests for mean-reversion (OU) strategy."""

    def test_mean_reversion_ou_params(
        self, mean_reverting_spread: pd.Series
    ) -> None:
        """estimate_ou_params must return dict with 'mu', 'kappa', 'sigma_ou'."""
        strat = MeanReversion()
        params = strat.estimate_ou_params(mean_reverting_spread)
        assert isinstance(params, dict), "estimate_ou_params must return a dict"
        for key in ("mu", "kappa", "sigma_ou"):
            assert key in params, f"Key '{key}' missing from OU params"

    def test_mean_reversion_half_life(
        self, mean_reverting_spread: pd.Series
    ) -> None:
        """kappa > 0 must yield a positive half-life."""
        strat = MeanReversion()
        params = strat.estimate_ou_params(mean_reverting_spread)
        kappa = params["kappa"]
        assert kappa > 0, f"kappa={kappa} should be positive for a mean-reverting series"
        half_life = MeanReversion.compute_half_life(kappa)
        assert half_life > 0, f"Half-life {half_life} is not positive for kappa={kappa}"

    def test_mean_reversion_kappa_positive(
        self, mean_reverting_spread: pd.Series
    ) -> None:
        """For a truly mean-reverting spread, kappa should be positive."""
        strat = MeanReversion()
        params = strat.estimate_ou_params(mean_reverting_spread)
        assert params["kappa"] > 0, "kappa is not positive for mean-reverting spread"

    def test_mean_reversion_half_life_formula(self) -> None:
        """compute_half_life(kappa) == ln(2) / kappa."""
        kappa = 0.5
        expected = np.log(2) / kappa
        result = MeanReversion.compute_half_life(kappa)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# MLAlpha tests
# ---------------------------------------------------------------------------


class TestMLAlpha:
    """Tests for ML alpha strategy."""

    _EXPECTED_FEATURES = {
        "ret_1d", "ret_5d", "ret_21d", "ret_63d",
        "vol_21d", "vol_63d", "rsi_14", "momentum_105d",
    }

    def test_ml_alpha_features(self, single_price_series: pd.DataFrame) -> None:
        """engineer_features must return DataFrame with expected feature columns."""
        strat = MLAlpha(lstm_epochs=1)
        features = strat.engineer_features(single_price_series)
        assert isinstance(features, pd.DataFrame), "engineer_features must return DataFrame"
        missing = self._EXPECTED_FEATURES - set(features.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_ml_alpha_features_no_nan(
        self, single_price_series: pd.DataFrame
    ) -> None:
        """Feature matrix returned by engineer_features must not contain NaN."""
        strat = MLAlpha(lstm_epochs=1)
        features = strat.engineer_features(single_price_series)
        assert not features.isnull().any().any(), (
            "engineer_features returned NaN values (dropna should eliminate them)"
        )

    def test_ml_alpha_signals_shape(
        self, single_price_series: pd.DataFrame
    ) -> None:
        """generate_signals must return a non-empty DataFrame with a 'pred_return' column."""
        strat = MLAlpha(lstm_epochs=1)
        signals = strat.generate_signals(single_price_series)
        assert isinstance(signals, pd.DataFrame), "generate_signals must return DataFrame"
        assert not signals.empty, "generate_signals returned empty DataFrame"
        assert "pred_return" in signals.columns, (
            "generate_signals must include 'pred_return' column"
        )

    def test_ml_alpha_signals_finite(
        self, single_price_series: pd.DataFrame
    ) -> None:
        """All signal values must be finite numbers."""
        strat = MLAlpha(lstm_epochs=1)
        signals = strat.generate_signals(single_price_series)
        assert np.all(np.isfinite(signals.values)), (
            "generate_signals contains non-finite values"
        )
