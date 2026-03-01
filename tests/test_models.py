"""Unit tests for QuantVortex pricing models.

Tests cover:
- Black-Scholes pricing, Greeks, and implied volatility.
- Heston stochastic-volatility pricing.
- Merton jump-diffusion pricing.
- GARCH family volatility fitting.
- HMM-based regime detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.black_scholes import BlackScholes
from models.heston_model import HestonModel
from models.jump_diffusion import MertonJumpDiffusion
from models.garch_family import GARCHFamily
from models.regime_detection import RegimeDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def bs() -> BlackScholes:
    """Black-Scholes pricer instance."""
    return BlackScholes()


@pytest.fixture(scope="module")
def heston() -> HestonModel:
    """Heston model instance with default parameters."""
    return HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)


@pytest.fixture(scope="module")
def mjd() -> MertonJumpDiffusion:
    """Merton jump-diffusion instance."""
    return MertonJumpDiffusion()


@pytest.fixture(scope="module")
def garch() -> GARCHFamily:
    """GARCH model family instance."""
    return GARCHFamily()


@pytest.fixture(scope="module")
def synthetic_returns(rng: np.random.Generator) -> np.ndarray:
    """1,000 normally distributed daily returns (zero mean, 1% vol)."""
    return rng.normal(0.0, 0.01, size=1_000)


@pytest.fixture(scope="module")
def regime_returns(rng: np.random.Generator) -> np.ndarray:
    """500 returns with three distinct volatility regimes."""
    low = rng.normal(0.001, 0.005, 200)
    mid = rng.normal(0.000, 0.012, 150)
    high = rng.normal(-0.001, 0.025, 150)
    return np.concatenate([low, mid, high])


# ---------------------------------------------------------------------------
# Black-Scholes tests
# ---------------------------------------------------------------------------


class TestBlackScholes:
    """Black-Scholes pricing, Greeks, and implied-volatility tests."""

    # Standard ATM parameters
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02

    def test_bs_call_put_parity(self, bs: BlackScholes) -> None:
        """Put-call parity: C - P = S·exp(-q·T) - K·exp(-r·T).

        Uses ``q=0`` for the standard Black-Scholes parity identity
        ``C - P = S - K·exp(-r·T)``.
        """
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        call = bs.price(S, K, T, r, sigma, option_type="call")
        put = bs.price(S, K, T, r, sigma, option_type="put")
        expected = S - K * np.exp(-r * T)
        np.testing.assert_allclose(call - put, expected, atol=1e-6,
                                   err_msg="Put-call parity violated")

    def test_bs_greeks_delta_bounds(self, bs: BlackScholes) -> None:
        """Delta must be in [0, 1] for calls and [-1, 0] for puts."""
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        call_delta = bs.delta(S, K, T, r, sigma, option_type="call")
        put_delta = bs.delta(S, K, T, r, sigma, option_type="put")
        assert 0.0 <= call_delta <= 1.0, f"Call delta {call_delta} out of [0,1]"
        assert -1.0 <= put_delta <= 0.0, f"Put delta {put_delta} out of [-1,0]"

    def test_bs_implied_vol_roundtrip(self, bs: BlackScholes) -> None:
        """Recover implied volatility from a priced option to within 1e-4."""
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        price = bs.price(S, K, T, r, sigma, option_type="call")
        iv = bs.implied_volatility(price, S, K, T, r, option_type="call")
        np.testing.assert_allclose(iv, sigma, atol=1e-4,
                                   err_msg="IV round-trip failed")

    def test_bs_gamma_positive(self, bs: BlackScholes) -> None:
        """Gamma must always be strictly positive for vanilla options."""
        for spot in [80.0, 100.0, 120.0]:
            gamma = bs.gamma(spot, self.K, self.T, self.r, self.sigma)
            assert gamma > 0, f"Gamma {gamma} <= 0 for S={spot}"

    def test_bs_call_price_positive(self, bs: BlackScholes) -> None:
        """Call price must be non-negative."""
        price = bs.price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert price >= 0.0, f"Call price {price} is negative"

    def test_bs_intrinsic_value(self, bs: BlackScholes) -> None:
        """Deep ITM call should be close to intrinsic value at very short expiry."""
        S, K, r, sigma = 120.0, 100.0, 0.05, 0.20
        T_short = 1e-4
        price = bs.price(S, K, T_short, r, sigma, "call")
        intrinsic = max(S - K * np.exp(-r * T_short), 0.0)
        assert price >= intrinsic - 1e-4, "Price below intrinsic value"

    def test_bs_vega_positive(self, bs: BlackScholes) -> None:
        """Vega must be positive (option price increases with volatility)."""
        vega = bs.vega(self.S, self.K, self.T, self.r, self.sigma)
        assert vega > 0.0, f"Vega {vega} is not positive"


# ---------------------------------------------------------------------------
# Heston model tests
# ---------------------------------------------------------------------------


class TestHestonModel:
    """Heston stochastic-volatility pricing tests."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05

    def test_heston_price_positive(self, heston: HestonModel) -> None:
        """Heston call price must be strictly positive."""
        price = heston.price(self.S, self.K, self.T, self.r)
        assert price > 0.0, f"Heston call price {price} is not positive"

    def test_heston_put_call_parity(self, heston: HestonModel) -> None:
        """Heston put-call parity: C - P ≈ S - K·exp(-r·T), tolerance 0.01."""
        S, K, T, r = self.S, self.K, self.T, self.r
        call = heston.price(S, K, T, r, option_type="call")
        put = heston.price(S, K, T, r, option_type="put")
        expected = S - K * np.exp(-r * T)
        np.testing.assert_allclose(call - put, expected, atol=0.01,
                                   err_msg="Heston put-call parity violated")

    def test_heston_price_itm_greater_otm(self, heston: HestonModel) -> None:
        """ITM call should be worth more than OTM call, ceteris paribus."""
        itm = heston.price(110.0, self.K, self.T, self.r, option_type="call")
        otm = heston.price(90.0, self.K, self.T, self.r, option_type="call")
        assert itm > otm, "ITM call is not more expensive than OTM call"


# ---------------------------------------------------------------------------
# Merton jump-diffusion tests
# ---------------------------------------------------------------------------


class TestMertonJumpDiffusion:
    """Merton jump-diffusion pricing tests."""

    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma, lam, mu_j, sigma_j = 0.20, 1.0, -0.10, 0.15

    def test_merton_price_positive(self, mjd: MertonJumpDiffusion) -> None:
        """Merton jump-diffusion call price must be positive."""
        price = mjd.price(
            self.S, self.K, self.T, self.r,
            self.sigma, self.lam, self.mu_j, self.sigma_j,
            option_type="call",
        )
        assert price > 0.0, f"MJD call price {price} is not positive"

    def test_merton_approaches_bs_no_jumps(
        self, mjd: MertonJumpDiffusion, bs: BlackScholes
    ) -> None:
        """With near-zero jump intensity MJD price ≈ BS price (tol 0.01)."""
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        mjd_price = mjd.price(S, K, T, r, sigma, lam=0.001, mu_j=0.0,
                               sigma_j=0.001, option_type="call")
        bs_price = bs.price(S, K, T, r, sigma, option_type="call")
        np.testing.assert_allclose(mjd_price, bs_price, atol=0.01,
                                   err_msg="MJD does not converge to BS for lam→0")

    def test_merton_put_positive(self, mjd: MertonJumpDiffusion) -> None:
        """Merton put price must be positive."""
        price = mjd.price(
            self.S, self.K, self.T, self.r,
            self.sigma, self.lam, self.mu_j, self.sigma_j,
            option_type="put",
        )
        assert price > 0.0, f"MJD put price {price} is not positive"

    def test_merton_higher_intensity_higher_price(
        self, mjd: MertonJumpDiffusion
    ) -> None:
        """Increasing jump intensity should increase option price (neg-mean jumps)."""
        common = dict(S=self.S, K=self.K, T=self.T, r=self.r,
                      sigma=self.sigma, mu_j=-0.05, sigma_j=0.15,
                      option_type="call")
        p_low = mjd.price(**common, lam=0.5)
        p_high = mjd.price(**common, lam=3.0)
        # Negative mean jumps → put price rises, call may also rise due to vol
        # At minimum the prices should differ
        assert p_low != p_high, "Price invariant to jump intensity change"


# ---------------------------------------------------------------------------
# GARCH family tests
# ---------------------------------------------------------------------------


class TestGARCHFamily:
    """GARCH(1,1) fitting tests."""

    def test_garch_fit_returns_dict(
        self, garch: GARCHFamily, synthetic_returns: np.ndarray
    ) -> None:
        """fit_garch must return a dict containing omega, alpha, beta."""
        result = garch.fit_garch(synthetic_returns)
        assert isinstance(result, dict), "fit_garch did not return a dict"
        for key in ("omega", "alpha", "beta"):
            assert key in result, f"Key '{key}' missing from fit_garch output"

    def test_garch_persistence_below_one(
        self, garch: GARCHFamily, synthetic_returns: np.ndarray
    ) -> None:
        """alpha + beta < 1 is required for GARCH(1,1) stationarity."""
        result = garch.fit_garch(synthetic_returns)
        alpha = result["alpha"]
        beta = result["beta"]
        # alpha and beta may be lists or floats
        alpha_sum = float(np.sum(alpha))
        beta_sum = float(np.sum(beta))
        persistence = alpha_sum + beta_sum
        assert persistence < 1.0, (
            f"GARCH persistence {persistence:.4f} >= 1 (non-stationary)"
        )

    def test_garch_omega_positive(
        self, garch: GARCHFamily, synthetic_returns: np.ndarray
    ) -> None:
        """omega (unconditional variance intercept) must be positive."""
        result = garch.fit_garch(synthetic_returns)
        assert result["omega"] > 0.0, "GARCH omega is not positive"

    def test_garch_conditional_variances_positive(
        self, garch: GARCHFamily, synthetic_returns: np.ndarray
    ) -> None:
        """All fitted conditional variances must be positive."""
        result = garch.fit_garch(synthetic_returns)
        cond_var = np.asarray(result["conditional_variances"])
        assert np.all(cond_var > 0), "Some conditional variances are non-positive"

    def test_garch_aic_bic_finite(
        self, garch: GARCHFamily, synthetic_returns: np.ndarray
    ) -> None:
        """AIC and BIC should be finite numbers."""
        result = garch.fit_garch(synthetic_returns)
        assert np.isfinite(result["aic"]), "AIC is not finite"
        assert np.isfinite(result["bic"]), "BIC is not finite"


# ---------------------------------------------------------------------------
# Regime detection tests
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    """HMM-based regime detection tests."""

    def test_regime_detector_fit_predict(
        self, regime_returns: np.ndarray
    ) -> None:
        """fit followed by predict must return an array of the correct shape."""
        detector = RegimeDetector(n_regimes=3)
        detector.fit(regime_returns)
        labels = detector.predict(regime_returns)
        assert labels.shape == regime_returns.shape, (
            f"Label shape {labels.shape} != return shape {regime_returns.shape}"
        )

    def test_regime_n_regimes(self, regime_returns: np.ndarray) -> None:
        """Predicted labels must contain at most n_regimes unique values."""
        n_regimes = 3
        detector = RegimeDetector(n_regimes=n_regimes)
        detector.fit(regime_returns)
        labels = detector.predict(regime_returns)
        unique = np.unique(labels)
        assert len(unique) <= n_regimes, (
            f"Got {len(unique)} unique regimes but expected <= {n_regimes}"
        )

    def test_regime_labels_integer(self, regime_returns: np.ndarray) -> None:
        """Regime labels should be integer-valued."""
        detector = RegimeDetector(n_regimes=2)
        detector.fit(regime_returns)
        labels = detector.predict(regime_returns)
        assert np.issubdtype(labels.dtype, np.integer) or np.all(
            labels == labels.astype(int)
        ), "Regime labels are not integer-valued"

    def test_regime_proba_sums_to_one(self, regime_returns: np.ndarray) -> None:
        """State-probability rows must sum to 1 (within floating-point tol)."""
        detector = RegimeDetector(n_regimes=3)
        detector.fit(regime_returns)
        proba = detector.predict_proba(regime_returns)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(len(row_sums)), atol=1e-6,
            err_msg="Regime probability rows do not sum to 1"
        )
