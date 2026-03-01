"""Black-Scholes option pricing model with Greeks and implied volatility.

This module provides a complete implementation of the Black-Scholes framework for
European option pricing, including all first- and second-order Greeks and a
Newton-Raphson implied volatility solver.
"""

import logging
import math
from typing import Union

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

_MIN_T = 1e-9
_MIN_SIGMA = 1e-9
_MAX_ITER_IV = 200
_IV_TOL = 1e-8


class BlackScholes:
    """Black-Scholes European option pricing and Greeks calculator.

    Implements the closed-form Black-Scholes formulas for European call and put
    options, all standard first- and second-order Greeks (delta, gamma, theta,
    vega, rho, vanna, volga, charm), and a Newton-Raphson implied volatility
    solver with bisection fallback.

    Examples
    --------
    >>> bs = BlackScholes()
    >>> price = bs.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> print(f"Call price: {price:.4f}")
    Call price: 10.4506
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _d1_d2(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> tuple[float, float]:
        """Compute d1 and d2 for the Black-Scholes formula.

        Parameters
        ----------
        S : float
            Current underlying price (> 0).
        K : float
            Strike price (> 0).
        T : float
            Time to expiry in years (> 0).
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility (> 0).

        Returns
        -------
        tuple[float, float]
            ``(d1, d2)`` scalars.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Compute the Black-Scholes price of a European option.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised implied volatility.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Option fair value.

        Raises
        ------
        ValueError
            If ``S``, ``K``, or ``sigma`` are non-positive, or ``option_type``
            is unrecognised.
        """
        if S <= 0 or K <= 0:
            raise ValueError(f"S and K must be positive, got S={S}, K={K}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got sigma={sigma}")
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        T = max(T, _MIN_T)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        discount = math.exp(-r * T)

        if option_type == "call":
            value = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        else:
            value = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

        logger.debug(
            "price(%s, S=%.4f, K=%.4f, T=%.4f, r=%.4f, sigma=%.4f) = %.6f",
            option_type, S, K, T, r, sigma, value,
        )
        return value

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """First-order sensitivity to underlying price (dV/dS).

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Delta of the option.
        """
        option_type = option_type.lower()
        T = max(T, _MIN_T)
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1.0
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Second-order sensitivity to underlying price (d²V/dS²).

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.

        Returns
        -------
        float
            Gamma of the option (same for call and put).
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    def theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Time decay per calendar day (dV/dt, expressed as daily decay).

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Theta (value change per calendar day, typically negative).
        """
        option_type = option_type.lower()
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        discount = math.exp(-r * T)
        common = -(S * norm.pdf(d1) * sigma) / (2.0 * sqrt_T)

        if option_type == "call":
            theta_annual = common - r * K * discount * norm.cdf(d2)
        elif option_type == "put":
            theta_annual = common + r * K * discount * norm.cdf(-d2)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        return theta_annual / 365.0

    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Sensitivity to a 1-unit (100%) change in volatility (dV/dσ).

        Conventionally returned per 1% move in vol; divide by 100 for raw.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.

        Returns
        -------
        float
            Vega per 1% move in volatility.
        """
        T = max(T, _MIN_T)
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) * 0.01

    def rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Sensitivity to a 1-unit change in the risk-free rate (dV/dr).

        Conventionally returned per 1% (100 bps) move in rates.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Rho per 1% move in interest rates.
        """
        option_type = option_type.lower()
        T = max(T, _MIN_T)
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        discount = math.exp(-r * T)

        if option_type == "call":
            return K * T * discount * norm.cdf(d2) * 0.01
        elif option_type == "put":
            return -K * T * discount * norm.cdf(-d2) * 0.01
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    def vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Cross-sensitivity dDelta/dVol (also d²V/dS dσ).

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.

        Returns
        -------
        float
            Vanna of the option.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        return -norm.pdf(d1) * d2 / sigma

    def volga(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Second-order sensitivity to volatility (d²V/dσ²), also called vomma.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.

        Returns
        -------
        float
            Volga (vomma) of the option.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        # vega * d1 * d2 / sigma
        raw_vega = S * norm.pdf(d1) * sqrt_T
        return raw_vega * d1 * d2 / sigma

    def charm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Delta decay rate dDelta/dTime (also called delta bleed).

        Returns the daily rate of change of delta with respect to time.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Charm per calendar day.
        """
        option_type = option_type.lower()
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)

        inner = 2.0 * r * T - d2 * sigma * sqrt_T
        annual_charm = -norm.pdf(d1) * inner / (2.0 * T * sigma * sqrt_T)

        if option_type == "put":
            annual_charm -= r * norm.cdf(-d1)
        elif option_type == "call":
            annual_charm += r * norm.cdf(d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        return annual_charm / 365.0

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        initial_guess: float = 0.2,
    ) -> float:
        """Solve for implied volatility using Newton-Raphson with bisection fallback.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.
        initial_guess : float, optional
            Starting volatility for Newton-Raphson, by default 0.2.

        Returns
        -------
        float
            Implied volatility (annualised).

        Raises
        ------
        ValueError
            If the market price violates no-arbitrage bounds.
        RuntimeError
            If the solver fails to converge within the iteration limit.
        """
        T = max(T, _MIN_T)
        discount = math.exp(-r * T)
        intrinsic = (
            max(S - K * discount, 0.0)
            if option_type.lower() == "call"
            else max(K * discount - S, 0.0)
        )
        if market_price < intrinsic - 1e-6:
            raise ValueError(
                f"market_price {market_price:.6f} is below intrinsic value {intrinsic:.6f}"
            )

        sigma = max(initial_guess, _MIN_SIGMA)

        # Newton-Raphson iterations
        for i in range(_MAX_ITER_IV):
            price_est = self.price(S, K, T, r, sigma, option_type)
            diff = price_est - market_price
            if abs(diff) < _IV_TOL:
                logger.debug("IV converged in %d NR iterations, sigma=%.6f", i, sigma)
                return sigma
            # raw vega (per unit sigma, not per 1%)
            T_eff = max(T, _MIN_T)
            d1, _ = self._d1_d2(S, K, T_eff, r, sigma)
            vega_raw = S * norm.pdf(d1) * math.sqrt(T_eff)
            if abs(vega_raw) < 1e-12:
                break  # fall through to bisection
            sigma -= diff / vega_raw
            sigma = max(sigma, _MIN_SIGMA)

        # Bisection fallback
        logger.debug("NR did not converge; switching to bisection for IV")
        lo, hi = 1e-6, 10.0
        for _ in range(500):
            mid = 0.5 * (lo + hi)
            diff = self.price(S, K, T, r, mid, option_type) - market_price
            if abs(diff) < _IV_TOL or (hi - lo) < _IV_TOL:
                return mid
            if diff > 0:
                hi = mid
            else:
                lo = mid

        raise RuntimeError(
            f"Implied volatility solver failed to converge for market_price={market_price}"
        )

    # ------------------------------------------------------------------
    # Vectorised pricing
    # ------------------------------------------------------------------

    def price_vectorized(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: str = "call",
    ) -> np.ndarray:
        """Vectorised Black-Scholes pricing over arrays of inputs.

        Parameters
        ----------
        S : np.ndarray
            Array of underlying prices.
        K : np.ndarray
            Array of strike prices (same shape as ``S``).
        T : np.ndarray
            Array of times to expiry in years (same shape as ``S``).
        r : float
            Continuously compounded risk-free rate (scalar).
        sigma : np.ndarray
            Array of annualised volatilities (same shape as ``S``).
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        np.ndarray
            Array of option prices with the same shape as ``S``.

        Raises
        ------
        ValueError
            If ``option_type`` is unrecognised or shapes are incompatible.
        """
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        T = np.maximum(T, _MIN_T)
        sigma = np.maximum(sigma, _MIN_SIGMA)
        sqrt_T = np.sqrt(T)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        discount = np.exp(-r * T)

        if option_type == "call":
            prices = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        else:
            prices = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

        return prices
