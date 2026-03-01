"""Heston stochastic volatility model: pricing, calibration, and simulation.

Implements the Heston (1993) model using semi-analytical characteristic function
integration for European option pricing, differential evolution for calibration,
and the Milstein discretisation scheme for Monte Carlo path simulation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.stats import norm

logger = logging.getLogger(__name__)

_MIN_T = 1e-9


class HestonModel:
    """Heston stochastic volatility option pricing model.

    The Heston model describes asset dynamics as:

    .. math::
        dS_t = r S_t dt + \\sqrt{v_t} S_t dW_t^S

        dv_t = \\kappa (\\theta - v_t) dt + \\sigma \\sqrt{v_t} dW_t^v

        \\text{Corr}(dW_t^S, dW_t^v) = \\rho

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    v0 : float
        Initial variance.

    Examples
    --------
    >>> hm = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
    >>> price = hm.price(S=100, K=100, T=1.0, r=0.05)
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        logger.debug(
            "HestonModel initialised: kappa=%.4f, theta=%.4f, sigma=%.4f, "
            "rho=%.4f, v0=%.4f",
            kappa, theta, sigma, rho, v0,
        )

    # ------------------------------------------------------------------
    # Characteristic function
    # ------------------------------------------------------------------

    @staticmethod
    def characteristic_function(
        phi: complex,
        S: float,
        K: float,
        T: float,
        r: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
    ) -> complex:
        """Heston characteristic function for log-price.

        Implements the original Heston (1993) characteristic function using
        the formulation that avoids the branch-cut discontinuity.

        Parameters
        ----------
        phi : complex
            Frequency variable.
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Risk-free rate.
        kappa : float
            Mean-reversion speed.
        theta : float
            Long-run variance.
        sigma : float
            Volatility of variance.
        rho : float
            Correlation.
        v0 : float
            Initial variance.

        Returns
        -------
        complex
            Value of the characteristic function at ``phi``.
        """
        T = max(T, _MIN_T)
        x = np.log(S / K)

        # Albrecher et al. (2007) rotation-count-free formulation
        xi = kappa - rho * sigma * phi * 1j
        d = np.sqrt(xi ** 2 + sigma ** 2 * (phi ** 2 + phi * 1j))

        g = (xi - d) / (xi + d + 1e-300)
        exp_dT = np.exp(-d * T)
        denom = 1.0 - g * exp_dT
        if np.abs(denom) < 1e-300:
            denom = complex(1e-300)

        # Log form to avoid overflow in C computation
        log_1mg = np.log(1.0 - g + 1e-300)
        log_denom = np.log(denom + 1e-300)
        C = r * phi * 1j * T + (kappa * theta / sigma ** 2) * (
            (xi - d) * T - 2.0 * (log_denom - log_1mg)
        )
        D = (xi - d) / sigma ** 2 * (1.0 - exp_dT) / denom

        exponent = C + D * v0 + 1j * phi * x
        # Clamp real part to prevent overflow
        real_part = np.real(exponent)
        if real_part > 500:
            exponent = complex(500.0, np.imag(exponent))
        return np.exp(exponent)

    # ------------------------------------------------------------------
    # Semi-analytical pricing
    # ------------------------------------------------------------------

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        kappa: Optional[float] = None,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        rho: Optional[float] = None,
        v0: Optional[float] = None,
        option_type: str = "call",
    ) -> float:
        """Price a European option under the Heston model via Gauss-Laguerre quadrature.

        Uses the Gil-Pelaez inversion formula:

        .. math::
            C = S \\Pi_1 - K e^{-rT} \\Pi_2

        where :math:`\\Pi_j` are computed as Fourier integrals of the
        characteristic function.

        Parameters
        ----------
        S : float
            Current underlying price.
        K : float
            Strike price.
        T : float
            Time to expiry in years.
        r : float
            Risk-free rate.
        kappa : float, optional
            Override mean-reversion speed (uses instance attribute if None).
        theta : float, optional
            Override long-run variance (uses instance attribute if None).
        sigma : float, optional
            Override vol-of-vol (uses instance attribute if None).
        rho : float, optional
            Override correlation (uses instance attribute if None).
        v0 : float, optional
            Override initial variance (uses instance attribute if None).
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.

        Returns
        -------
        float
            Heston option price.

        Raises
        ------
        ValueError
            If ``option_type`` is unrecognised.
        """
        kappa = kappa if kappa is not None else self.kappa
        theta = theta if theta is not None else self.theta
        sigma = sigma if sigma is not None else self.sigma
        rho = rho if rho is not None else self.rho
        v0 = v0 if v0 is not None else self.v0
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        T = max(T, _MIN_T)

        def integrand_p1(phi: float) -> float:
            try:
                cf = self.characteristic_function(
                    phi - 1j, S, K, T, r, kappa, theta, sigma, rho, v0
                )
                cf0 = self.characteristic_function(
                    -1j, S, K, T, r, kappa, theta, sigma, rho, v0
                )
                if abs(cf0) < 1e-15 or not np.isfinite(abs(cf)) or not np.isfinite(abs(cf0)):
                    return 0.0
                # CF already encodes x=ln(S/K), so no extra e^{-i*phi*ln(K)} needed
                val = np.real(cf / (1j * phi * cf0))
                return float(val) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        def integrand_p2(phi: float) -> float:
            try:
                cf = self.characteristic_function(
                    phi, S, K, T, r, kappa, theta, sigma, rho, v0
                )
                if not np.isfinite(abs(cf)):
                    return 0.0
                # CF already encodes x=ln(S/K), so no extra e^{-i*phi*ln(K)} needed
                val = np.real(cf / (1j * phi))
                return float(val) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        upper_limit = 200.0
        limit = 200
        p1_integral, _ = quad(integrand_p1, 1e-8, upper_limit, limit=limit, epsabs=1e-6, epsrel=1e-6)
        p2_integral, _ = quad(integrand_p2, 1e-8, upper_limit, limit=limit, epsabs=1e-6, epsrel=1e-6)

        P1 = 0.5 + p1_integral / np.pi
        P2 = 0.5 + p2_integral / np.pi

        discount = np.exp(-r * T)
        call_price = S * P1 - K * discount * P2
        call_price = max(call_price, max(S - K * discount, 0.0))

        if option_type == "call":
            result = call_price
        else:
            result = call_price - S + K * discount  # put-call parity

        logger.debug(
            "Heston price(%s, S=%.4f, K=%.4f, T=%.4f) = %.6f",
            option_type, S, K, T, result,
        )
        return float(result)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        market_data: pd.DataFrame,
        S: float,
        r: float,
    ) -> dict:
        """Calibrate Heston parameters to market option prices using differential evolution.

        Parameters
        ----------
        market_data : pd.DataFrame
            DataFrame with columns: ``strike``, ``expiry`` (in years),
            ``market_price``, ``option_type`` (``'call'`` or ``'put'``).
        S : float
            Current underlying spot price.
        r : float
            Risk-free rate.

        Returns
        -------
        dict
            Calibrated parameters: ``kappa``, ``theta``, ``sigma``,
            ``rho``, ``v0``, ``rmse`` (root-mean-square pricing error).

        Raises
        ------
        ValueError
            If ``market_data`` is missing required columns.
        """
        required = {"strike", "expiry", "market_price", "option_type"}
        missing = required - set(market_data.columns)
        if missing:
            raise ValueError(f"market_data is missing columns: {missing}")

        strikes = market_data["strike"].values
        expiries = market_data["expiry"].values
        mkt_prices = market_data["market_price"].values
        opt_types = market_data["option_type"].values

        def objective(params: np.ndarray) -> float:
            kappa_, theta_, sigma_, rho_, v0_ = params
            # Feller condition soft penalty
            feller_penalty = max(0.0, 2 * kappa_ * theta_ - sigma_ ** 2)
            errors = []
            for K, T, mp, ot in zip(strikes, expiries, mkt_prices, opt_types):
                try:
                    model_price = self.price(
                        S, K, T, r, kappa_, theta_, sigma_, rho_, v0_, ot
                    )
                    errors.append((model_price - mp) ** 2)
                except Exception:
                    errors.append(1e6)
            rmse = np.sqrt(np.mean(errors))
            return rmse + 1e-4 * max(0.0, -(2 * kappa_ * theta_ - sigma_ ** 2))

        bounds = [
            (0.01, 20.0),   # kappa
            (0.001, 1.0),   # theta
            (0.01, 2.0),    # sigma
            (-0.99, 0.99),  # rho
            (0.001, 1.0),   # v0
        ]

        logger.info("Starting Heston calibration with %d market quotes", len(market_data))
        result = differential_evolution(
            objective,
            bounds,
            maxiter=300,
            tol=1e-7,
            seed=42,
            workers=1,
            popsize=15,
        )

        kappa, theta, sigma, rho, v0 = result.x
        # Update instance parameters
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = (
            kappa, theta, sigma, rho, v0
        )

        calibrated = {
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma": float(sigma),
            "rho": float(rho),
            "v0": float(v0),
            "rmse": float(result.fun),
            "success": bool(result.success),
            "message": result.message,
        }
        logger.info("Heston calibration complete: RMSE=%.6f, success=%s", result.fun, result.success)
        return calibrated

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        kappa: Optional[float] = None,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        rho: Optional[float] = None,
        v0: Optional[float] = None,
        n_steps: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate Heston asset paths using the Milstein discretisation scheme.

        Uses the full-truncation scheme to handle the variance process and
        Milstein corrections for second-order accuracy.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        T : float
            Total simulation horizon in years.
        r : float
            Risk-free drift rate.
        kappa : float, optional
            Mean-reversion speed (instance default if None).
        theta : float, optional
            Long-run variance (instance default if None).
        sigma : float, optional
            Vol-of-vol (instance default if None).
        rho : float, optional
            Correlation (instance default if None).
        v0 : float, optional
            Initial variance (instance default if None).
        n_steps : int, optional
            Number of time steps, by default 252.
        n_paths : int, optional
            Number of simulated paths, by default 1000.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Asset price paths of shape ``(n_paths, n_steps + 1)``.
        """
        kappa = kappa if kappa is not None else self.kappa
        theta = theta if theta is not None else self.theta
        sigma = sigma if sigma is not None else self.sigma
        rho = rho if rho is not None else self.rho
        v0 = v0 if v0 is not None else self.v0

        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.full((n_paths, n_steps + 1), S0, dtype=float)
        v = np.full(n_paths, v0, dtype=float)

        sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

        logger.info(
            "Simulating %d Heston paths with %d steps (Milstein scheme)",
            n_paths, n_steps,
        )

        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            # Correlated Brownian increments
            dW_S = z1 * sqrt_dt
            dW_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt

            v_pos = np.maximum(v, 0.0)  # full truncation
            sqrt_v = np.sqrt(v_pos)

            # Milstein scheme for variance
            v_new = (
                v
                + kappa * (theta - v_pos) * dt
                + sigma * sqrt_v * dW_v
                + 0.25 * sigma ** 2 * (dW_v ** 2 - dt)
            )
            v = np.maximum(v_new, 0.0)

            # Log-Euler scheme for asset price
            S[:, t + 1] = S[:, t] * np.exp(
                (r - 0.5 * v_pos) * dt + sqrt_v * dW_S
            )

        return S
