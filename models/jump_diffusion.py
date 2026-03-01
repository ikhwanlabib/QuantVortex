"""Merton jump-diffusion model: pricing, simulation, and calibration.

Implements the Merton (1976) jump-diffusion model for European option pricing
via the closed-form Poisson-weighted series of Black-Scholes prices, Monte Carlo
path simulation with compound Poisson jumps, and maximum-likelihood calibration
from historical return data.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

_MIN_T = 1e-9
_MIN_SIGMA = 1e-9


class MertonJumpDiffusion:
    """Merton (1976) jump-diffusion model for European options.

    The asset price follows:

    .. math::
        \\frac{dS}{S} = (\\mu - \\lambda \\bar{k}) dt
                        + \\sigma dW_t
                        + (J - 1) dN_t

    where :math:`N_t` is a Poisson process with intensity :math:`\\lambda`,
    :math:`\\ln J \\sim \\mathcal{N}(\\mu_j, \\sigma_j^2)`, and
    :math:`\\bar{k} = e^{\\mu_j + \\sigma_j^2 / 2} - 1`.

    Examples
    --------
    >>> mjd = MertonJumpDiffusion()
    >>> p = mjd.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    ...               lam=0.5, mu_j=-0.1, sigma_j=0.15)
    >>> print(f"Merton price: {p:.4f}")
    """

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
        lam: float,
        mu_j: float,
        sigma_j: float,
        option_type: str = "call",
        n_terms: int = 50,
    ) -> float:
        """Closed-form Merton jump-diffusion price via Poisson series.

        The price is computed as a probability-weighted sum of Black-Scholes
        prices, each conditioned on exactly :math:`n` jumps occurring:

        .. math::
            C = \\sum_{n=0}^{N} \\frac{e^{-\\lambda' T} (\\lambda' T)^n}{n!}
                \\cdot C_{\\text{BS}}(S, K, T, r_n, \\sigma_n)

        where :math:`\\lambda' = \\lambda e^{\\mu_j + \\sigma_j^2/2}`,
        :math:`r_n = r - \\lambda \\bar{k} + n (\\mu_j + \\sigma_j^2/2) / T`,
        :math:`\\sigma_n^2 = \\sigma^2 + n \\sigma_j^2 / T`.

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
            Diffusion (non-jump) volatility.
        lam : float
            Jump intensity (expected jumps per year, :math:`\\lambda \\geq 0`).
        mu_j : float
            Mean of log-jump size.
        sigma_j : float
            Standard deviation of log-jump size.
        option_type : str, optional
            ``'call'`` or ``'put'``, by default ``'call'``.
        n_terms : int, optional
            Number of Poisson series terms, by default 50.

        Returns
        -------
        float
            Merton jump-diffusion option price.

        Raises
        ------
        ValueError
            If input parameters are invalid.
        """
        if S <= 0 or K <= 0:
            raise ValueError(f"S and K must be positive, got S={S}, K={K}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {sigma_j}")
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        T = max(T, _MIN_T)

        # Expected jump factor
        k_bar = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
        lam_prime = lam * (1.0 + k_bar)  # risk-neutral jump intensity

        price_total = 0.0
        log_fact = 0.0  # log(n!) accumulated

        for n in range(n_terms):
            # Poisson weight: e^{-lam' T} * (lam' T)^n / n!
            poisson_log = (
                -lam_prime * T + n * math.log(lam_prime * T + 1e-300) - log_fact
            )
            poisson_weight = math.exp(poisson_log)

            if poisson_weight < 1e-20:
                break

            # Adjusted parameters for n jumps
            sigma_n = math.sqrt(max(sigma ** 2 + n * sigma_j ** 2 / T, _MIN_SIGMA ** 2))
            r_n = r - lam * k_bar + n * (mu_j + 0.5 * sigma_j ** 2) / T

            bs_price = self._bs_price(S, K, T, r_n, sigma_n, option_type)
            price_total += poisson_weight * bs_price

            if n > 0:
                log_fact += math.log(n + 1)

        logger.debug(
            "Merton price(%s, S=%.4f, K=%.4f, T=%.4f, lam=%.4f) = %.6f",
            option_type, S, K, T, lam, price_total,
        )
        return price_total

    @staticmethod
    def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Black-Scholes price helper (no input validation)."""
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_SIGMA)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        discount = math.exp(-r * T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
        n_steps: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate Merton jump-diffusion paths via exact Euler-Maruyama + compound Poisson jumps.

        Each time step simulates diffusion and compound Poisson jumps separately,
        exploiting the independence of the jump and diffusion components.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        T : float
            Simulation horizon in years.
        r : float
            Risk-free drift rate.
        sigma : float
            Diffusion volatility (annualised).
        lam : float
            Jump intensity (jumps per year).
        mu_j : float
            Mean of log-jump size.
        sigma_j : float
            Standard deviation of log-jump size.
        n_steps : int, optional
            Number of time steps, by default 252.
        n_paths : int, optional
            Number of paths, by default 1000.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Simulated paths of shape ``(n_paths, n_steps + 1)``.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        k_bar = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

        S = np.full((n_paths, n_steps + 1), float(S0))

        logger.info(
            "Simulating %d Merton jump-diffusion paths with %d steps",
            n_paths, n_steps,
        )

        for t in range(n_steps):
            # Diffusion component
            z = rng.standard_normal(n_paths)
            diffusion = (r - lam * k_bar - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z

            # Jump component: number of jumps in [t, t+dt]
            n_jumps = rng.poisson(lam * dt, n_paths)
            log_jump = np.zeros(n_paths)
            for i in np.where(n_jumps > 0)[0]:
                log_jump[i] = np.sum(
                    rng.normal(mu_j, sigma_j, n_jumps[i])
                )

            S[:, t + 1] = S[:, t] * np.exp(diffusion + log_jump)

        return S

    # ------------------------------------------------------------------
    # Calibration via MLE
    # ------------------------------------------------------------------

    def calibrate(
        self,
        returns: np.ndarray,
        initial_params: Optional[dict] = None,
    ) -> dict:
        """Estimate Merton jump-diffusion parameters via maximum likelihood.

        The log-likelihood of observed daily log-returns is evaluated as a
        Poisson-weighted mixture of normal densities (conditioned on jump count).

        Parameters
        ----------
        returns : np.ndarray
            Array of continuously compounded daily returns (log-returns).
        initial_params : dict, optional
            Starting values with keys ``sigma``, ``lam``, ``mu_j``, ``sigma_j``.
            Sensible defaults are used if not provided.

        Returns
        -------
        dict
            Estimated parameters: ``sigma``, ``lam``, ``mu_j``, ``sigma_j``,
            ``log_likelihood``, ``aic``, ``bic``.

        Raises
        ------
        ValueError
            If ``returns`` is empty or contains NaN values.
        RuntimeError
            If optimisation fails to converge.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            raise ValueError("returns array is empty")
        if np.any(np.isnan(returns)):
            raise ValueError("returns array contains NaN values")

        dt = 1.0 / 252.0  # assume daily returns

        if initial_params is None:
            rv_std = np.std(returns)
            initial_params = {
                "sigma": max(rv_std * np.sqrt(252) * 0.8, 0.05),
                "lam": 5.0,
                "mu_j": np.mean(returns[np.abs(returns) > 2 * rv_std]) if np.any(np.abs(returns) > 2 * rv_std) else -0.02,
                "sigma_j": rv_std * 2.0,
            }

        x0 = [
            initial_params["sigma"],
            initial_params["lam"],
            initial_params["mu_j"],
            initial_params["sigma_j"],
        ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            sigma, lam, mu_j, sigma_j = params
            if sigma <= 0 or lam < 0 or sigma_j <= 0:
                return 1e12

            n_terms = min(30, max(10, int(lam * 5)))
            k_bar = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
            drift = -lam * k_bar  # risk-neutral adjustment (ignored under P; kept for structure)

            total_ll = np.zeros(len(returns))
            log_fact = 0.0
            for n in range(n_terms):
                poisson_weight = math.exp(-lam * dt) * (lam * dt) ** n / math.exp(log_fact)
                mu_n = drift * dt + n * mu_j
                sigma_n = math.sqrt(sigma ** 2 * dt + n * sigma_j ** 2)
                if sigma_n < 1e-10:
                    sigma_n = 1e-10
                total_ll += poisson_weight * norm.pdf(returns, loc=mu_n, scale=sigma_n)
                if n > 0:
                    log_fact += math.log(n + 1)

            total_ll = np.maximum(total_ll, 1e-300)
            return -np.sum(np.log(total_ll))

        bounds = [(0.001, 5.0), (0.0, 100.0), (-2.0, 2.0), (0.001, 2.0)]

        logger.info("Starting Merton MLE calibration on %d return observations", len(returns))
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("Merton MLE did not fully converge: %s", result.message)

        sigma, lam, mu_j, sigma_j = result.x
        n_params = 4
        ll = -result.fun
        aic = 2 * n_params - 2 * ll
        bic = n_params * math.log(len(returns)) - 2 * ll

        calibrated = {
            "sigma": float(sigma),
            "lam": float(lam),
            "mu_j": float(mu_j),
            "sigma_j": float(sigma_j),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "success": bool(result.success),
            "message": result.message,
        }
        logger.info(
            "Merton calibration: sigma=%.4f, lam=%.4f, mu_j=%.4f, sigma_j=%.4f, "
            "LL=%.2f, AIC=%.2f",
            sigma, lam, mu_j, sigma_j, ll, aic,
        )
        return calibrated
