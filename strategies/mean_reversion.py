"""
Ornstein-Uhlenbeck mean-reversion strategy for QuantVortex.

Fits OU parameters via maximum-likelihood estimation, derives optimal
entry/exit thresholds, constructs a PCA-based spread over a basket of
assets, and generates z-score trading signals.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import lstsq
from sklearn.decomposition import PCA
import statsmodels.api as sm

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversion(BaseStrategy):
    """Ornstein-Uhlenbeck mean-reversion strategy.

    Parameters
    ----------
    name : str, optional
        Strategy name. Default is ``"MeanReversion"``.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    n_components : int, optional
        Number of PCA components used to construct the spread. Default is 1.
    zscore_window : int, optional
        Rolling window for spread z-score. Default is 60.
    entry_z : float, optional
        Z-score magnitude that triggers position entry. Default is 1.5.
    exit_z : float, optional
        Z-score magnitude that triggers position exit. Default is 0.0.
    transaction_cost : float, optional
        Round-trip cost used in optimal threshold calculation. Default is 0.01.

    Attributes
    ----------
    ou_params : dict
        Most recently estimated OU parameters ``{mu, kappa, sigma}``.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model (set after :meth:`pca_spread` is called).
    """

    def __init__(
        self,
        name: str = "MeanReversion",
        allocation: float = 1.0,
        n_components: int = 1,
        zscore_window: int = 60,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        transaction_cost: float = 0.01,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        self.n_components = n_components
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.transaction_cost = transaction_cost
        self.ou_params: Dict[str, float] = {}
        self.pca_model: Optional[PCA] = None

    # ------------------------------------------------------------------
    # OU parameter estimation (MLE)
    # ------------------------------------------------------------------

    def estimate_ou_params(self, spread: pd.Series) -> Dict[str, float]:
        """Estimate OU process parameters via maximum-likelihood estimation.

        The discrete-time OU likelihood follows from the exact transition
        density (Gaussian with mean and variance depending on dt):

        .. math::

            X_{t+dt} | X_t \\sim \\mathcal{N}\\bigl(
                X_t e^{-\\kappa dt} + \\mu(1-e^{-\\kappa dt}),\\;
                \\frac{\\sigma^2}{2\\kappa}(1-e^{-2\\kappa dt})
            \\bigr)

        Parameters
        ----------
        spread : pd.Series
            Mean-reverting spread or price series (must have ≥ 20 values).

        Returns
        -------
        dict
            Keys: ``"mu"``, ``"kappa"``, ``"sigma_ou"``.

        Raises
        ------
        ValueError
            If fewer than 20 non-NaN observations are available.
        RuntimeError
            If the optimisation fails to converge.
        """
        clean = spread.dropna()
        if len(clean) < 20:
            raise ValueError(
                f"Need ≥ 20 observations for OU MLE; got {len(clean)}."
            )

        x = clean.values.astype(float)
        dt = 1.0  # daily observations (unit step)

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, log_kappa, log_sigma = params
            kappa = np.exp(log_kappa)
            sigma = np.exp(log_sigma)

            exp_kappa = np.exp(-kappa * dt)
            mean_cond = x[:-1] * exp_kappa + mu * (1 - exp_kappa)
            var_cond = (sigma ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * dt))

            if var_cond <= 0:
                return 1e12

            residuals = x[1:] - mean_cond
            n = len(residuals)
            ll = (
                -0.5 * n * np.log(2 * np.pi * var_cond)
                - 0.5 * np.sum(residuals ** 2) / var_cond
            )
            return -ll

        mu0 = float(x.mean())
        kappa0 = 0.1
        sigma0 = float(x.std())
        x0 = np.array([mu0, np.log(kappa0), np.log(max(sigma0, 1e-8))])

        result = minimize(
            neg_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-8},
        )
        if not result.success:
            self.logger.warning(
                "OU MLE did not converge cleanly: %s. Using best-effort params.",
                result.message,
            )

        mu_hat, log_kappa_hat, log_sigma_hat = result.x
        self.ou_params = {
            "mu": float(mu_hat),
            "kappa": float(np.exp(log_kappa_hat)),
            "sigma_ou": float(np.exp(log_sigma_hat)),
        }
        self.logger.debug("OU params: %s", self.ou_params)
        return self.ou_params

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_half_life(kappa: float) -> float:
        """Compute the OU mean-reversion half-life from speed parameter.

        .. math::

            t_{1/2} = \\frac{\\ln 2}{\\kappa}

        Parameters
        ----------
        kappa : float
            Mean-reversion speed (must be > 0).

        Returns
        -------
        float
            Half-life in the same time units as kappa.

        Raises
        ------
        ValueError
            If ``kappa ≤ 0``.
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}.")
        return float(np.log(2) / kappa)

    @staticmethod
    def optimal_thresholds(
        kappa: float,
        sigma: float,
        cost: float = 0.01,
    ) -> Tuple[float, float]:
        """Compute simple optimal entry/exit thresholds for an OU strategy.

        Approximates the entry threshold as the stationary standard
        deviation (with a cost-dependent adjustment) and sets the exit
        at zero:

        .. math::

            \\text{entry} = \\sigma \\sqrt{\\frac{1}{2\\kappa}} \\times (1 + c)

        Parameters
        ----------
        kappa : float
            Mean-reversion speed (> 0).
        sigma : float
            Diffusion coefficient of the OU process (> 0).
        cost : float, optional
            Round-trip transaction cost fraction. Default is 0.01.

        Returns
        -------
        tuple of float
            ``(entry_threshold, exit_threshold)``.  The entry threshold is
            positive; the exit threshold is always 0.

        Raises
        ------
        ValueError
            If ``kappa ≤ 0`` or ``sigma ≤ 0``.
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}.")

        sigma_stationary = sigma * np.sqrt(1.0 / (2.0 * kappa))
        entry = float(sigma_stationary * (1.0 + cost))
        exit_level = 0.0
        return entry, exit_level

    # ------------------------------------------------------------------
    # PCA spread construction
    # ------------------------------------------------------------------

    def pca_spread(
        self, prices: pd.DataFrame, n_components: Optional[int] = None
    ) -> pd.Series:
        """Construct a synthetic spread using PCA on normalised log prices.

        The spread is the projection of log-prices onto the first principal
        component(s), capturing the dominant common factor.  The residual
        from the first component approximates a stationary mean-reverting
        process when the panel is cointegrated.

        Parameters
        ----------
        prices : pd.DataFrame
            Wide-format price DataFrame (DatetimeIndex × tickers). Must
            have at least 2 columns.
        n_components : int or None, optional
            Override ``self.n_components``. Default is ``None``.

        Returns
        -------
        pd.Series
            First-PC-reconstructed spread series aligned to ``prices``'s
            index.

        Raises
        ------
        ValueError
            If ``prices`` has fewer than 2 columns or insufficient rows.
        """
        self._validate_price_df(prices)
        if prices.shape[1] < 2:
            raise ValueError("pca_spread requires at least 2 asset columns.")

        nc = n_components if n_components is not None else self.n_components
        nc = min(nc, prices.shape[1])

        log_prices = np.log(prices.replace(0, np.nan)).dropna()
        # Standardise columns
        mu_col = log_prices.mean()
        sd_col = log_prices.std().replace(0, np.nan)
        log_norm = (log_prices - mu_col) / sd_col

        self.pca_model = PCA(n_components=nc, svd_solver="full")
        scores = self.pca_model.fit_transform(log_norm.values)
        reconstructed = self.pca_model.inverse_transform(scores)

        # Spread = residual from PC reconstruction
        residuals = log_norm.values - reconstructed
        spread = pd.Series(
            residuals[:, 0] if residuals.ndim > 1 else residuals,
            index=log_prices.index,
            name="pca_spread",
        )
        self.logger.debug(
            "pca_spread: explained variance ratio=%s",
            self.pca_model.explained_variance_ratio_,
        )
        return spread

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-reversion signals from a price DataFrame.

        Pipeline:
        1. Construct PCA spread.
        2. Estimate OU parameters.
        3. Compute rolling z-score of the spread.
        4. Return a single-column DataFrame of z-scores.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format adjusted closing prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame ``["zscore"]`` with the spread z-score.
        """
        self._validate_price_df(data, min_rows=self.zscore_window + 10)

        spread = self.pca_spread(data)

        try:
            ou = self.estimate_ou_params(spread)
            hl = self.compute_half_life(ou["kappa"])
            self.logger.info(
                "OU params: mu=%.4f, kappa=%.4f, sigma=%.4f, half-life=%.1f days",
                ou["mu"],
                ou["kappa"],
                ou["sigma_ou"],
                hl,
            )
        except Exception as exc:
            self.logger.warning("OU estimation failed: %s. Proceeding with z-score.", exc)

        zscore = self._zscore(spread, window=self.zscore_window)
        signals = zscore.to_frame(name="zscore")
        self.logger.info(
            "generate_signals: %d rows, %d NaN",
            len(signals),
            int(signals["zscore"].isna().sum()),
        )
        return signals

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert z-score signals to ±1 position series.

        Entry at ``|z| ≥ entry_z``, exit at ``|z| ≤ exit_z``.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals` (column ``"zscore"``).

        Returns
        -------
        pd.DataFrame
            Single-column position DataFrame (values in ``{-1, 0, +1}``).
        """
        if signals.empty or "zscore" not in signals.columns:
            self.logger.warning("compute_positions: invalid signals input.")
            return pd.DataFrame(index=signals.index, columns=["position"])

        z = signals["zscore"]
        pos = np.zeros(len(z))
        current = 0.0

        for t, zt in enumerate(z.values):
            if np.isnan(zt):
                pos[t] = current
                continue
            if current == 0:
                if zt > self.entry_z:
                    current = -1.0
                elif zt < -self.entry_z:
                    current = 1.0
            else:
                if abs(zt) <= self.exit_z:
                    current = 0.0
            pos[t] = current

        positions = pd.DataFrame({"position": pos}, index=signals.index)
        self.logger.info(
            "compute_positions: long=%d, short=%d, flat=%d",
            int((positions["position"] > 0).sum()),
            int((positions["position"] < 0).sum()),
            int((positions["position"] == 0).sum()),
        )
        return positions
