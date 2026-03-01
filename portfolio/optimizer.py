"""
Portfolio optimization module for QuantVortex trading engine.

Implements multiple portfolio construction paradigms including mean-variance,
Black-Litterman, risk parity, hierarchical risk parity, and Kelly criterion.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Multi-paradigm portfolio optimizer for production quantitative trading.

    Provides a suite of portfolio construction methods ranging from classical
    Markowitz mean-variance to modern machine-learning-inspired hierarchical
    risk parity.

    Parameters
    ----------
    rf_rate : float, optional
        Annual risk-free rate used in Sharpe ratio calculations. Default is 0.0.

    Examples
    --------
    >>> opt = PortfolioOptimizer(rf_rate=0.04)
    >>> weights = opt.mean_variance(returns_df)
    """

    def __init__(self, rf_rate: float = 0.0) -> None:
        self.rf_rate = rf_rate
        logger.info("PortfolioOptimizer initialised with rf_rate=%.4f", rf_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """Return annualised sample covariance matrix."""
        return returns.cov().values * 252

    def _mean_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """Return annualised mean returns vector."""
        return returns.mean().values * 252

    def _portfolio_stats(
        self, weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute (expected_return, volatility, sharpe) for a weight vector."""
        port_ret = float(weights @ mu)
        port_var = float(weights @ sigma @ weights)
        port_vol = float(np.sqrt(max(port_var, 1e-12)))
        sharpe = (port_ret - self.rf_rate) / port_vol
        return port_ret, port_vol, sharpe

    # ------------------------------------------------------------------
    # Mean-Variance Optimisation
    # ------------------------------------------------------------------

    def mean_variance(
        self,
        returns: pd.DataFrame,
        risk_aversion: float = 1.0,
        long_only: bool = True,
    ) -> np.ndarray:
        """Markowitz mean-variance optimisation.

        Maximises the quadratic utility ``mu'w - (lambda/2) * w'Sigma w``
        subject to ``sum(w) == 1`` and, optionally, ``w >= 0``.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.
        risk_aversion : float, optional
            Lambda coefficient controlling the return/risk trade-off. Default 1.0.
        long_only : bool, optional
            Enforce non-negative weights when ``True``. Default ``True``.

        Returns
        -------
        np.ndarray
            Optimal weight vector of shape ``(N,)``.

        Raises
        ------
        ValueError
            If ``returns`` is empty or contains fewer than two assets.
        """
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets.")

        n = returns.shape[1]
        mu = self._mean_returns(returns)
        sigma = self._covariance_matrix(returns)

        def neg_utility(w: np.ndarray) -> float:
            return -(w @ mu - 0.5 * risk_aversion * w @ sigma @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            logger.warning("mean_variance optimisation did not converge: %s", result.message)

        weights = result.x
        weights = np.clip(weights, 0.0, None) if long_only else weights
        weights /= weights.sum()
        logger.debug("mean_variance weights computed, max=%.4f min=%.4f", weights.max(), weights.min())
        return weights

    # ------------------------------------------------------------------
    # Efficient Frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self, returns: pd.DataFrame, n_points: int = 100
    ) -> pd.DataFrame:
        """Trace the mean-variance efficient frontier.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.
        n_points : int, optional
            Number of frontier portfolios to compute. Default 100.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['return', 'volatility', 'sharpe', 'weights']``.
            The ``weights`` column contains ``np.ndarray`` objects.

        Raises
        ------
        ValueError
            If ``n_points`` is less than 2.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets.")

        n = returns.shape[1]
        mu = self._mean_returns(returns)
        sigma = self._covariance_matrix(returns)
        bounds = [(0.0, 1.0)] * n
        constraints_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n

        min_ret = float(mu.min())
        max_ret = float(mu.max())
        target_returns = np.linspace(min_ret, max_ret, n_points)

        records: list[dict] = []
        for target in target_returns:
            cons = constraints_base + [
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
            ]
            result = minimize(
                lambda w: float(w @ sigma @ w),
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if not result.success:
                logger.debug("efficient_frontier: point at target=%.4f did not converge", target)
                continue
            w = result.x
            w = np.clip(w, 0.0, None)
            w /= w.sum()
            ret, vol, sharpe = self._portfolio_stats(w, mu, sigma)
            records.append({"return": ret, "volatility": vol, "sharpe": sharpe, "weights": w})

        if not records:
            raise RuntimeError("Efficient frontier computation produced no valid portfolios.")

        frontier = pd.DataFrame(records)
        logger.info("efficient_frontier: computed %d valid points", len(frontier))
        return frontier

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        returns: pd.DataFrame,
        views: np.ndarray,
        view_confidences: np.ndarray,
        market_caps: Optional[np.ndarray] = None,
        tau: float = 0.05,
    ) -> np.ndarray:
        """Black-Litterman posterior optimal weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.
        views : np.ndarray
            Absolute return views vector, shape ``(K,)``.
        view_confidences : np.ndarray
            Diagonal confidence values for the view uncertainty matrix ``Omega``,
            shape ``(K,)``. Larger values indicate lower confidence.
        market_caps : np.ndarray, optional
            Market capitalisation weights for equilibrium prior. If ``None``,
            equal weights are used.
        tau : float, optional
            Scaling factor for the prior covariance. Default 0.05.

        Returns
        -------
        np.ndarray
            Posterior optimal weight vector of shape ``(N,)``.

        Raises
        ------
        ValueError
            If dimension mismatches are detected.

        Notes
        -----
        Uses a pick matrix ``P`` of shape ``(K, N)`` built as an identity-like
        matrix (one view per asset in order). If K < N, only the first K assets
        receive absolute views.
        """
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        n = returns.shape[1]
        k = len(views)
        if len(view_confidences) != k:
            raise ValueError("views and view_confidences must have the same length.")
        if k > n:
            raise ValueError("Number of views cannot exceed number of assets.")

        sigma = self._covariance_matrix(returns)
        mu_eq_raw = self._mean_returns(returns)

        if market_caps is not None:
            if len(market_caps) != n:
                raise ValueError("market_caps length must match number of assets.")
            w_eq = np.array(market_caps, dtype=float)
            w_eq /= w_eq.sum()
        else:
            w_eq = np.ones(n) / n

        # Implied equilibrium excess returns: pi = delta * Sigma * w_eq
        delta = 2.5  # market risk aversion
        pi = delta * sigma @ w_eq

        # Pick matrix: absolute views on first K assets
        P = np.zeros((k, n))
        for i in range(k):
            P[i, i] = 1.0

        omega = np.diag(view_confidences)

        # BL posterior mean
        tau_sigma = tau * sigma
        M = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
        mu_bl = M @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ views)

        # Posterior weights via mean-variance with posterior mean
        returns_bl = returns.copy()
        # Inject posterior mean as synthetic return signal
        try:
            weights = self.mean_variance(
                returns_bl,
                risk_aversion=delta,
                long_only=True,
            )
            # Tilt toward BL signal
            raw = np.linalg.inv(sigma) @ mu_bl
            raw = np.clip(raw, 0.0, None)
            raw_sum = float(raw.sum())
            if raw_sum > 1e-12:
                weights = raw / raw_sum
            else:
                logger.warning("BL signal produced all-zero weights; retaining mean-variance result.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("BL mean-variance step failed (%s); falling back to equal weights.", exc)
            weights = np.ones(n) / n

        logger.info("black_litterman: posterior weights computed for %d assets", n)
        return weights

    # ------------------------------------------------------------------
    # Risk Parity
    # ------------------------------------------------------------------

    def risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Equal risk contribution (risk parity) portfolio.

        Minimises ``sum_i (w_i * RC_i - 1/N)^2`` where
        ``RC_i = w_i * (Sigma @ w)_i / sigma_p`` is the fractional risk
        contribution of asset *i*.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.

        Returns
        -------
        np.ndarray
            Risk-parity weight vector of shape ``(N,)``.
        """
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets.")

        n = returns.shape[1]
        sigma = self._covariance_matrix(returns)
        target = np.ones(n) / n

        def objective(w: np.ndarray) -> float:
            port_var = float(w @ sigma @ w)
            if port_var < 1e-14:
                return 0.0
            mrc = sigma @ w  # marginal risk contributions (unnormalised)
            rc = w * mrc / np.sqrt(port_var)
            return float(np.sum((rc - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, 1.0)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 2000},
        )

        if not result.success:
            logger.warning("risk_parity did not fully converge: %s", result.message)

        weights = np.clip(result.x, 0.0, None)
        weights /= weights.sum()
        logger.debug("risk_parity weights: %s", weights)
        return weights

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity
    # ------------------------------------------------------------------

    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity (HRP) portfolio construction.

        Implements López de Prado's HRP algorithm:
        1. Compute correlation-based distance matrix.
        2. Cluster assets via Ward linkage.
        3. Allocate weights by recursive bisection using inverse-variance.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.

        Returns
        -------
        np.ndarray
            HRP weight vector of shape ``(N,)``, ordered to match ``returns.columns``.
        """
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets.")

        corr = returns.corr().values
        cov = self._covariance_matrix(returns)
        n = returns.shape[1]

        # Distance matrix from correlation
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
        dist_condensed = squareform(dist, checks=False)

        link = linkage(dist_condensed, method="ward")

        # Quasi-diagonalisation: recover leaf order from linkage
        def _get_quasi_diag(link_mat: np.ndarray) -> list[int]:
            link_mat = link_mat.astype(int)
            sort_ix = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
            n_items = link_mat[-1, 3]
            while sort_ix.max() >= n_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= n_items]
                i = df0.index
                j = df0.values - n_items
                sort_ix[i] = link_mat[j, 0]
                df0 = pd.Series(link_mat[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0]).sort_index()
                sort_ix = sort_ix.drop_duplicates()
            return sort_ix.tolist()

        sorted_indices = _get_quasi_diag(link)
        sorted_indices = [int(i) for i in sorted_indices if i < n]

        # Recursive bisection
        weights = pd.Series(1.0, index=range(n))

        def _ivp(cov_slice: np.ndarray) -> np.ndarray:
            ivp = 1.0 / np.diag(cov_slice)
            return ivp / ivp.sum()

        def _cluster_var(cov_full: np.ndarray, cluster: list[int]) -> float:
            sub_cov = cov_full[np.ix_(cluster, cluster)]
            w = _ivp(sub_cov)
            return float(w @ sub_cov @ w)

        def _hrp_alloc(cov_full: np.ndarray, sorted_idx: list[int]) -> None:
            if len(sorted_idx) <= 1:
                return
            mid = len(sorted_idx) // 2
            left = sorted_idx[:mid]
            right = sorted_idx[mid:]
            var_left = _cluster_var(cov_full, left)
            var_right = _cluster_var(cov_full, right)
            alpha = 1.0 - var_left / (var_left + var_right + 1e-14)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            _hrp_alloc(cov_full, left)
            _hrp_alloc(cov_full, right)

        _hrp_alloc(cov, sorted_indices)

        w_array = weights.values.astype(float)
        w_array /= w_array.sum()
        logger.info("hierarchical_risk_parity: weights computed for %d assets", n)
        return w_array

    # ------------------------------------------------------------------
    # Kelly Criterion
    # ------------------------------------------------------------------

    def kelly_criterion(self, returns: pd.DataFrame) -> np.ndarray:
        """Fractional (half-Kelly) optimal weights.

        Computes ``f* = 0.5 * Sigma^{-1} * mu`` and projects onto the
        probability simplex (long-only, sum-to-one).

        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns, shape ``(T, N)``.

        Returns
        -------
        np.ndarray
            Half-Kelly weight vector of shape ``(N,)`` normalised to sum to 1.
        """
        if returns.empty or returns.shape[1] < 1:
            raise ValueError("returns must be non-empty.")

        mu = self._mean_returns(returns)
        sigma = self._covariance_matrix(returns)

        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular; using pseudo-inverse for Kelly.")
            sigma_inv = np.linalg.pinv(sigma)

        f_star = 0.5 * sigma_inv @ mu
        f_star = np.clip(f_star, 0.0, None)

        if f_star.sum() < 1e-14:
            logger.warning("Kelly criterion produced all-zero weights; returning equal weights.")
            return np.ones(len(mu)) / len(mu)

        weights = f_star / f_star.sum()
        logger.debug("kelly_criterion weights: max=%.4f", weights.max())
        return weights

    # ------------------------------------------------------------------
    # Constraint Application
    # ------------------------------------------------------------------

    def apply_constraints(
        self,
        weights: np.ndarray,
        max_position: float = 0.10,
        max_sector: float = 0.30,
        sectors: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Apply position and sector concentration limits to a weight vector.

        Iteratively clips individual weights at ``max_position`` and, when
        ``sectors`` is provided, clips aggregate sector weights at
        ``max_sector``. The resulting vector is renormalised to sum to 1
        after each pass.

        Parameters
        ----------
        weights : np.ndarray
            Raw weight vector of shape ``(N,)``.
        max_position : float, optional
            Maximum weight for any single asset. Default 0.10.
        max_sector : float, optional
            Maximum aggregate weight for any sector. Default 0.30.
        sectors : dict of str to int, optional
            Mapping from asset name/index string to integer sector id.
            Keys must align with the positional index of ``weights``.

        Returns
        -------
        np.ndarray
            Constrained and normalised weight vector of shape ``(N,)``.

        Raises
        ------
        ValueError
            If ``weights`` is empty or contains NaN values.
        """
        if len(weights) == 0:
            raise ValueError("weights array must not be empty.")
        if np.any(np.isnan(weights)):
            raise ValueError("weights array contains NaN values.")

        w = np.array(weights, dtype=float)

        # Position limit: iterative proportional clipping
        for _ in range(200):
            excess = np.maximum(w - max_position, 0.0)
            if excess.sum() < 1e-12:
                break
            w = np.minimum(w, max_position)
            remaining = 1.0 - w.sum()
            below_mask = w < max_position
            n_below = below_mask.sum()
            if n_below == 0:
                break
            below_sum = w[below_mask].sum()
            if below_sum > 1e-14:
                w[below_mask] += remaining * (w[below_mask] / below_sum)
            else:
                # Distribute equally among unconstrained assets
                w[below_mask] += remaining / n_below

        # Sector limit
        if sectors is not None:
            sector_ids = np.array(list(sectors.values()), dtype=int)
            unique_sectors = np.unique(sector_ids)
            for sid in unique_sectors:
                mask = sector_ids == sid
                sector_total = w[mask].sum()
                if sector_total > max_sector:
                    scale = max_sector / sector_total
                    w[mask] *= scale

        # Final normalisation
        total = w.sum()
        if total < 1e-14:
            logger.warning("apply_constraints: all weights near zero; returning equal weights.")
            return np.ones(len(w)) / len(w)

        w /= total
        logger.debug("apply_constraints: final weights sum=%.6f, max=%.4f", w.sum(), w.max())
        return w
