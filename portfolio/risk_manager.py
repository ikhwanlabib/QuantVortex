"""
Risk management module for QuantVortex trading engine.

Provides parametric, historical, and Monte Carlo VaR/CVaR calculations,
drawdown analytics, position limit checks, stress testing, option Greeks
decomposition, and correlation-regime monitoring.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management for production portfolio systems.

    Provides Value-at-Risk (VaR) via multiple methodologies, CVaR/Expected
    Shortfall, drawdown analytics, stress testing, Greeks decomposition, and
    correlation-regime monitoring.

    Parameters
    ----------
    annual_factor : int, optional
        Number of trading days per year used for annualisation. Default 252.

    Examples
    --------
    >>> rm = RiskManager()
    >>> var = rm.var_parametric(returns_series, confidence=0.99)
    """

    def __init__(self, annual_factor: int = 252) -> None:
        self.annual_factor = annual_factor
        logger.info("RiskManager initialised with annual_factor=%d", annual_factor)

        # Pre-defined stress scenarios: multipliers on historical returns
        self._stress_scenarios: Dict[str, Dict[str, float]] = {
            "2008_crisis": {
                "start": "2008-09-01",
                "end": "2009-03-31",
                "label": "Global Financial Crisis 2008-09",
            },
            "covid": {
                "start": "2020-02-19",
                "end": "2020-03-23",
                "label": "COVID-19 Market Crash 2020",
            },
            "dot_com": {
                "start": "2000-03-10",
                "end": "2002-10-09",
                "label": "Dot-com Bubble Burst 2000-02",
            },
        }

    # ------------------------------------------------------------------
    # Parametric VaR
    # ------------------------------------------------------------------

    def var_parametric(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
    ) -> float:
        """Gaussian (parametric) Value-at-Risk.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        confidence : float, optional
            Confidence level in ``(0, 1)``. Default 0.99.
        window : int, optional
            Rolling window (most-recent observations) for moment estimation.
            Default 252.

        Returns
        -------
        float
            VaR as a positive number (i.e. the potential loss).

        Raises
        ------
        ValueError
            If ``confidence`` is not in ``(0, 1)`` or ``returns`` is empty.
        """
        self._validate_confidence(confidence)
        if returns.empty:
            raise ValueError("returns Series is empty.")

        data = returns.dropna().iloc[-window:] if len(returns) >= window else returns.dropna()
        if len(data) < 2:
            raise ValueError("Insufficient observations after dropping NaN values.")

        mu = float(data.mean())
        sigma = float(data.std(ddof=1))
        z = stats.norm.ppf(1.0 - confidence)
        var = -(mu + z * sigma)
        logger.debug("var_parametric(conf=%.2f): VaR=%.6f", confidence, var)
        return float(var)

    # ------------------------------------------------------------------
    # Historical VaR
    # ------------------------------------------------------------------

    def var_historical(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
    ) -> float:
        """Historical simulation Value-at-Risk.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        confidence : float, optional
            Confidence level in ``(0, 1)``. Default 0.99.

        Returns
        -------
        float
            VaR as a positive number.
        """
        self._validate_confidence(confidence)
        if returns.empty:
            raise ValueError("returns Series is empty.")

        data = returns.dropna().values
        if len(data) < 2:
            raise ValueError("Insufficient observations after dropping NaN values.")

        var = -float(np.percentile(data, (1.0 - confidence) * 100.0))
        logger.debug("var_historical(conf=%.2f): VaR=%.6f", confidence, var)
        return var

    # ------------------------------------------------------------------
    # Cornish-Fisher VaR
    # ------------------------------------------------------------------

    def var_cornish_fisher(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
    ) -> float:
        """Cornish-Fisher (modified) Value-at-Risk.

        Adjusts the Gaussian quantile for observed skewness and excess
        kurtosis, providing better tail estimates for non-normal return
        distributions.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        confidence : float, optional
            Confidence level in ``(0, 1)``. Default 0.99.

        Returns
        -------
        float
            Modified VaR as a positive number.
        """
        self._validate_confidence(confidence)
        if returns.empty:
            raise ValueError("returns Series is empty.")

        data = returns.dropna().values
        if len(data) < 4:
            raise ValueError("At least 4 observations required for Cornish-Fisher expansion.")

        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        skew = float(stats.skew(data))
        kurt = float(stats.kurtosis(data))  # excess kurtosis

        z = stats.norm.ppf(1.0 - confidence)
        # Cornish-Fisher adjusted quantile
        z_cf = (
            z
            + (z ** 2 - 1) * skew / 6.0
            + (z ** 3 - 3 * z) * kurt / 24.0
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36.0
        )
        var = -(mu + z_cf * sigma)
        logger.debug(
            "var_cornish_fisher(conf=%.2f): z_cf=%.4f, VaR=%.6f", confidence, z_cf, var
        )
        return float(var)

    # ------------------------------------------------------------------
    # Monte Carlo VaR
    # ------------------------------------------------------------------

    def var_monte_carlo(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        n_simulations: int = 10_000,
    ) -> float:
        """Monte Carlo Value-at-Risk via Gaussian simulation.

        Fits a normal distribution to the return history and draws
        ``n_simulations`` samples to estimate the loss quantile.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        confidence : float, optional
            Confidence level in ``(0, 1)``. Default 0.99.
        n_simulations : int, optional
            Number of Monte Carlo draws. Default 10 000.

        Returns
        -------
        float
            Monte Carlo VaR as a positive number.
        """
        self._validate_confidence(confidence)
        if returns.empty:
            raise ValueError("returns Series is empty.")

        data = returns.dropna().values
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))

        rng = np.random.default_rng(seed=42)
        simulated = rng.normal(mu, sigma, n_simulations)
        var = -float(np.percentile(simulated, (1.0 - confidence) * 100.0))
        logger.debug("var_monte_carlo(conf=%.2f, n=%d): VaR=%.6f", confidence, n_simulations, var)
        return var

    # ------------------------------------------------------------------
    # CVaR / Expected Shortfall
    # ------------------------------------------------------------------

    def cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        method: str = "historical",
    ) -> float:
        """Conditional Value-at-Risk (Expected Shortfall).

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns.
        confidence : float, optional
            Confidence level in ``(0, 1)``. Default 0.99.
        method : {'historical', 'parametric', 'monte_carlo'}, optional
            Estimation method. Default ``'historical'``.

        Returns
        -------
        float
            CVaR as a positive number (expected loss beyond VaR threshold).

        Raises
        ------
        ValueError
            If ``method`` is not recognised or inputs are invalid.
        """
        self._validate_confidence(confidence)
        valid_methods = {"historical", "parametric", "monte_carlo"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'.")
        if returns.empty:
            raise ValueError("returns Series is empty.")

        data = returns.dropna().values
        if len(data) < 2:
            raise ValueError("Insufficient observations.")

        if method == "historical":
            threshold = np.percentile(data, (1.0 - confidence) * 100.0)
            tail = data[data <= threshold]
            cvar_val = -float(np.mean(tail)) if len(tail) > 0 else self.var_historical(returns, confidence)
        elif method == "parametric":
            mu = float(np.mean(data))
            sigma = float(np.std(data, ddof=1))
            z = stats.norm.ppf(1.0 - confidence)
            cvar_val = -(mu - sigma * stats.norm.pdf(z) / (1.0 - confidence))
        else:  # monte_carlo
            rng = np.random.default_rng(seed=42)
            sim = rng.normal(np.mean(data), np.std(data, ddof=1), 100_000)
            threshold = np.percentile(sim, (1.0 - confidence) * 100.0)
            tail = sim[sim <= threshold]
            cvar_val = -float(np.mean(tail)) if len(tail) > 0 else self.var_monte_carlo(returns, confidence)

        logger.debug("cvar(method=%s, conf=%.2f): CVaR=%.6f", method, confidence, cvar_val)
        return float(cvar_val)

    # ------------------------------------------------------------------
    # Maximum Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Compute maximum peak-to-trough drawdown.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity / NAV time series.

        Returns
        -------
        float
            Maximum drawdown as a positive fraction (e.g., 0.35 means -35%).

        Raises
        ------
        ValueError
            If ``equity_curve`` is empty or contains non-positive values.
        """
        if equity_curve.empty:
            raise ValueError("equity_curve is empty.")
        if (equity_curve <= 0).any():
            raise ValueError("equity_curve must contain strictly positive values.")

        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        mdd = float(-drawdowns.min())
        logger.debug("max_drawdown: MDD=%.4f", mdd)
        return mdd

    # ------------------------------------------------------------------
    # Drawdown DataFrame
    # ------------------------------------------------------------------

    def compute_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Compute full drawdown time series with start and end dates.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity / NAV time series with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with columns:
            ``['drawdown', 'peak', 'trough', 'drawdown_start', 'recovery_date']``.
            ``drawdown`` is a non-positive fraction.
        """
        if equity_curve.empty:
            raise ValueError("equity_curve is empty.")
        if (equity_curve <= 0).any():
            raise ValueError("equity_curve must contain strictly positive values.")

        ec = equity_curve.copy()
        rolling_max = ec.cummax()
        drawdown = (ec - rolling_max) / rolling_max

        result = pd.DataFrame(
            {
                "drawdown": drawdown,
                "peak": rolling_max,
                "trough": ec,
            },
            index=ec.index,
        )

        # Determine drawdown_start for each observation
        peak_idx = rolling_max.values
        start_dates = []
        for i, val in enumerate(ec.values):
            # Most recent date at which rolling max equals the current running peak
            pk = peak_idx[i]
            subset = ec.iloc[: i + 1]
            peak_positions = subset[subset == pk].index
            start_dates.append(peak_positions[-1] if len(peak_positions) else ec.index[0])
        result["drawdown_start"] = start_dates

        # Recovery date: next date equity returns to prior peak (NaT if not yet recovered)
        recovery_dates = []
        for i in range(len(ec)):
            pk = peak_idx[i]
            future = ec.iloc[i:]
            recovered = future[future >= pk]
            recovery_dates.append(recovered.index[0] if len(recovered) > 0 else pd.NaT)
        result["recovery_date"] = recovery_dates

        logger.debug("compute_drawdowns: %d observations processed", len(result))
        return result

    # ------------------------------------------------------------------
    # Position Limits
    # ------------------------------------------------------------------

    def check_position_limits(
        self,
        positions: pd.Series,
        max_position: float = 0.10,
    ) -> bool:
        """Check whether all positions are within the allowed size limit.

        Parameters
        ----------
        positions : pd.Series
            Portfolio weight series (values should be in ``[0, 1]``).
        max_position : float, optional
            Maximum allowed weight per asset. Default 0.10.

        Returns
        -------
        bool
            ``True`` if all positions are within limits, ``False`` otherwise.
        """
        if positions.empty:
            logger.warning("check_position_limits: empty positions series.")
            return True

        breaches = positions[positions.abs() > max_position]
        if not breaches.empty:
            logger.warning(
                "check_position_limits: %d breach(es) found. Max=%.4f, limit=%.4f",
                len(breaches),
                float(positions.abs().max()),
                max_position,
            )
            return False

        logger.debug("check_position_limits: all positions within limit %.4f", max_position)
        return True

    # ------------------------------------------------------------------
    # Stress Testing
    # ------------------------------------------------------------------

    def stress_test(
        self,
        returns: pd.DataFrame,
        scenario: str = "2008_crisis",
    ) -> pd.Series:
        """Apply a named historical stress scenario to a multi-asset return DataFrame.

        Extracts the sub-period corresponding to the chosen scenario and computes
        the portfolio return under equal-weight allocation (as a proxy).
        If the returns DataFrame does not span the scenario period, a synthetic
        scenario is applied by scaling return statistics to match the scenario's
        historical severity.

        Parameters
        ----------
        returns : pd.DataFrame
            Multi-asset daily returns with a DatetimeIndex.
        scenario : {'2008_crisis', 'covid', 'dot_com'}, optional
            Named scenario identifier. Default ``'2008_crisis'``.

        Returns
        -------
        pd.Series
            Daily equal-weight portfolio returns during the scenario period (or
            synthetic stress returns if historical data is unavailable).

        Raises
        ------
        ValueError
            If ``scenario`` is not a recognised identifier.
        """
        if scenario not in self._stress_scenarios:
            raise ValueError(
                f"scenario must be one of {list(self._stress_scenarios.keys())}, got '{scenario}'."
            )

        scenario_info = self._stress_scenarios[scenario]
        start = pd.Timestamp(scenario_info["start"])
        end = pd.Timestamp(scenario_info["end"])

        eq_weights = np.ones(returns.shape[1]) / returns.shape[1]

        if isinstance(returns.index, pd.DatetimeIndex):
            mask = (returns.index >= start) & (returns.index <= end)
            scenario_data = returns.loc[mask]
        else:
            scenario_data = pd.DataFrame()

        if not scenario_data.empty:
            port_returns = scenario_data.values @ eq_weights
            result = pd.Series(port_returns, index=scenario_data.index, name=scenario)
            logger.info(
                "stress_test('%s'): %d days, cumulative return=%.4f",
                scenario,
                len(result),
                float((1 + result).prod() - 1),
            )
            return result

        # Synthetic stress: scale portfolio returns to mimic scenario severity
        severity_map = {
            "2008_crisis": {"scale": 3.5, "drift": -0.0025},
            "covid": {"scale": 5.0, "drift": -0.005},
            "dot_com": {"scale": 2.5, "drift": -0.0015},
        }
        params = severity_map[scenario]
        data = returns.dropna().values @ eq_weights
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))

        n_days = (end - start).days
        rng = np.random.default_rng(seed=0)
        synthetic = rng.normal(
            params["drift"], sigma * params["scale"], n_days
        )
        idx = pd.date_range(start, periods=n_days, freq="B")[:len(synthetic)]
        result = pd.Series(synthetic[: len(idx)], index=idx, name=scenario)
        logger.warning(
            "stress_test('%s'): historical data unavailable; using synthetic stress with %d days.",
            scenario,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Greeks Risk Decomposition
    # ------------------------------------------------------------------

    def greeks_risk_decomposition(
        self,
        portfolio_greeks: Dict[str, float],
    ) -> pd.DataFrame:
        """Decompose option portfolio risk into Greek sensitivities.

        Parameters
        ----------
        portfolio_greeks : dict of str to float
            Mapping of Greek name to its aggregate value. Expected keys include
            any subset of ``{'delta', 'gamma', 'theta', 'vega', 'rho'}``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['greek', 'value', 'risk_type', 'description']``
            and one row per Greek supplied.

        Raises
        ------
        ValueError
            If ``portfolio_greeks`` is empty.
        """
        if not portfolio_greeks:
            raise ValueError("portfolio_greeks must not be empty.")

        greek_meta: Dict[str, Dict[str, str]] = {
            "delta": {
                "risk_type": "directional",
                "description": "Sensitivity to 1% move in underlying price",
            },
            "gamma": {
                "risk_type": "convexity",
                "description": "Rate of change of delta per 1% move in underlying",
            },
            "theta": {
                "risk_type": "time_decay",
                "description": "Daily P&L decay from passage of time",
            },
            "vega": {
                "risk_type": "volatility",
                "description": "Sensitivity to 1% change in implied volatility",
            },
            "rho": {
                "risk_type": "interest_rate",
                "description": "Sensitivity to 1% change in risk-free rate",
            },
        }

        records = []
        for greek, value in portfolio_greeks.items():
            meta = greek_meta.get(
                greek.lower(),
                {"risk_type": "other", "description": f"Custom Greek: {greek}"},
            )
            records.append(
                {
                    "greek": greek,
                    "value": float(value),
                    "risk_type": meta["risk_type"],
                    "description": meta["description"],
                }
            )

        df = pd.DataFrame(records)
        logger.info("greeks_risk_decomposition: decomposed %d Greeks", len(df))
        return df

    # ------------------------------------------------------------------
    # Correlation Regime Monitor
    # ------------------------------------------------------------------

    def monitor_correlation_regime(
        self,
        returns: pd.DataFrame,
        window: int = 63,
    ) -> float:
        """Compute rolling average pairwise correlation as a regime indicator.

        A high average correlation (> ~0.5) typically signals a risk-off or
        crisis regime where diversification benefits break down.

        Parameters
        ----------
        returns : pd.DataFrame
            Multi-asset daily returns, shape ``(T, N)``.
        window : int, optional
            Look-back window in trading days. Default 63 (~1 quarter).

        Returns
        -------
        float
            Average pairwise Pearson correlation over the most-recent ``window``
            observations. Returns ``np.nan`` if insufficient data.
        """
        if returns.empty or returns.shape[1] < 2:
            logger.warning("monitor_correlation_regime: need at least 2 assets.")
            return float("nan")

        data = returns.dropna().iloc[-window:] if len(returns) >= window else returns.dropna()
        if len(data) < 2:
            logger.warning("monitor_correlation_regime: insufficient observations.")
            return float("nan")

        corr_matrix = data.corr().values
        n = corr_matrix.shape[0]
        # Extract upper-triangle off-diagonal elements
        idx = np.triu_indices(n, k=1)
        pairwise = corr_matrix[idx]
        avg_corr = float(np.nanmean(pairwise))
        logger.debug(
            "monitor_correlation_regime(window=%d): avg_corr=%.4f", window, avg_corr
        )
        return avg_corr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Raise ``ValueError`` if ``confidence`` is not in ``(0, 1)``."""
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
