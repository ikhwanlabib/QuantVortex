"""
Comprehensive performance analytics for QuantVortex.

Provides risk-adjusted return metrics, drawdown analysis, factor attribution,
and rolling statistics for evaluating backtested and live strategies.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR: int = 252


class PerformanceAnalytics:
    """
    Collection of performance and risk analytics for equity curves and returns.

    All methods are static and stateless; instantiate once and call as needed,
    or use the methods directly via the class.

    Examples
    --------
    >>> pa = PerformanceAnalytics()
    >>> returns = pa.compute_returns(equity_curve)
    >>> report = pa.full_report(equity_curve, benchmark=benchmark_series)
    """

    # ------------------------------------------------------------------
    # Return series
    # ------------------------------------------------------------------

    @staticmethod
    def compute_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Compute simple percentage returns from an equity curve.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity (NAV) indexed by date. Must have at least two
            non-NaN values.

        Returns
        -------
        pd.Series
            Period returns; first element is ``NaN``.

        Raises
        ------
        ValueError
            If ``equity_curve`` has fewer than two observations.
        """
        if len(equity_curve) < 2:
            raise ValueError("equity_curve must have at least 2 observations.")
        returns = equity_curve.pct_change()
        logger.debug("compute_returns: %d observations", len(returns.dropna()))
        return returns

    # ------------------------------------------------------------------
    # Return & growth metrics
    # ------------------------------------------------------------------

    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """
        Compute the Compound Annual Growth Rate (CAGR).

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity indexed by date.

        Returns
        -------
        float
            CAGR as a decimal (e.g. 0.12 for 12%).

        Raises
        ------
        ValueError
            If the equity curve spans less than one full trading day.
        """
        equity_curve = equity_curve.dropna()
        if len(equity_curve) < 2:
            raise ValueError("equity_curve must have at least 2 non-NaN values.")

        start_val = float(equity_curve.iloc[0])
        end_val = float(equity_curve.iloc[-1])

        if start_val <= 0:
            raise ValueError("equity_curve start value must be positive.")

        n_years = len(equity_curve) / _TRADING_DAYS_PER_YEAR
        if n_years <= 0:
            raise ValueError("equity_curve spans zero years.")

        result = (end_val / start_val) ** (1.0 / n_years) - 1.0
        logger.debug("CAGR=%.4f over %.2f years", result, n_years)
        return result

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """
        Compute annualized volatility (standard deviation of returns).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series (NaN values are dropped).

        Returns
        -------
        float
            Annualized volatility as a decimal.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return np.nan
        vol = float(clean.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        logger.debug("Annualized volatility=%.4f", vol)
        return vol

    # ------------------------------------------------------------------
    # Risk-adjusted return metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        """
        Compute the annualized Sharpe ratio.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        risk_free : float, optional
            Daily risk-free rate. Default is ``0.0``.

        Returns
        -------
        float
            Annualized Sharpe ratio. Returns ``NaN`` if volatility is zero.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return np.nan
        excess = clean - risk_free
        std = float(excess.std(ddof=1))
        if std == 0.0:
            return np.nan
        sharpe = float(excess.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))
        logger.debug("Sharpe ratio=%.4f", sharpe)
        return sharpe

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        """
        Compute the annualized Sortino ratio (downside deviation normalised).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        risk_free : float, optional
            Daily risk-free rate. Default is ``0.0``.

        Returns
        -------
        float
            Annualized Sortino ratio. Returns ``NaN`` if downside deviation is zero.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return np.nan
        excess = clean - risk_free
        downside = excess[excess < 0.0]
        if len(downside) == 0:
            return np.inf
        downside_std = float(downside.std(ddof=1))
        if downside_std == 0.0:
            return np.nan
        sortino = float(excess.mean() / downside_std * np.sqrt(_TRADING_DAYS_PER_YEAR))
        logger.debug("Sortino ratio=%.4f", sortino)
        return sortino

    @staticmethod
    def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
        """
        Compute the Calmar ratio (CAGR divided by maximum drawdown).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        equity_curve : pd.Series
            Portfolio equity indexed by date.

        Returns
        -------
        float
            Calmar ratio. Returns ``NaN`` if max drawdown is zero.
        """
        try:
            ann_return = PerformanceAnalytics.cagr(equity_curve)
        except ValueError:
            return np.nan
        mdd = PerformanceAnalytics.max_drawdown(equity_curve)
        if mdd == 0.0:
            return np.nan
        calmar = ann_return / abs(mdd)
        logger.debug("Calmar ratio=%.4f", calmar)
        return calmar

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Compute the Omega ratio.

        The Omega ratio is the ratio of the probability-weighted gains above
        a threshold to the probability-weighted losses below it.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        threshold : float, optional
            Minimum acceptable return. Default is ``0.0``.

        Returns
        -------
        float
            Omega ratio. Returns ``NaN`` if there are no losses.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        gains = float((clean[clean > threshold] - threshold).sum())
        losses = float((threshold - clean[clean <= threshold]).sum())
        if losses == 0.0:
            return np.inf
        omega = gains / losses
        logger.debug("Omega ratio=%.4f", omega)
        return omega

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute the maximum drawdown of an equity curve.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity indexed by date.

        Returns
        -------
        float
            Maximum drawdown as a negative decimal (e.g. ``-0.35`` for 35%).
        """
        equity = equity_curve.dropna()
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        mdd = float(drawdown.min())
        logger.debug("Max drawdown=%.4f", mdd)
        return mdd

    @staticmethod
    def drawdown_duration(equity_curve: pd.Series) -> int:
        """
        Compute the maximum drawdown duration in trading days.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity indexed by date.

        Returns
        -------
        int
            Longest number of consecutive days spent below a previous peak.
        """
        equity = equity_curve.dropna()
        if len(equity) < 2:
            return 0

        rolling_max = equity.cummax()
        in_drawdown = (equity < rolling_max).astype(int)

        max_duration = 0
        current_duration = 0
        for val in in_drawdown:
            if val == 1:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        logger.debug("Drawdown duration=%d days", max_duration)
        return max_duration

    # ------------------------------------------------------------------
    # Tail risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def var_95(returns: pd.Series) -> float:
        """
        Compute the historical 95% Value-at-Risk (VaR).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            VaR at the 95% confidence level as a negative decimal.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        var = float(np.percentile(clean, 5))
        logger.debug("VaR(95)=%.4f", var)
        return var

    @staticmethod
    def cvar_95(returns: pd.Series) -> float:
        """
        Compute the historical 95% Conditional Value-at-Risk (CVaR / Expected Shortfall).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            CVaR at the 95% confidence level as a negative decimal.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        var = np.percentile(clean, 5)
        cvar = float(clean[clean <= var].mean())
        logger.debug("CVaR(95)=%.4f", cvar)
        return cvar

    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        """
        Compute the tail ratio: abs(95th percentile) / abs(5th percentile).

        A value greater than 1 indicates the right tail is larger than the
        left tail (more extreme gains than losses).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            Tail ratio. Returns ``NaN`` if the 5th percentile is zero.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        p95 = float(np.percentile(clean, 95))
        p05 = float(np.percentile(clean, 5))
        if p05 == 0.0:
            return np.nan
        ratio = abs(p95) / abs(p05)
        logger.debug("Tail ratio=%.4f", ratio)
        return ratio

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Compute the fraction of positive return periods.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            Win rate in [0, 1].
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        rate = float((clean > 0).mean())
        logger.debug("Win rate=%.4f", rate)
        return rate

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Compute the profit factor: sum of gains / abs(sum of losses).

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            Profit factor. Returns ``inf`` if there are no losing periods.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return np.nan
        gains = float(clean[clean > 0].sum())
        losses = float(abs(clean[clean < 0].sum()))
        if losses == 0.0:
            return np.inf
        pf = gains / losses
        logger.debug("Profit factor=%.4f", pf)
        return pf

    # ------------------------------------------------------------------
    # Stability & regression-based metrics
    # ------------------------------------------------------------------

    @staticmethod
    def stability_of_returns(returns: pd.Series) -> float:
        """
        Compute the R² of an OLS regression of log(cumulative returns) on time.

        A value close to 1 indicates a very smooth, consistent equity curve.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        float
            R² in [0, 1], or ``NaN`` if computation fails.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return np.nan

        cum_log_returns = np.log1p(clean).cumsum().values
        t = np.arange(len(cum_log_returns)).reshape(-1, 1)

        try:
            model = LinearRegression().fit(t, cum_log_returns)
            r2 = float(model.score(t, cum_log_returns))
        except Exception as exc:  # pragma: no cover
            logger.warning("stability_of_returns OLS failed: %s", exc)
            return np.nan

        logger.debug("Stability of returns R²=%.4f", r2)
        return r2

    @staticmethod
    def compute_alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Tuple[float, float]:
        """
        Compute Jensen's alpha and market beta via OLS regression.

        Parameters
        ----------
        returns : pd.Series
            Strategy daily returns.
        benchmark_returns : pd.Series
            Benchmark daily returns (e.g., S&P 500).

        Returns
        -------
        tuple of float
            ``(alpha, beta)`` where alpha is annualised and beta is the
            OLS slope coefficient.

        Raises
        ------
        ValueError
            If the aligned series have fewer than 2 observations.
        """
        aligned = pd.concat(
            [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()

        if len(aligned) < 2:
            raise ValueError(
                "Fewer than 2 overlapping observations for alpha/beta computation."
            )

        x = aligned["benchmark"].values.reshape(-1, 1)
        y = aligned["strategy"].values

        model = LinearRegression().fit(x, y)
        beta = float(model.coef_[0])
        daily_alpha = float(model.intercept_)
        annualized_alpha = daily_alpha * _TRADING_DAYS_PER_YEAR

        logger.debug("Alpha=%.4f Beta=%.4f", annualized_alpha, beta)
        return annualized_alpha, beta

    @staticmethod
    def compute_factor_loadings(
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate factor loadings via OLS regression.

        Parameters
        ----------
        returns : pd.Series
            Strategy daily returns.
        factors : pd.DataFrame
            DataFrame of daily factor returns, one column per factor.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``factor``, ``loading``, ``t_stat``,
            ``p_value``, and ``r_squared``.

        Raises
        ------
        ValueError
            If there are fewer than ``len(factors.columns) + 2`` observations.
        """
        aligned = pd.concat(
            [returns.rename("returns"), factors],
            axis=1,
        ).dropna()

        min_obs = factors.shape[1] + 2
        if len(aligned) < min_obs:
            raise ValueError(
                f"Insufficient observations ({len(aligned)}) for {factors.shape[1]} factors. "
                f"Need at least {min_obs}."
            )

        y = aligned["returns"].values
        x = aligned[factors.columns].values
        n, k = x.shape

        model = LinearRegression().fit(x, y)
        r2 = float(model.score(x, y))
        y_pred = model.predict(x)
        residuals = y - y_pred
        mse = float((residuals ** 2).sum() / (n - k - 1))

        try:
            xtx_inv = np.linalg.inv(x.T @ x)
            se = np.sqrt(np.diag(mse * xtx_inv))
            t_stats = model.coef_ / se
            p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in factor loading computation; SE unavailable.")
            se = np.full(k, np.nan)
            t_stats = np.full(k, np.nan)
            p_values = np.full(k, np.nan)

        records = [
            {
                "factor": factor_name,
                "loading": float(model.coef_[i]),
                "t_stat": float(t_stats[i]),
                "p_value": float(p_values[i]),
                "r_squared": r2,
            }
            for i, factor_name in enumerate(factors.columns)
        ]
        result = pd.DataFrame(records).set_index("factor")
        logger.debug("Factor loadings computed for %d factors; R²=%.4f", k, r2)
        return result

    # ------------------------------------------------------------------
    # Rolling analytics
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Compute the rolling annualized Sharpe ratio.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        window : int, optional
            Rolling window in trading days. Default is ``252``.

        Returns
        -------
        pd.Series
            Rolling Sharpe ratio, with ``NaN`` for the first ``window - 1`` entries.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        roll_mean = returns.rolling(window=window).mean()
        roll_std = returns.rolling(window=window).std(ddof=1)
        rolling = (roll_mean / roll_std) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        logger.debug("Rolling Sharpe computed (window=%d)", window)
        return rolling

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    @staticmethod
    def full_report(
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive performance report in a single DataFrame.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity (NAV) indexed by date.
        benchmark : pd.Series, optional
            Benchmark equity curve. If provided, alpha and beta are included.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame indexed by metric name, with column
            ``value``.

        Examples
        --------
        >>> report = PerformanceAnalytics.full_report(equity_curve, benchmark)
        >>> print(report)
        """
        pa = PerformanceAnalytics
        returns = pa.compute_returns(equity_curve)

        metrics: dict[str, float] = {}

        # Growth
        metrics["CAGR"] = _safe(pa.cagr, equity_curve)
        metrics["Total Return"] = (
            float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
            if len(equity_curve) >= 2 and equity_curve.iloc[0] != 0
            else np.nan
        )

        # Volatility & risk-adjusted
        metrics["Annualized Volatility"] = pa.annualized_volatility(returns)
        metrics["Sharpe Ratio"] = pa.sharpe_ratio(returns)
        metrics["Sortino Ratio"] = pa.sortino_ratio(returns)
        metrics["Calmar Ratio"] = _safe(pa.calmar_ratio, returns, equity_curve)
        metrics["Omega Ratio"] = pa.omega_ratio(returns)

        # Drawdown
        metrics["Max Drawdown"] = pa.max_drawdown(equity_curve)
        metrics["Max Drawdown Duration (days)"] = float(
            pa.drawdown_duration(equity_curve)
        )

        # Tail risk
        metrics["VaR 95%"] = pa.var_95(returns)
        metrics["CVaR 95%"] = pa.cvar_95(returns)
        metrics["Tail Ratio"] = pa.tail_ratio(returns)

        # Trade-level
        metrics["Win Rate"] = pa.win_rate(returns)
        metrics["Profit Factor"] = pa.profit_factor(returns)

        # Stability
        metrics["Stability of Returns (R²)"] = pa.stability_of_returns(returns)

        # Alpha / Beta
        if benchmark is not None:
            bm_returns = pa.compute_returns(benchmark)
            try:
                alpha, beta = pa.compute_alpha_beta(returns, bm_returns)
                metrics["Alpha (annualized)"] = alpha
                metrics["Beta"] = beta
            except ValueError as exc:
                logger.warning("Alpha/beta skipped: %s", exc)
                metrics["Alpha (annualized)"] = np.nan
                metrics["Beta"] = np.nan

        report = pd.DataFrame.from_dict(
            {"value": metrics}, orient="columns"
        )
        logger.info("Full report generated with %d metrics.", len(report))
        return report


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _safe(fn, *args, **kwargs) -> float:
    """Call *fn* with *args*; return ``NaN`` on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("Metric computation failed for %s: %s", fn.__name__, exc)
        return np.nan
