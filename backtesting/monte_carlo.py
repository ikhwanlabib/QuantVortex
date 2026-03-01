"""
Monte Carlo simulation framework for QuantVortex strategy validation.

Provides block bootstrap and IID bootstrap methods for constructing
confidence intervals, estimating ruin probability, and characterising
the distribution of drawdown outcomes.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from backtesting.performance import PerformanceAnalytics

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR: int = 252


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for strategy validation and risk estimation.

    All methods are deterministic given a fixed ``random_state``; set the
    instance's ``random_state`` attribute (or pass ``seed`` at construction)
    for reproducibility.

    Parameters
    ----------
    seed : int or None, optional
        Seed for the random number generator. If ``None`` (default) results
        are non-deterministic across runs.

    Examples
    --------
    >>> sim = MonteCarloSimulator(seed=42)
    >>> ci = sim.compute_sharpe_ci(returns, n_simulations=2000)
    >>> report = sim.run_full_analysis(returns)
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)
        logger.info("MonteCarloSimulator initialised | seed=%s", seed)

    # ------------------------------------------------------------------
    # Bootstrap sampling
    # ------------------------------------------------------------------

    def bootstrap_returns(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        block_size: int = 20,
    ) -> np.ndarray:
        """
        Generate simulated return paths using the block (moving-blocks) bootstrap.

        Block bootstrapping preserves short-range autocorrelation structure in
        the original series by resampling contiguous blocks of observations.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series (NaN values are dropped).
        n_simulations : int, optional
            Number of simulated paths to generate. Default is ``1000``.
        block_size : int, optional
            Length of each contiguous block in trading days. Default is ``20``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_simulations, len(returns))`` containing
            simulated daily return paths.

        Raises
        ------
        ValueError
            If ``block_size`` is greater than the length of ``returns``.
        """
        clean = returns.dropna().values.astype(float)
        n = len(clean)

        if n < 2:
            raise ValueError("returns must contain at least 2 non-NaN observations.")
        if block_size > n:
            raise ValueError(
                f"block_size ({block_size}) cannot exceed the length of returns ({n})."
            )
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}.")

        n_blocks = int(np.ceil(n / block_size))
        max_start = n - block_size  # inclusive upper bound for block starts

        simulated = np.empty((n_simulations, n), dtype=float)

        for i in range(n_simulations):
            starts = self.rng.integers(0, max_start + 1, size=n_blocks)
            path = np.concatenate([clean[s: s + block_size] for s in starts])
            simulated[i] = path[:n]  # trim to original length

        logger.debug(
            "bootstrap_returns: %d simulations, block_size=%d, n=%d",
            n_simulations,
            block_size,
            n,
        )
        return simulated

    def simple_bootstrap(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
    ) -> np.ndarray:
        """
        Generate simulated return paths using the IID (simple) bootstrap.

        Observations are drawn independently with replacement, so the
        temporal structure of the original series is not preserved.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series (NaN values are dropped).
        n_simulations : int, optional
            Number of simulated paths to generate. Default is ``1000``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_simulations, len(returns))`` containing
            simulated daily return paths.

        Raises
        ------
        ValueError
            If ``returns`` contains fewer than 2 non-NaN observations.
        """
        clean = returns.dropna().values.astype(float)
        n = len(clean)

        if n < 2:
            raise ValueError("returns must contain at least 2 non-NaN observations.")
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}.")

        indices = self.rng.integers(0, n, size=(n_simulations, n))
        simulated = clean[indices]

        logger.debug(
            "simple_bootstrap: %d simulations, n=%d", n_simulations, n
        )
        return simulated

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def compute_sharpe_ci(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        n_simulations: int = 1000,
    ) -> Tuple[float, float]:
        """
        Construct a bootstrap confidence interval for the annualized Sharpe ratio.

        Uses the block bootstrap to preserve autocorrelation.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series.
        confidence : float, optional
            Confidence level in (0, 1). Default is ``0.95``.
        n_simulations : int, optional
            Number of bootstrap replications. Default is ``1000``.

        Returns
        -------
        tuple of float
            ``(lower, upper)`` bounds of the confidence interval.

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0, 1).
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

        simulated = self.bootstrap_returns(returns, n_simulations=n_simulations)
        sharpes = np.array(
            [
                _bootstrap_sharpe(sim_path)
                for sim_path in simulated
            ]
        )
        # Remove NaN/Inf
        sharpes = sharpes[np.isfinite(sharpes)]

        alpha = 1.0 - confidence
        lower = float(np.percentile(sharpes, 100.0 * alpha / 2.0))
        upper = float(np.percentile(sharpes, 100.0 * (1.0 - alpha / 2.0)))

        logger.debug(
            "Sharpe CI (%.0f%%): [%.4f, %.4f]", 100 * confidence, lower, upper
        )
        return lower, upper

    def compute_return_ci(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        n_simulations: int = 1000,
    ) -> Tuple[float, float]:
        """
        Construct a bootstrap confidence interval for the cumulative return.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series.
        confidence : float, optional
            Confidence level in (0, 1). Default is ``0.95``.
        n_simulations : int, optional
            Number of bootstrap replications. Default is ``1000``.

        Returns
        -------
        tuple of float
            ``(lower, upper)`` bounds of the cumulative return confidence interval.
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

        simulated = self.bootstrap_returns(returns, n_simulations=n_simulations)
        cum_returns = np.array(
            [float(np.prod(1.0 + path) - 1.0) for path in simulated]
        )

        alpha = 1.0 - confidence
        lower = float(np.percentile(cum_returns, 100.0 * alpha / 2.0))
        upper = float(np.percentile(cum_returns, 100.0 * (1.0 - alpha / 2.0)))

        logger.debug(
            "Return CI (%.0f%%): [%.4f, %.4f]", 100 * confidence, lower, upper
        )
        return lower, upper

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def probability_of_ruin(
        self,
        returns: pd.Series,
        ruin_threshold: float = -0.5,
        n_simulations: int = 1000,
    ) -> float:
        """
        Estimate the probability of ruin via block bootstrap simulation.

        Ruin is defined as the cumulative portfolio return falling below
        ``ruin_threshold`` at any point during the simulated period.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series.
        ruin_threshold : float, optional
            Cumulative return level below which the portfolio is considered
            ruined. Default is ``-0.5`` (50% loss).
        n_simulations : int, optional
            Number of bootstrap replications. Default is ``1000``.

        Returns
        -------
        float
            Estimated probability of ruin in [0, 1].

        Raises
        ------
        ValueError
            If ``ruin_threshold`` is not negative.
        """
        if ruin_threshold >= 0.0:
            raise ValueError(
                f"ruin_threshold must be negative, got {ruin_threshold}."
            )

        simulated = self.bootstrap_returns(returns, n_simulations=n_simulations)
        ruin_count = 0

        for path in simulated:
            # Compute running cumulative return
            cum_equity = np.cumprod(1.0 + path)
            cum_return = cum_equity - 1.0
            if np.any(cum_return <= ruin_threshold):
                ruin_count += 1

        prob = ruin_count / n_simulations
        logger.debug(
            "Probability of ruin (threshold=%.2f): %.4f", ruin_threshold, prob
        )
        return prob

    def max_drawdown_distribution(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
    ) -> np.ndarray:
        """
        Compute the distribution of maximum drawdowns via block bootstrap.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series.
        n_simulations : int, optional
            Number of bootstrap replications. Default is ``1000``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_simulations,)`` containing the maximum
            drawdown (as a negative decimal) for each simulated path.
        """
        simulated = self.bootstrap_returns(returns, n_simulations=n_simulations)
        mdd_values = np.empty(n_simulations, dtype=float)

        for i, path in enumerate(simulated):
            equity = np.cumprod(1.0 + path)
            rolling_max = np.maximum.accumulate(equity)
            drawdowns = (equity - rolling_max) / rolling_max
            mdd_values[i] = float(drawdowns.min())

        logger.debug(
            "Max drawdown distribution: median=%.4f, 5th pct=%.4f",
            float(np.median(mdd_values)),
            float(np.percentile(mdd_values, 5)),
        )
        return mdd_values

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        ruin_threshold: float = -0.5,
    ) -> pd.DataFrame:
        """
        Run all Monte Carlo analyses and return a consolidated summary DataFrame.

        Parameters
        ----------
        returns : pd.Series
            Empirical daily returns series.
        n_simulations : int, optional
            Number of bootstrap replications for each analysis. Default is ``1000``.
        confidence : float, optional
            Confidence level for interval estimates. Default is ``0.95``.
        ruin_threshold : float, optional
            Cumulative return level defining ruin. Default is ``-0.5``.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame indexed by metric name with column
            ``value``.  Interval estimates are reported as separate
            ``lower`` / ``upper`` rows.

        Examples
        --------
        >>> sim = MonteCarloSimulator(seed=0)
        >>> report = sim.run_full_analysis(returns)
        >>> print(report)
        """
        logger.info(
            "run_full_analysis: n_simulations=%d confidence=%.2f ruin_threshold=%.2f",
            n_simulations,
            confidence,
            ruin_threshold,
        )

        results: dict[str, float] = {}

        # Empirical metrics
        pa = PerformanceAnalytics
        results["Empirical Sharpe Ratio"] = pa.sharpe_ratio(returns)
        results["Empirical CAGR"] = _safe_call(
            pa.cagr, _equity_from_returns(returns)
        )
        results["Empirical Max Drawdown"] = pa.max_drawdown(
            _equity_from_returns(returns)
        )
        results["Empirical Annualized Volatility"] = pa.annualized_volatility(returns)

        # Bootstrap confidence intervals
        sharpe_lo, sharpe_hi = self.compute_sharpe_ci(
            returns, confidence=confidence, n_simulations=n_simulations
        )
        results[f"Sharpe CI Lower ({int(confidence*100)}%)"] = sharpe_lo
        results[f"Sharpe CI Upper ({int(confidence*100)}%)"] = sharpe_hi

        ret_lo, ret_hi = self.compute_return_ci(
            returns, confidence=confidence, n_simulations=n_simulations
        )
        results[f"Cumulative Return CI Lower ({int(confidence*100)}%)"] = ret_lo
        results[f"Cumulative Return CI Upper ({int(confidence*100)}%)"] = ret_hi

        # Ruin probability
        results[f"Probability of Ruin (threshold={ruin_threshold:.0%})"] = (
            self.probability_of_ruin(
                returns,
                ruin_threshold=ruin_threshold,
                n_simulations=n_simulations,
            )
        )

        # Max drawdown distribution statistics
        mdd_dist = self.max_drawdown_distribution(returns, n_simulations=n_simulations)
        results["MDD Distribution Mean"] = float(np.mean(mdd_dist))
        results["MDD Distribution Median"] = float(np.median(mdd_dist))
        results["MDD Distribution 5th Percentile"] = float(np.percentile(mdd_dist, 5))
        results["MDD Distribution 95th Percentile"] = float(
            np.percentile(mdd_dist, 95)
        )

        report = pd.DataFrame.from_dict({"value": results}, orient="columns")
        logger.info("run_full_analysis complete: %d metrics computed.", len(report))
        return report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bootstrap_sharpe(returns_path: np.ndarray) -> float:
    """Compute annualized Sharpe for a single simulated return path."""
    std = float(returns_path.std(ddof=1))
    if std == 0.0:
        return np.nan
    return float(returns_path.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    """Convert a returns series to an equity curve starting at 1.0."""
    clean = returns.dropna()
    equity = (1.0 + clean).cumprod()
    # Prepend starting value of 1.0
    first_date = equity.index[0] - pd.Timedelta(days=1) if len(equity) else pd.Timestamp("1970-01-01")
    start = pd.Series([1.0], index=[first_date])
    return pd.concat([start, equity])


def _safe_call(fn, *args, **kwargs) -> float:
    """Call *fn* with *args*; return ``NaN`` on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("Monte Carlo metric failed for %s: %s", getattr(fn, "__name__", fn), exc)
        return np.nan
