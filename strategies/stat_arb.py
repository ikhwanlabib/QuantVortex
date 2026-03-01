"""
Statistical arbitrage (pairs trading) strategy for QuantVortex.

Uses Engle-Granger cointegration tests to identify pairs, a manual
Kalman filter to track dynamic hedge ratios, and z-score thresholds
to generate mean-reversion signals.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StatArb(BaseStrategy):
    """Pairs trading strategy using Engle-Granger cointegration and Kalman filter.

    Parameters
    ----------
    name : str
        Strategy name.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    entry_z : float, optional
        Z-score threshold to open a position. Default is 2.0.
    exit_z : float, optional
        Z-score threshold to close a position. Default is 0.5.
    min_half_life : int, optional
        Minimum acceptable spread half-life in days. Default is 5.
    max_half_life : int, optional
        Maximum acceptable spread half-life in days. Default is 126.
    coint_pvalue : float, optional
        Maximum p-value for the cointegration test. Default is 0.05.
    kalman_delta : float, optional
        State-noise variance scaling for the Kalman filter. Default is 1e-4.
    zscore_window : int, optional
        Rolling window for z-score calculation. Default is 60.

    Attributes
    ----------
    pairs : list of tuple of str
        Detected cointegrated pairs.
    hedge_ratios : dict
        Latest hedge ratio per pair keyed by ``(leg1, leg2)``.
    """

    def __init__(
        self,
        name: str = "StatArb",
        allocation: float = 1.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        min_half_life: int = 5,
        max_half_life: int = 126,
        coint_pvalue: float = 0.05,
        kalman_delta: float = 1e-4,
        zscore_window: int = 60,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.coint_pvalue = coint_pvalue
        self.kalman_delta = kalman_delta
        self.zscore_window = zscore_window
        self.pairs: List[Tuple[str, str]] = []
        self.hedge_ratios: Dict[Tuple[str, str], pd.Series] = {}

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    def find_pairs(
        self,
        prices: pd.DataFrame,
        min_half_life: Optional[int] = None,
        max_half_life: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """Identify cointegrated pairs using the Engle-Granger test.

        For each candidate pair the method:

        1. Runs ``statsmodels.tsa.stattools.coint`` and keeps pairs whose
           p-value is below ``self.coint_pvalue``.
        2. Computes the OLS hedge ratio, constructs the spread, and filters
           on half-life bounds.

        Parameters
        ----------
        prices : pd.DataFrame
            Wide-format price DataFrame with tickers as columns and a
            DatetimeIndex.
        min_half_life : int or None, optional
            Override ``self.min_half_life``. Default is ``None``.
        max_half_life : int or None, optional
            Override ``self.max_half_life``. Default is ``None``.

        Returns
        -------
        list of tuple of str
            Pairs ``(ticker_a, ticker_b)`` passing all filters.
        """
        min_hl = min_half_life if min_half_life is not None else self.min_half_life
        max_hl = max_half_life if max_half_life is not None else self.max_half_life

        self._validate_price_df(prices)
        tickers = prices.columns.tolist()
        valid_pairs: List[Tuple[str, str]] = []

        for i, a in enumerate(tickers):
            for b in tickers[i + 1 :]:
                series_a = prices[a].dropna()
                series_b = prices[b].dropna()
                common = series_a.index.intersection(series_b.index)
                if len(common) < 60:
                    continue
                try:
                    _, pval, _ = coint(series_a.loc[common], series_b.loc[common])
                except Exception as exc:
                    self.logger.debug("coint(%s,%s) failed: %s", a, b, exc)
                    continue

                if pval > self.coint_pvalue:
                    continue

                # Estimate static hedge ratio via OLS for half-life check
                x = sm.add_constant(series_b.loc[common])
                ols = sm.OLS(series_a.loc[common], x).fit()
                beta = ols.params.iloc[-1]
                spread = series_a.loc[common] - beta * series_b.loc[common]
                hl = self.compute_half_life(spread)

                if min_hl <= hl <= max_hl:
                    valid_pairs.append((a, b))
                    self.logger.debug(
                        "Pair (%s, %s) accepted: p=%.4f, hl=%.1f", a, b, pval, hl
                    )

        self.logger.info("find_pairs found %d valid pairs.", len(valid_pairs))
        self.pairs = valid_pairs
        return valid_pairs

    # ------------------------------------------------------------------
    # Kalman filter hedge ratio
    # ------------------------------------------------------------------

    def estimate_hedge_ratio(
        self, price1: pd.Series, price2: pd.Series
    ) -> pd.Series:
        """Estimate a time-varying hedge ratio using a scalar Kalman filter.

        The observation model is::

            price1[t] = beta[t] * price2[t] + alpha + noise

        State transition is a random walk: ``beta[t] = beta[t-1] + w``.

        Parameters
        ----------
        price1 : pd.Series
            Dependent-leg price series (DatetimeIndex).
        price2 : pd.Series
            Independent-leg price series (DatetimeIndex).

        Returns
        -------
        pd.Series
            Time-varying hedge-ratio estimates aligned to the common index.

        Raises
        ------
        ValueError
            If the common index has fewer than 10 observations.
        """
        common = price1.index.intersection(price2.index)
        if len(common) < 10:
            raise ValueError(
                "Insufficient common observations for Kalman filter "
                f"({len(common)} found, need ≥ 10)."
            )
        y = price1.loc[common].values.astype(float)
        x = price2.loc[common].values.astype(float)

        # State: [beta, alpha] (2-dimensional)
        # Transition matrix F = I_2
        # Observation matrix H[t] = [x[t], 1]
        n = len(y)
        delta = self.kalman_delta

        # Process noise covariance (2×2 diagonal)
        Q = delta / (1 - delta) * np.eye(2)
        # Measurement noise variance (scalar)
        R = 1.0

        # Initialise
        beta_hat = np.zeros((n, 2))
        P = np.eye(2) * 10.0  # initial state covariance

        state = np.array([0.0, 0.0])  # [beta, alpha]

        betas = np.empty(n)
        for t in range(n):
            H = np.array([x[t], 1.0])

            # Predict
            state_pred = state  # F = I
            P_pred = P + Q

            # Innovation
            innovation = y[t] - H @ state_pred
            S = float(H @ P_pred @ H) + R  # scalar innovation variance

            # Kalman gain
            K = P_pred @ H / S  # shape (2,)

            # Update
            state = state_pred + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            betas[t] = state[0]

        return pd.Series(betas, index=common, name="hedge_ratio")

    # ------------------------------------------------------------------
    # Half-life estimation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_half_life(spread: pd.Series) -> float:
        """Estimate the mean-reversion half-life via OLS on the spread.

        Fits the AR(1) regression: ``Δspread[t] = λ * spread[t-1] + ε``
        and returns ``-ln(2) / λ``.

        Parameters
        ----------
        spread : pd.Series
            Stationary spread series.

        Returns
        -------
        float
            Half-life in the same time units as the spread index.
            Returns ``np.inf`` if the regression yields a non-negative ``λ``.
        """
        spread_clean = spread.dropna()
        if len(spread_clean) < 10:
            return np.inf

        lag = spread_clean.shift(1).dropna()
        delta = spread_clean.diff().dropna()
        common = lag.index.intersection(delta.index)

        X = sm.add_constant(lag.loc[common])
        try:
            result = sm.OLS(delta.loc[common], X).fit()
        except Exception:
            return np.inf

        lam = result.params.iloc[-1]
        if lam >= 0:
            return np.inf
        return float(-np.log(2) / lam)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate z-score entry/exit signals for all cointegrated pairs.

        If ``self.pairs`` is empty, :meth:`find_pairs` is called on ``data``
        first.  For each pair a dynamic hedge ratio is estimated via the
        Kalman filter, the spread is computed, and a rolling z-score is
        computed over ``self.zscore_window`` bars.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format price DataFrame (DatetimeIndex × ticker columns).

        Returns
        -------
        pd.DataFrame
            DataFrame with one column per pair (formatted as
            ``"LEG1_vs_LEG2"``), containing z-score values.
        """
        self._validate_price_df(data)

        if not self.pairs:
            self.logger.info("No pairs configured; running find_pairs.")
            self.find_pairs(data)

        if not self.pairs:
            self.logger.warning("No cointegrated pairs found; returning empty signals.")
            return pd.DataFrame(index=data.index)

        result: Dict[str, pd.Series] = {}

        for a, b in self.pairs:
            if a not in data.columns or b not in data.columns:
                self.logger.warning("Pair (%s, %s) not in data columns; skipping.", a, b)
                continue

            try:
                hedge = self.estimate_hedge_ratio(data[a], data[b])
            except Exception as exc:
                self.logger.error(
                    "Hedge ratio estimation failed for (%s, %s): %s", a, b, exc
                )
                continue

            self.hedge_ratios[(a, b)] = hedge
            spread = data[a] - hedge * data[b]
            zscore = self._zscore(spread.dropna(), window=self.zscore_window)
            result[f"{a}_vs_{b}"] = zscore

        signals = pd.DataFrame(result)
        self.logger.info("generate_signals produced %d pair columns.", signals.shape[1])
        return signals

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert z-score signals into long/short pair positions.

        Entry rule: open a position when ``|z| ≥ entry_z``.
        Exit rule:  close the position when ``|z| ≤ exit_z``.
        Position is ``-1`` (short spread) when z > entry_z, ``+1`` (long
        spread) when z < -entry_z.  Carries the position forward while
        ``exit_z < |z| < entry_z``.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals` (z-scores per pair column).

        Returns
        -------
        pd.DataFrame
            Integer positions (−1, 0, +1) per pair column aligned to
            ``signals``'s index.
        """
        if signals.empty:
            self.logger.warning("Received empty signals; returning empty positions.")
            return pd.DataFrame(index=signals.index)

        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for col in signals.columns:
            z = signals[col]
            pos = np.zeros(len(z))
            current = 0.0

            for t, zt in enumerate(z.values):
                if np.isnan(zt):
                    pos[t] = current
                    continue
                if current == 0:
                    if zt > self.entry_z:
                        current = -1.0  # spread too high → short spread
                    elif zt < -self.entry_z:
                        current = 1.0   # spread too low → long spread
                else:
                    if abs(zt) <= self.exit_z:
                        current = 0.0
                pos[t] = current

            positions[col] = pos

        return positions
