"""
Abstract base class for all QuantVortex trading strategies.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Parameters
    ----------
    name : str
        Human-readable identifier for the strategy.
    allocation : float, optional
        Fraction of portfolio capital allocated to this strategy.
        Must be in (0, 1]. Default is 1.0.

    Attributes
    ----------
    name : str
        Strategy name.
    allocation : float
        Capital allocation fraction.
    logger : logging.Logger
        Strategy-level logger.
    positions : pd.Series
        Most recent computed positions, keyed by asset symbol.
    signals : pd.DataFrame
        Most recent computed signals.
    """

    def __init__(self, name: str, allocation: float = 1.0) -> None:
        if not 0 < allocation <= 1.0:
            raise ValueError(f"allocation must be in (0, 1], got {allocation}")
        self.name: str = name
        self.allocation: float = allocation
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.positions: pd.Series = pd.Series(dtype=float)
        self.signals: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data.

        Parameters
        ----------
        data : pd.DataFrame
            Market data with DatetimeIndex rows and asset columns
            (or a MultiIndex column structure).

        Returns
        -------
        pd.DataFrame
            Signal values aligned to ``data``'s index.
        """

    @abstractmethod
    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Compute target positions from signals.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals`.

        Returns
        -------
        pd.DataFrame
            Target position weights aligned to ``signals``'s index.
        """

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline.

        Chains :meth:`generate_signals` → :meth:`compute_positions`,
        stores results on the instance, and applies the allocation scalar.

        Parameters
        ----------
        data : pd.DataFrame
            Raw market data passed directly to :meth:`generate_signals`.

        Returns
        -------
        pd.DataFrame
            Allocation-scaled target positions.

        Raises
        ------
        RuntimeError
            If either pipeline step raises an unexpected exception.
        """
        self.logger.info("Running strategy '%s'", self.name)
        try:
            self.signals = self.generate_signals(data)
            positions = self.compute_positions(self.signals)
            positions = positions * self.allocation
            # Cache the latest cross-sectional position vector
            if not positions.empty:
                self.positions = positions.iloc[-1].dropna()
            self.logger.info(
                "Strategy '%s' produced %d position rows", self.name, len(positions)
            )
            return positions
        except Exception as exc:
            self.logger.exception("Strategy '%s' pipeline failed: %s", self.name, exc)
            raise RuntimeError(
                f"Strategy '{self.name}' pipeline failed"
            ) from exc

    # ------------------------------------------------------------------
    # Utility helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_price_df(data: pd.DataFrame, min_rows: int = 30) -> None:
        """Raise ValueError for obviously invalid price DataFrames.

        Parameters
        ----------
        data : pd.DataFrame
            Price data to validate.
        min_rows : int, optional
            Minimum required rows. Default is 30.

        Raises
        ------
        ValueError
            If ``data`` is empty, has fewer than ``min_rows`` rows, or
            contains only NaN values.
        """
        if data is None or data.empty:
            raise ValueError("Price data must be a non-empty DataFrame.")
        if len(data) < min_rows:
            raise ValueError(
                f"Price data has only {len(data)} rows; need at least {min_rows}."
            )
        if data.isnull().all().all():
            raise ValueError("Price data contains only NaN values.")

    @staticmethod
    def _zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """Compute rolling or full-sample z-score.

        Parameters
        ----------
        series : pd.Series
            Input time series.
        window : int or None, optional
            If provided, compute a rolling z-score with this lookback window.
            If ``None``, compute the full-sample z-score. Default is ``None``.

        Returns
        -------
        pd.Series
            Z-scored series with the same index as ``series``.
        """
        if window is not None:
            mu = series.rolling(window).mean()
            sd = series.rolling(window).std()
        else:
            mu = series.mean()
            sd = series.std()
        sd = sd.replace(0, np.nan) if isinstance(sd, pd.Series) else (sd if sd != 0 else np.nan)
        return (series - mu) / sd
