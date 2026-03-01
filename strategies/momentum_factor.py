"""
Cross-sectional momentum (Jegadeesh-Titman) strategy for QuantVortex.

Computes momentum scores across a universe of assets, applies optional
sector neutralisation and volatility scaling, then takes equal-weighted
long/short positions in the top/bottom N assets.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumFactor(BaseStrategy):
    """Cross-sectional momentum strategy (Jegadeesh-Titman style).

    Parameters
    ----------
    name : str, optional
        Strategy name. Default is ``"MomentumFactor"``.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    lookback : int, optional
        Momentum signal lookback window in trading days. Default is 126.
    skip_last : int, optional
        Number of most-recent days to skip (avoids short-term reversal).
        Default is 21.
    long_n : int, optional
        Number of top assets in the long book. Default is 5.
    short_n : int, optional
        Number of bottom assets in the short book. Default is 5.
    vol_window : int, optional
        Rolling volatility window for vol-scaling. Default is 21.
    sector_neutral : bool, optional
        Whether to demean scores within each sector. Default is ``False``.
    vol_scale : bool, optional
        Whether to divide scores by rolling volatility. Default is ``False``.
    sectors : dict, optional
        Mapping of ``{ticker: sector_label}``. Required when
        ``sector_neutral=True``. Default is ``None``.

    Attributes
    ----------
    sectors : dict
        Ticker → sector mapping.
    """

    def __init__(
        self,
        name: str = "MomentumFactor",
        allocation: float = 1.0,
        lookback: int = 126,
        skip_last: int = 21,
        long_n: int = 5,
        short_n: int = 5,
        vol_window: int = 21,
        sector_neutral: bool = False,
        vol_scale: bool = False,
        sectors: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        if skip_last >= lookback:
            raise ValueError(
                f"skip_last ({skip_last}) must be less than lookback ({lookback})."
            )
        self.lookback = lookback
        self.skip_last = skip_last
        self.long_n = long_n
        self.short_n = short_n
        self.vol_window = vol_window
        self.sector_neutral = sector_neutral
        self.vol_scale = vol_scale
        self.sectors: Dict[str, str] = sectors or {}

    # ------------------------------------------------------------------
    # Core score computation
    # ------------------------------------------------------------------

    def compute_momentum_scores(
        self,
        returns: pd.DataFrame,
        lookback: Optional[int] = None,
        skip_last: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute Jegadeesh-Titman momentum scores for each date.

        Score for asset *i* at date *t* is the cumulative return over
        ``[t - lookback, t - skip_last]`` (inclusive).

        Parameters
        ----------
        returns : pd.DataFrame
            Simple or log returns with DatetimeIndex rows and ticker columns.
        lookback : int or None, optional
            Override ``self.lookback``. Default is ``None``.
        skip_last : int or None, optional
            Override ``self.skip_last``. Default is ``None``.

        Returns
        -------
        pd.DataFrame
            Momentum scores (same shape as ``returns``).  Rows where
            insufficient history exists are ``NaN``.
        """
        lb = lookback if lookback is not None else self.lookback
        sl = skip_last if skip_last is not None else self.skip_last

        if returns.empty:
            raise ValueError("returns DataFrame is empty.")

        scores = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for t in range(lb, len(returns)):
            window = returns.iloc[t - lb : t - sl]
            if window.empty:
                continue
            scores.iloc[t] = window.sum(axis=0)

        self.logger.debug("compute_momentum_scores: computed for %d dates.", len(scores))
        return scores

    # ------------------------------------------------------------------
    # Score adjustments
    # ------------------------------------------------------------------

    def sector_neutral_momentum(
        self,
        scores: pd.Series,
        sectors: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """Demean momentum scores within each sector.

        Parameters
        ----------
        scores : pd.Series
            Cross-sectional momentum scores for a single date, indexed by
            ticker.
        sectors : dict or None, optional
            Override ``self.sectors``.  Default is ``None``.

        Returns
        -------
        pd.Series
            Sector-demeaned scores.  Tickers not present in ``sectors``
            are left unchanged.
        """
        sec_map = sectors if sectors is not None else self.sectors
        if not sec_map:
            self.logger.warning(
                "sector_neutral_momentum called but no sector map provided; "
                "returning raw scores."
            )
            return scores

        adjusted = scores.copy().astype(float)
        sector_series = pd.Series(sec_map)
        common = scores.index.intersection(sector_series.index)

        for sector in sector_series.loc[common].unique():
            tickers_in_sector = sector_series[sector_series == sector].index
            mask = common.intersection(tickers_in_sector)
            if len(mask) < 2:
                continue
            sector_mean = adjusted.loc[mask].mean()
            adjusted.loc[mask] -= sector_mean

        return adjusted

    def volatility_scaled_momentum(
        self,
        scores: pd.DataFrame,
        returns: pd.DataFrame,
        vol_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Divide momentum scores by rolling realized volatility.

        Parameters
        ----------
        scores : pd.DataFrame
            Momentum scores (DatetimeIndex × tickers).
        returns : pd.DataFrame
            Return series used to estimate volatility.
        vol_window : int or None, optional
            Rolling window override. Default is ``None``.

        Returns
        -------
        pd.DataFrame
            Volatility-scaled momentum scores.  Zero or NaN vols are
            replaced with ``NaN`` to avoid division by zero.
        """
        vw = vol_window if vol_window is not None else self.vol_window
        rolling_vol = returns.rolling(vw).std()
        rolling_vol = rolling_vol.replace(0, np.nan)

        # Align indices
        common_idx = scores.index.intersection(rolling_vol.index)
        common_col = scores.columns.intersection(rolling_vol.columns)

        scaled = scores.loc[common_idx, common_col] / rolling_vol.loc[common_idx, common_col]
        self.logger.debug("volatility_scaled_momentum: scaled %d rows.", len(scaled))
        return scaled

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum signals from price data.

        Expects ``data`` to be a price DataFrame.  Converts internally to
        simple returns, then computes momentum scores with optional sector
        neutralisation and volatility scaling.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format adjusted closing prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Momentum scores per ticker per date.
        """
        self._validate_price_df(data, min_rows=self.lookback + 5)
        returns = data.pct_change().dropna(how="all")

        scores = self.compute_momentum_scores(returns)

        if self.sector_neutral and self.sectors:
            scores = scores.apply(
                lambda row: self.sector_neutral_momentum(row, self.sectors)
                if not row.isna().all()
                else row,
                axis=1,
            )

        if self.vol_scale:
            scores = self.volatility_scaled_momentum(scores, returns)

        self.logger.info(
            "generate_signals: shape=%s, non-null rows=%d",
            scores.shape,
            int(scores.notna().any(axis=1).sum()),
        )
        return scores

    def compute_positions(
        self,
        signals: pd.DataFrame,
        long_n: Optional[int] = None,
        short_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Construct equal-weight long/short portfolio from momentum signals.

        For each date, rank the cross-section of momentum scores and
        assign ``+1/long_n`` to the top ``long_n`` assets and
        ``-1/short_n`` to the bottom ``short_n`` assets.

        Parameters
        ----------
        signals : pd.DataFrame
            Momentum scores as returned by :meth:`generate_signals`.
        long_n : int or None, optional
            Override ``self.long_n``. Default is ``None``.
        short_n : int or None, optional
            Override ``self.short_n``. Default is ``None``.

        Returns
        -------
        pd.DataFrame
            Position weights in ``[-1, +1]`` range.
        """
        ln = long_n if long_n is not None else self.long_n
        sn = short_n if short_n is not None else self.short_n

        if signals.empty:
            self.logger.warning("compute_positions received empty signals.")
            return pd.DataFrame(index=signals.index)

        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date, row in signals.iterrows():
            valid = row.dropna()
            if len(valid) < ln + sn:
                continue
            ranked = valid.rank(ascending=True)
            n_assets = len(ranked)

            top_n = ranked.nlargest(ln).index
            bottom_n = ranked.nsmallest(sn).index

            positions.loc[date, top_n] = 1.0 / ln
            positions.loc[date, bottom_n] = -1.0 / sn

        self.logger.info(
            "compute_positions: non-zero rows=%d",
            int((positions != 0).any(axis=1).sum()),
        )
        return positions
