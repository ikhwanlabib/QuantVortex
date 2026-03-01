"""
data/preprocessor.py
====================
Data cleaning and feature engineering for QuantVortex.

All technical indicators are implemented from scratch using pandas / NumPy so
there is no external TA library dependency.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ReturnMethod = Literal["log", "simple"]
NormalizeMethod = Literal["zscore", "minmax"]


class DataPreprocessor:
    """Clean prices, compute returns, and engineer technical features.

    All public methods are pure (no in-place mutation of the caller's
    DataFrame) and return a new :class:`~pandas.DataFrame`.

    Examples
    --------
    >>> prep = DataPreprocessor()
    >>> clean = prep.clean_prices(raw_df)
    >>> returns = prep.compute_returns(clean)
    >>> enriched = prep.add_technical_indicators(clean)
    """

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill gaps and remove rows that are entirely NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame (columns: Open, High, Low, Close, Volume).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with the same column structure.

        Notes
        -----
        * Rows where **all** values are NaN are dropped first.
        * A forward-fill (``ffill``) propagates the most recent valid value
          into gaps (e.g. holidays, suspensions).
        * A subsequent backward-fill handles any leading NaN rows.

        Examples
        --------
        >>> prep = DataPreprocessor()
        >>> clean = prep.clean_prices(df)
        """
        if df is None or df.empty:
            logger.warning("clean_prices received an empty DataFrame.")
            return df

        original_len = len(df)
        result = df.copy()

        # Drop rows where *every* column is NaN.
        result.dropna(how="all", inplace=True)

        # Forward-fill then backward-fill to handle interior and leading gaps.
        result = result.ffill().bfill()

        dropped = original_len - len(result)
        logger.info(
            "clean_prices | dropped %d all-NaN rows, ffill/bfill applied.", dropped
        )
        return result

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        prices: pd.DataFrame,
        method: ReturnMethod = "log",
    ) -> pd.DataFrame:
        """Compute period-over-period returns from a price series.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame whose columns are price series (typically the ``Close``
            column of an OHLCV frame, or a wide DataFrame of close prices for
            multiple tickers).
        method : {"log", "simple"}, optional
            ``"log"``   – natural-log returns ``ln(P_t / P_{t-1})``  (default).
            ``"simple"`` – arithmetic returns ``(P_t - P_{t-1}) / P_{t-1}``.

        Returns
        -------
        pd.DataFrame
            Returns DataFrame with the same columns; first row is NaN and is
            dropped.

        Raises
        ------
        ValueError
            If *method* is not one of ``{"log", "simple"}``.

        Examples
        --------
        >>> returns = prep.compute_returns(df[["Close"]], method="log")
        """
        if method not in ("log", "simple"):
            raise ValueError(f"method must be 'log' or 'simple', got '{method}'.")

        if prices.empty:
            logger.warning("compute_returns received an empty DataFrame.")
            return prices

        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        returns = returns.iloc[1:]  # drop leading NaN row
        logger.info(
            "compute_returns | method=%s  shape=%s", method, returns.shape
        )
        return returns

    # ------------------------------------------------------------------
    # Technical indicators (all from scratch)
    # ------------------------------------------------------------------

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append 20+ technical indicators to an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``.  The index must be a :class:`~pandas.DatetimeIndex`.

        Returns
        -------
        pd.DataFrame
            Original OHLCV columns plus all indicator columns listed below.

        Notes
        -----
        Indicators added:

        =============================  =================
        Indicator                      Column name(s)
        =============================  =================
        Simple MA (20, 50, 200)        SMA_20, SMA_50, SMA_200
        Exponential MA (12, 26)        EMA_12, EMA_26
        MACD line                      MACD
        MACD signal (EMA-9 of MACD)    MACD_signal
        MACD histogram                 MACD_hist
        RSI (14)                       RSI_14
        Bollinger Band upper           BB_upper
        Bollinger Band middle          BB_mid
        Bollinger Band lower           BB_lower
        Average True Range (14)        ATR_14
        On-Balance Volume              OBV
        Stochastic %K (14)             STOCH_K
        Stochastic %D (3)              STOCH_D
        Williams %R (14)               WILLIAMS_R
        Rate of Change (10)            ROC_10
        Money Flow Index (14)          MFI_14
        Commodity Channel Index (20)   CCI_20
        ADX (14)                       ADX_14
        VWAP rolling (20)              VWAP_20
        =============================  =================

        Examples
        --------
        >>> enriched = prep.add_technical_indicators(clean_df)
        >>> "RSI_14" in enriched.columns
        True
        """
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"add_technical_indicators requires columns {required}; "
                f"missing: {missing}"
            )

        result = df.copy()
        close = result["Close"]
        high = result["High"]
        low = result["Low"]
        volume = result["Volume"]

        # ---- Moving averages ------------------------------------------------
        result["SMA_20"] = close.rolling(window=20).mean()
        result["SMA_50"] = close.rolling(window=50).mean()
        result["SMA_200"] = close.rolling(window=200).mean()
        result["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        result["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        # ---- MACD -----------------------------------------------------------
        result["MACD"] = result["EMA_12"] - result["EMA_26"]
        result["MACD_signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
        result["MACD_hist"] = result["MACD"] - result["MACD_signal"]

        # ---- RSI (14) -------------------------------------------------------
        result["RSI_14"] = self._rsi(close, period=14)

        # ---- Bollinger Bands (20, 2σ) ---------------------------------------
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std(ddof=1)
        result["BB_mid"] = bb_mid
        result["BB_upper"] = bb_mid + 2 * bb_std
        result["BB_lower"] = bb_mid - 2 * bb_std

        # ---- ATR (14) -------------------------------------------------------
        result["ATR_14"] = self._atr(high, low, close, period=14)

        # ---- OBV ------------------------------------------------------------
        result["OBV"] = self._obv(close, volume)

        # ---- Stochastic %K / %D (14, 3) ------------------------------------
        stoch_k, stoch_d = self._stochastic(high, low, close, k_period=14, d_period=3)
        result["STOCH_K"] = stoch_k
        result["STOCH_D"] = stoch_d

        # ---- Williams %R (14) -----------------------------------------------
        result["WILLIAMS_R"] = self._williams_r(high, low, close, period=14)

        # ---- Rate of Change (10) -------------------------------------------
        result["ROC_10"] = self._roc(close, period=10)

        # ---- Money Flow Index (14) ------------------------------------------
        result["MFI_14"] = self._mfi(high, low, close, volume, period=14)

        # ---- Commodity Channel Index (20) -----------------------------------
        result["CCI_20"] = self._cci(high, low, close, period=20)

        # ---- ADX (14) -------------------------------------------------------
        result["ADX_14"] = self._adx(high, low, close, period=14)

        # ---- Rolling VWAP (20) ----------------------------------------------
        result["VWAP_20"] = self._vwap(high, low, close, volume, window=20)

        logger.info(
            "add_technical_indicators | added %d indicator columns.",
            len(result.columns) - len(df.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_features(
        self,
        df: pd.DataFrame,
        method: NormalizeMethod = "zscore",
    ) -> pd.DataFrame:
        """Normalise feature columns in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of numeric features to normalise.
        method : {"zscore", "minmax"}, optional
            ``"zscore"``  – subtract mean and divide by standard deviation.
            ``"minmax"`` – scale each column to the ``[0, 1]`` range.
            Defaults to ``"zscore"``.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with the same shape and column names.

        Raises
        ------
        ValueError
            If *method* is not ``"zscore"`` or ``"minmax"``.

        Notes
        -----
        Columns whose standard deviation (or range) is zero are left
        unchanged to avoid division by zero.

        Examples
        --------
        >>> normed = prep.normalize_features(feature_df, method="minmax")
        """
        if method not in ("zscore", "minmax"):
            raise ValueError(
                f"method must be 'zscore' or 'minmax', got '{method}'."
            )

        result = df.copy()

        if method == "zscore":
            mu = result.mean()
            sigma = result.std(ddof=1)
            # Avoid division by zero for constant columns.
            sigma = sigma.replace(0.0, np.nan)
            result = (result - mu) / sigma
        else:  # minmax
            col_min = result.min()
            col_max = result.max()
            col_range = (col_max - col_min).replace(0.0, np.nan)
            result = (result - col_min) / col_range

        logger.info("normalize_features | method=%s  shape=%s", method, result.shape)
        return result

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def compute_rolling_stats(
        self,
        returns: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """Compute rolling summary statistics for a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            Period returns (e.g. output of :meth:`compute_returns`).
        window : int, optional
            Look-back window in periods.  Defaults to 21 (≈ one trading month).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``{col}_mean``, ``{col}_std``,
            ``{col}_skew``, ``{col}_kurt`` for every column in *returns*.

        Examples
        --------
        >>> stats = prep.compute_rolling_stats(returns, window=21)
        >>> stats.columns[:4].tolist()
        ['Close_mean', 'Close_std', 'Close_skew', 'Close_kurt']
        """
        if returns.empty:
            logger.warning("compute_rolling_stats received an empty DataFrame.")
            return returns

        frames = []
        for col in returns.columns:
            s = returns[col]
            rolling = s.rolling(window=window, min_periods=window // 2)
            stats = pd.DataFrame(
                {
                    f"{col}_mean": rolling.mean(),
                    f"{col}_std": rolling.std(ddof=1),
                    f"{col}_skew": rolling.skew(),
                    f"{col}_kurt": rolling.kurt(),
                }
            )
            frames.append(stats)

        result = pd.concat(frames, axis=1)
        logger.info(
            "compute_rolling_stats | window=%d  output_shape=%s",
            window,
            result.shape,
        )
        return result

    # ------------------------------------------------------------------
    # Private indicator implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.

        Parameters
        ----------
        close : pd.Series
            Closing price series.
        period : int
            Look-back period.

        Returns
        -------
        pd.Series
            RSI values in the range ``[0, 100]``.
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.rename("RSI")

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        period : int

        Returns
        -------
        pd.Series
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume.

        Parameters
        ----------
        close : pd.Series
        volume : pd.Series

        Returns
        -------
        pd.Series
        """
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def _stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator %K and %D.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        k_period : int
        d_period : int

        Returns
        -------
        tuple of (pd.Series, pd.Series)
            ``(%K, %D)``
        """
        roll_high = high.rolling(window=k_period).max()
        roll_low = low.rolling(window=k_period).min()
        k = 100.0 * (close - roll_low) / (roll_high - roll_low).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def _williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Williams %R.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        period : int

        Returns
        -------
        pd.Series
            Values in the range ``[-100, 0]``.
        """
        roll_high = high.rolling(window=period).max()
        roll_low = low.rolling(window=period).min()
        wr = -100.0 * (roll_high - close) / (roll_high - roll_low).replace(0, np.nan)
        return wr

    @staticmethod
    def _roc(close: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change.

        Parameters
        ----------
        close : pd.Series
        period : int

        Returns
        -------
        pd.Series
            Percentage rate of change.
        """
        return close.pct_change(periods=period) * 100.0

    @staticmethod
    def _mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Money Flow Index.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        volume : pd.Series
        period : int

        Returns
        -------
        pd.Series
            MFI values in the range ``[0, 100]``.
        """
        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume

        tp_diff = typical_price.diff()
        positive_mf = raw_money_flow.where(tp_diff > 0, 0.0)
        negative_mf = raw_money_flow.where(tp_diff < 0, 0.0)

        pos_sum = positive_mf.rolling(window=period).sum()
        neg_sum = negative_mf.rolling(window=period).sum()

        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return mfi

    @staticmethod
    def _cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        period : int

        Returns
        -------
        pd.Series
        """
        typical_price = (high + low + close) / 3.0
        ma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (typical_price - ma) / (0.015 * mad.replace(0, np.nan))
        return cci

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average Directional Index.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        period : int

        Returns
        -------
        pd.Series
            ADX values (0–100 trend strength indicator).
        """
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional movement
        up_move = high - prev_high
        down_move = prev_low - low

        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        smooth_tr = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        smooth_pos = pos_dm.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        smooth_neg = neg_dm.ewm(com=period - 1, min_periods=period, adjust=False).mean()

        pos_di = 100.0 * smooth_pos / smooth_tr.replace(0, np.nan)
        neg_di = 100.0 * smooth_neg / smooth_tr.replace(0, np.nan)

        dx = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        return adx

    @staticmethod
    def _vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Rolling Volume-Weighted Average Price.

        Parameters
        ----------
        high : pd.Series
        low : pd.Series
        close : pd.Series
        volume : pd.Series
        window : int
            Rolling window in bars.

        Returns
        -------
        pd.Series
        """
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume
        vwap = tp_vol.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return vwap
