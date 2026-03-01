"""
data/fetcher.py
===============
Multi-source market data fetcher for QuantVortex.

Supports yfinance as the primary data source with local pickle-based caching,
rate limiting, and graceful error handling for missing or delisted tickers.
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path("data/cache")
_RATE_LIMIT_SECONDS = 0.5  # pause between individual ticker requests


class DataFetcher:
    """Fetches OHLCV market data from yfinance with local caching.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory used to store pickled DataFrames.  Created automatically
        if it does not exist.  Defaults to ``data/cache``.
    use_cache : bool, optional
        When ``True`` (default) previously downloaded data is served from
        disk; set to ``False`` to always re-download.

    Examples
    --------
    >>> fetcher = DataFetcher()
    >>> df = fetcher.fetch_prices(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
    """

    def __init__(
        self,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataFetcher initialised | cache_dir=%s", self.cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> Path:
        """Return the pickle path for a given query.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        start : str
            Start date string.
        end : str
            End date string.
        interval : str
            Bar interval string (e.g. ``"1d"``).

        Returns
        -------
        Path
            Absolute path to the cache file.
        """
        safe_ticker = ticker.replace("/", "_").replace("^", "")
        filename = f"{safe_ticker}_{start}_{end}_{interval}.pkl"
        return self.cache_dir / filename

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a pickled DataFrame from *path*.

        Parameters
        ----------
        path : Path
            Cache file to load.

        Returns
        -------
        pd.DataFrame or None
            Cached DataFrame, or ``None`` if the file does not exist or
            cannot be deserialised.
        """
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                df: pd.DataFrame = pickle.load(fh)
            logger.debug("Cache hit: %s", path.name)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load cache %s: %s", path, exc)
            return None

    def _save_cache(self, path: Path, df: pd.DataFrame) -> None:
        """Persist *df* to *path* as a pickle file.

        Parameters
        ----------
        path : Path
            Destination file path.
        df : pd.DataFrame
            DataFrame to serialise.
        """
        try:
            with open(path, "wb") as fh:
                pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Cached to disk: %s", path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write cache %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for one or more tickers via yfinance.

        When multiple tickers are requested the result is a DataFrame with a
        two-level column MultiIndex ``(field, ticker)`` e.g. ``("Close",
        "AAPL")``.  When a single ticker is requested the columns are the
        plain OHLCV field names.

        Parameters
        ----------
        tickers : list of str
            List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format (exclusive in yfinance).
        interval : str, optional
            Bar size.  Any interval accepted by yfinance (``"1d"``,
            ``"1h"``, ``"15m"``, …).  Defaults to ``"1d"``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``datetime``.  Columns are a MultiIndex
            ``(field, ticker)`` for multiple tickers, or plain OHLCV fields
            for a single ticker.

        Raises
        ------
        ValueError
            If *tickers* is empty or if no data could be retrieved for any
            of the requested symbols.

        Examples
        --------
        >>> fetcher = DataFetcher()
        >>> df = fetcher.fetch_prices(["AAPL"], "2023-01-01", "2024-01-01")
        >>> df.columns.tolist()
        ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        if not tickers:
            raise ValueError("tickers list must not be empty.")

        tickers = [t.upper().strip() for t in tickers]
        logger.info(
            "Fetching prices | tickers=%s  start=%s  end=%s  interval=%s",
            tickers,
            start,
            end,
            interval,
        )

        # For a single ticker we return a simple (non-multi) DataFrame.
        if len(tickers) == 1:
            return self._fetch_single(tickers[0], start, end, interval)

        # For multiple tickers we build a MultiIndex DataFrame.
        frames: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._fetch_single(ticker, start, end, interval)
            if df is not None and not df.empty:
                frames[ticker] = df
            time.sleep(_RATE_LIMIT_SECONDS)

        if not frames:
            raise ValueError(
                f"No data retrieved for any of the requested tickers: {tickers}"
            )

        multi = pd.concat(frames, axis=1)
        multi.columns.names = ["field", "ticker"]
        logger.info("Fetched multi-ticker frame: shape=%s", multi.shape)
        return multi

    def _fetch_single(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """Download (or load from cache) OHLCV data for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        start : str
            Start date string.
        end : str
            End date string.
        interval : str
            Bar size string.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame.  Returns an empty DataFrame on failure instead
            of raising so callers can skip missing symbols.
        """
        cache_path = self._cache_path(ticker, start, end, interval)

        if self.use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        try:
            logger.debug("Downloading %s from yfinance …", ticker)
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("No data returned by yfinance for %s", ticker)
            return pd.DataFrame()

        # Flatten MultiIndex columns produced by recent yfinance versions.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalise column names (title-case to preserve e.g. "Adj Close").
        raw.columns = [c.title() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "Date"

        logger.info("Downloaded %s: %d rows", ticker, len(raw))
        self._save_cache(cache_path, raw)
        return raw

    def fetch_multiple(
        self,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for each ticker and return a per-ticker mapping.

        Unlike :meth:`fetch_prices`, this method returns a plain ``dict``
        mapping each ticker symbol to its own OHLCV :class:`~pandas.DataFrame`
        rather than a combined MultiIndex frame.  Tickers for which no data
        is available are omitted from the result.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to download.
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.
        interval : str, optional
            Bar size.  Defaults to ``"1d"``.

        Returns
        -------
        dict of {str: pd.DataFrame}
            Mapping ``{ticker: ohlcv_dataframe}``.

        Examples
        --------
        >>> fetcher = DataFetcher()
        >>> data = fetcher.fetch_multiple(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
        >>> list(data.keys())
        ['AAPL', 'MSFT']
        """
        if not tickers:
            raise ValueError("tickers list must not be empty.")

        tickers = [t.upper().strip() for t in tickers]
        result: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            df = self._fetch_single(ticker, start, end, interval)
            if df is not None and not df.empty:
                result[ticker] = df
            else:
                logger.warning("Skipping %s – no data available.", ticker)
            time.sleep(_RATE_LIMIT_SECONDS)

        logger.info(
            "fetch_multiple complete | requested=%d  retrieved=%d",
            len(tickers),
            len(result),
        )
        return result

    def fetch_from_config(self, config_path: str) -> pd.DataFrame:
        """Load fetch parameters from a YAML config and download data.

        Reads ``data.universe``, ``data.start_date``, ``data.end_date``, and
        optionally ``data.cache_dir`` from the supplied YAML file then calls
        :meth:`fetch_prices`.

        Parameters
        ----------
        config_path : str
            Path to a YAML settings file.  Expected structure::

                data:
                  universe: ["AAPL", "MSFT", ...]
                  start_date: "2015-01-01"
                  end_date:   "2025-12-31"
                  cache_dir:  "data/cache"   # optional

        Returns
        -------
        pd.DataFrame
            MultiIndex OHLCV DataFrame (or single-column DataFrame when only
            one ticker is configured).

        Raises
        ------
        FileNotFoundError
            If *config_path* does not exist.
        KeyError
            If required keys are absent from the YAML file.

        Examples
        --------
        >>> fetcher = DataFetcher()
        >>> df = fetcher.fetch_from_config("config/settings.yaml")
        """
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path_obj, "r") as fh:
            config = yaml.safe_load(fh)

        data_cfg = config.get("data", {})

        universe: List[str] = data_cfg.get("universe", [])
        start: str = data_cfg.get("start_date", "")
        end: str = data_cfg.get("end_date", "")

        if not universe:
            raise KeyError("'data.universe' is missing or empty in the config.")
        if not start or not end:
            raise KeyError(
                "'data.start_date' and 'data.end_date' must be set in the config."
            )

        # Override cache dir if specified in config.
        cfg_cache = data_cfg.get("cache_dir")
        if cfg_cache:
            self.cache_dir = Path(cfg_cache)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "fetch_from_config | universe=%d tickers  %s → %s",
            len(universe),
            start,
            end,
        )
        return self.fetch_prices(universe, start, end)
