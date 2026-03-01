"""
data/alternative_data.py
========================
Sentiment analysis utilities for QuantVortex.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to score
synthetic news headlines when real API access is unavailable.  When a live
news feed is configured the same interface can be used without changes to
downstream code.
"""

import hashlib
import logging
import random
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-built headline templates used for synthetic data generation.
# These span a realistic range of sentiment to give VADER scores spread
# across the full [-1, +1] spectrum.
# ---------------------------------------------------------------------------
_BULLISH_TEMPLATES = [
    "{ticker} beats quarterly earnings estimates by wide margin",
    "{ticker} announces record revenue and raises full-year guidance",
    "{ticker} shares surge after CEO reveals bold expansion strategy",
    "{ticker} secures landmark government contract worth billions",
    "{ticker} reports strong cash flow growth and doubles dividend",
    "Analysts upgrade {ticker} to 'Strong Buy' on robust demand outlook",
    "{ticker} posts blowout sales figures, stock hits all-time high",
    "{ticker} partners with leading tech giant in transformative deal",
    "{ticker} achieves FDA approval for breakthrough product",
    "{ticker} buyback programme signals management confidence",
]

_BEARISH_TEMPLATES = [
    "{ticker} misses earnings expectations; shares fall sharply",
    "{ticker} warns of revenue shortfall, citing weak consumer demand",
    "{ticker} faces regulatory probe over alleged accounting irregularities",
    "Analyst downgrades {ticker} citing competitive pressures and margin squeeze",
    "{ticker} slashes dividend as cash flow deteriorates",
    "{ticker} recalls major product line amid safety concerns",
    "{ticker} CEO unexpectedly resigns, board launches search",
    "{ticker} loses key contract to rival; guidance slashed",
    "{ticker} reports steepest quarterly loss in a decade",
    "{ticker} stock slides on disappointing guidance and weakening demand",
]

_NEUTRAL_TEMPLATES = [
    "{ticker} to report quarterly earnings next week",
    "{ticker} appoints new Chief Financial Officer",
    "{ticker} to present at industry conference on Thursday",
    "{ticker} updates investor relations calendar for fiscal year",
    "{ticker} files routine 10-Q with SEC",
    "{ticker} holds annual shareholder meeting without major surprises",
    "{ticker} announces board member rotation in line with governance policy",
    "{ticker} reaffirms full-year guidance at analyst day",
    "{ticker} completes previously announced share repurchase",
    "{ticker} sets date for next earnings release",
]


class SentimentAnalyzer:
    """Compute sentiment scores for equity tickers from news headlines.

    Uses VADER for scoring.  When live news data is unavailable, the class
    generates deterministic synthetic headlines so that downstream pipelines
    can be tested and back-tested without requiring paid data feeds.

    Parameters
    ----------
    random_seed : int, optional
        Seed for the synthetic headline generator.  Fixing the seed
        produces reproducible sentiment series.  Defaults to ``42``.

    Examples
    --------
    >>> analyzer = SentimentAnalyzer()
    >>> sentiment_df = analyzer.get_news_sentiment("AAPL", days_back=30)
    >>> momentum = analyzer.compute_sentiment_momentum(sentiment_df, window=5)
    """

    def __init__(self, random_seed: int = 42) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._random_seed = random_seed
        logger.info("SentimentAnalyzer initialised (VADER backend).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_synthetic_headlines(
        self,
        ticker: str,
        date: datetime,
        n: int = 3,
    ) -> list[str]:
        """Generate *n* deterministic synthetic headlines for *ticker* on *date*.

        The headlines are chosen pseudo-randomly using a seed derived from
        the ticker symbol and the date so that re-runs always produce the
        same output.

        Parameters
        ----------
        ticker : str
            Equity ticker symbol.
        date : datetime
            Trading date for which headlines are generated.
        n : int, optional
            Number of headlines per day.  Defaults to ``3``.

        Returns
        -------
        list of str
            Synthetic headline strings.
        """
        # Build a deterministic seed from ticker + date so results are
        # reproducible across runs without relying on global state.
        seed_str = f"{ticker}_{date.strftime('%Y%m%d')}_{self._random_seed}"
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(seed_int)

        all_templates = _BULLISH_TEMPLATES + _BEARISH_TEMPLATES + _NEUTRAL_TEMPLATES
        chosen = rng.choices(all_templates, k=n)
        return [t.format(ticker=ticker) for t in chosen]

    def _score_headline(self, headline: str) -> float:
        """Score a single headline using VADER.

        Parameters
        ----------
        headline : str
            Raw headline text.

        Returns
        -------
        float
            Compound sentiment score in the range ``[-1.0, +1.0]``.
        """
        scores = self._vader.polarity_scores(headline)
        return float(scores["compound"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_news_sentiment(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Return a daily sentiment DataFrame for *ticker*.

        When a live news API is not configured this method falls back to
        scoring synthetic headlines generated by
        :meth:`_generate_synthetic_headlines`.

        Parameters
        ----------
        ticker : str
            Equity ticker symbol (e.g. ``"AAPL"``).
        days_back : int, optional
            Number of calendar days of history to generate.  Defaults to
            ``30``.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and the following columns:

            * ``ticker``          – ticker symbol (str)
            * ``sentiment_score`` – mean VADER compound score for the day
              in ``[-1.0, +1.0]`` (float)
            * ``headline_count``  – number of headlines scored (int)

        Notes
        -----
        Weekends are included in the output because sentiment data may be
        generated on non-trading days.  Callers should align with a trading
        calendar if needed.

        Examples
        --------
        >>> df = analyzer.get_news_sentiment("MSFT", days_back=7)
        >>> df.columns.tolist()
        ['ticker', 'sentiment_score', 'headline_count']
        """
        ticker = ticker.upper().strip()
        logger.info(
            "get_news_sentiment | ticker=%s  days_back=%d", ticker, days_back
        )

        end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days_back - 1)

        records = []
        current = start_date
        while current <= end_date:
            try:
                headlines = self._generate_synthetic_headlines(ticker, current, n=3)
                scores = [self._score_headline(h) for h in headlines]
                mean_score = float(sum(scores) / len(scores)) if scores else 0.0
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sentiment scoring failed for %s on %s: %s",
                    ticker,
                    current.date(),
                    exc,
                )
                mean_score = 0.0
                headlines = []

            records.append(
                {
                    "date": current,
                    "ticker": ticker,
                    "sentiment_score": mean_score,
                    "headline_count": len(headlines),
                }
            )
            current += timedelta(days=1)

        result = pd.DataFrame(records).set_index("date")
        result.index = pd.to_datetime(result.index)
        result.index.name = "date"

        logger.info(
            "get_news_sentiment | %s: %d records, mean_score=%.4f",
            ticker,
            len(result),
            result["sentiment_score"].mean(),
        )
        return result

    def compute_sentiment_momentum(
        self,
        sentiment_df: pd.DataFrame,
        window: int = 5,
    ) -> pd.Series:
        """Compute rolling-average sentiment momentum.

        Parameters
        ----------
        sentiment_df : pd.DataFrame
            Output of :meth:`get_news_sentiment` (must contain a
            ``sentiment_score`` column).
        window : int, optional
            Rolling window in days.  Defaults to ``5``.

        Returns
        -------
        pd.Series
            Rolling mean of ``sentiment_score`` with the same index as
            *sentiment_df*.  Named ``sentiment_momentum``.

        Raises
        ------
        ValueError
            If *sentiment_df* does not contain a ``sentiment_score`` column.

        Examples
        --------
        >>> momentum = analyzer.compute_sentiment_momentum(df, window=5)
        >>> momentum.name
        'sentiment_momentum'
        """
        if "sentiment_score" not in sentiment_df.columns:
            raise ValueError(
                "sentiment_df must contain a 'sentiment_score' column."
            )

        if sentiment_df.empty:
            logger.warning("compute_sentiment_momentum received an empty DataFrame.")
            return pd.Series(dtype=float, name="sentiment_momentum")

        momentum = (
            sentiment_df["sentiment_score"]
            .rolling(window=window, min_periods=1)
            .mean()
            .rename("sentiment_momentum")
        )

        logger.info(
            "compute_sentiment_momentum | window=%d  records=%d",
            window,
            len(momentum),
        )
        return momentum

    def aggregate_sentiment(self, tickers: List[str]) -> pd.DataFrame:
        """Aggregate sentiment across multiple tickers into a wide DataFrame.

        Calls :meth:`get_news_sentiment` for each ticker and pivots the
        result so that each column represents one ticker's daily sentiment
        score.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to aggregate.

        Returns
        -------
        pd.DataFrame
            Wide DataFrame indexed by date; columns are ticker symbols.
            Missing dates for a ticker are forward-filled then
            backward-filled.

        Raises
        ------
        ValueError
            If *tickers* is empty.

        Examples
        --------
        >>> agg = analyzer.aggregate_sentiment(["AAPL", "MSFT", "GOOGL"])
        >>> agg.columns.tolist()
        ['AAPL', 'GOOGL', 'MSFT']
        """
        if not tickers:
            raise ValueError("tickers list must not be empty.")

        tickers = [t.upper().strip() for t in tickers]
        logger.info("aggregate_sentiment | tickers=%s", tickers)

        frames: list[pd.Series] = []
        for ticker in tickers:
            try:
                df = self.get_news_sentiment(ticker)
                series = df["sentiment_score"].rename(ticker)
                frames.append(series)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to fetch sentiment for %s: %s", ticker, exc
                )

        if not frames:
            logger.warning("aggregate_sentiment: no data retrieved for any ticker.")
            return pd.DataFrame()

        wide = pd.concat(frames, axis=1).sort_index()
        wide = wide.ffill().bfill()

        logger.info(
            "aggregate_sentiment | shape=%s  date_range=%s to %s",
            wide.shape,
            wide.index.min().date() if not wide.empty else "N/A",
            wide.index.max().date() if not wide.empty else "N/A",
        )
        return wide
