"""Hidden Markov Model-based financial regime detection.

Detects latent market regimes (e.g. bull, bear, sideways) from return series
using a Gaussian HMM, with regime-conditional statistics, transition matrices,
and probabilistic regime assignments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """HMM-based financial regime detector.

    Fits a Gaussian Hidden Markov Model to return series to identify latent
    market regimes (e.g. bull, bear, sideways). Provides hard regime labels,
    soft (probabilistic) assignments, per-regime statistics, and the
    regime transition probability matrix.

    Parameters
    ----------
    n_regimes : int, optional
        Number of hidden regimes, by default 3.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> returns = rng.normal(0, 0.01, 500)
    >>> rd = RegimeDetector(n_regimes=3)
    >>> rd.fit(returns)
    >>> labels = rd.predict(returns)
    >>> print(labels[:10])
    """

    _REGIME_NAMES = {2: ["Bear", "Bull"], 3: ["Bear", "Sideways", "Bull"]}

    def __init__(self, n_regimes: int = 3) -> None:
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        self.n_regimes = n_regimes
        self._model: Optional[object] = None
        self._is_fitted: bool = False
        logger.debug("RegimeDetector initialised with %d regimes", n_regimes)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        """Fit a Gaussian HMM to the return series.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of log-returns (or simple returns).

        Returns
        -------
        RegimeDetector
            Self (for method chaining).

        Raises
        ------
        ValueError
            If ``returns`` is empty, contains NaN, or is too short.
        ImportError
            If ``hmmlearn`` is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("hmmlearn is required: pip install hmmlearn") from exc

        returns = self._prepare_returns(returns)
        X = returns.reshape(-1, 1)

        best_model = None
        best_score = -np.inf
        # Multiple restarts for robustness
        for seed in range(10):
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",
                    n_iter=200,
                    tol=1e-6,
                    random_state=seed,
                    verbose=False,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as exc:
                logger.debug("HMM fit attempt %d failed: %s", seed, exc)

        if best_model is None:
            raise RuntimeError("All HMM fitting attempts failed")

        self._model = best_model
        self._is_fitted = True
        logger.info(
            "RegimeDetector fitted: %d regimes, log-likelihood=%.4f",
            self.n_regimes, best_score,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Assign a hard regime label to each return observation.

        Regime indices are re-ordered so that lower indices correspond to
        more negative mean returns (i.e. index 0 = most bearish).

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns to decode.

        Returns
        -------
        np.ndarray
            Integer regime labels of length ``len(returns)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()
        returns = self._prepare_returns(returns)
        X = returns.reshape(-1, 1)
        raw_labels = self._model.predict(X)
        return self._reorder_regimes(raw_labels)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Return posterior regime probabilities for each observation.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(len(returns), n_regimes)``.
            Columns are sorted in the same regime order as ``predict``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()
        returns = self._prepare_returns(returns)
        X = returns.reshape(-1, 1)
        _, posteriors = self._model.score_samples(X)
        order = self._regime_order()
        return posteriors[:, order]

    # ------------------------------------------------------------------
    # Regime statistics
    # ------------------------------------------------------------------

    def get_regime_stats(
        self,
        returns: np.ndarray,
        regimes: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compute per-regime annualised statistics.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.
        regimes : np.ndarray, optional
            Pre-computed regime labels. If None, ``predict(returns)`` is called.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by regime with columns: ``regime_name``,
            ``mean_return_ann``, ``volatility_ann``, ``sharpe_ratio``,
            ``n_observations``, ``pct_observations``, ``skewness``, ``kurtosis``.
        """
        returns = self._prepare_returns(returns)
        if regimes is None:
            regimes = self.predict(returns)

        records = []
        names = self._REGIME_NAMES.get(self.n_regimes, [str(i) for i in range(self.n_regimes)])
        n_total = len(returns)

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            r = returns[mask]
            if len(r) < 2:
                logger.warning("Regime %d has < 2 observations; skipping stats", regime_id)
                continue
            mean_ann = float(np.mean(r) * 252)
            std_ann = float(np.std(r, ddof=1) * np.sqrt(252))
            sharpe = mean_ann / std_ann if std_ann > 1e-10 else 0.0
            skew = float(_skewness(r))
            kurt = float(_kurtosis(r))
            name = names[regime_id] if regime_id < len(names) else f"Regime {regime_id}"
            records.append({
                "regime": regime_id,
                "regime_name": name,
                "mean_return_ann": mean_ann,
                "volatility_ann": std_ann,
                "sharpe_ratio": sharpe,
                "n_observations": int(np.sum(mask)),
                "pct_observations": float(np.sum(mask)) / n_total,
                "skewness": skew,
                "kurtosis": kurt,
            })

        df = pd.DataFrame(records).set_index("regime")
        return df

    def get_transition_matrix(self) -> np.ndarray:
        """Return the regime transition probability matrix.

        Returns
        -------
        np.ndarray
            Square matrix of shape ``(n_regimes, n_regimes)`` where entry
            ``[i, j]`` is the probability of transitioning from regime ``i``
            to regime ``j``.  Rows sum to 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()
        order = self._regime_order()
        raw_transmat = self._model.transmat_
        # Re-order rows and columns
        reordered = raw_transmat[np.ix_(order, order)]
        return reordered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_returns(returns: np.ndarray) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        if r.ndim != 1:
            raise ValueError("returns must be a 1-D array")
        if len(r) < 10:
            raise ValueError("returns array is too short (< 10 observations)")
        if np.any(np.isnan(r)):
            raise ValueError("returns contains NaN values")
        return r

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _regime_order(self) -> np.ndarray:
        """Return regime indices sorted by ascending mean return."""
        means = self._model.means_.flatten()
        return np.argsort(means)

    def _reorder_regimes(self, raw_labels: np.ndarray) -> np.ndarray:
        """Re-map raw HMM state indices so that 0 = most bearish."""
        order = self._regime_order()
        mapping = {old: new for new, old in enumerate(order)}
        return np.vectorize(mapping.get)(raw_labels)


# ------------------------------------------------------------------
# Module-level statistical helpers (no external dependencies)
# ------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    """Sample skewness with bias correction."""
    n = len(x)
    if n < 3:
        return float("nan")
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3) * n * (n - 1) / (n - 2) if n > 2 else 0.0)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition, normal = 0)."""
    n = len(x)
    if n < 4:
        return float("nan")
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)
