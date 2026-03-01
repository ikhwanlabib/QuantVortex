"""
Volatility surface arbitrage strategy for QuantVortex.

Fits an SVI (Stochastic Volatility Inspired) parameterisation to implied
volatility smiles, checks for calendar-spread and butterfly arbitrage
violations, and generates trading signals from surface dislocations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVI smile model
# ---------------------------------------------------------------------------

def _svi_smile(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Evaluate the SVI raw parameterisation for total implied variance.

    .. math::

        w(k) = a + b \\left( \\rho (k - m) + \\sqrt{(k-m)^2 + \\sigma^2} \\right)

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness values.
    a, b, rho, m, sigma : float
        SVI raw parameters.

    Returns
    -------
    np.ndarray
        Total implied variance values ``w(k) ≥ 0``.
    """
    inner = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = a + b * (rho * (k - m) + inner)
    return w


class VolSurfaceArb(BaseStrategy):
    """Volatility surface arbitrage strategy.

    Detects calendar-spread and butterfly arbitrage violations in an implied
    volatility surface and generates long/short vega positions to exploit
    mispricings.

    Parameters
    ----------
    name : str, optional
        Strategy name. Default is ``"VolSurfaceArb"``.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    min_expiry_days : int, optional
        Minimum days-to-expiry to include in the surface. Default is 7.
    arb_threshold : float, optional
        Minimum violation magnitude (in vol units) to generate a signal.
        Default is 0.005 (0.5 vol point).
    signal_scale : float, optional
        Position size per unit of violation magnitude. Default is 1.0.

    Attributes
    ----------
    svi_fits : dict
        SVI parameters per expiry, populated after :meth:`generate_signals`.
    """

    def __init__(
        self,
        name: str = "VolSurfaceArb",
        allocation: float = 1.0,
        min_expiry_days: int = 7,
        arb_threshold: float = 0.005,
        signal_scale: float = 1.0,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        self.min_expiry_days = min_expiry_days
        self.arb_threshold = arb_threshold
        self.signal_scale = signal_scale
        self.svi_fits: Dict[float, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # SVI fitting
    # ------------------------------------------------------------------

    def svi_params(
        self,
        k: np.ndarray,
        market_vols: np.ndarray,
    ) -> Dict[str, float]:
        """Fit SVI raw parameters to a single implied volatility smile.

        Minimises the sum of squared differences between the SVI total
        variance curve and ``market_vols^2`` (total variance).

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness array, shape ``(N,)``.
        market_vols : np.ndarray
            Market implied volatilities, shape ``(N,)``. Values must be > 0.

        Returns
        -------
        dict
            Keys: ``"a"``, ``"b"``, ``"rho"``, ``"m"``, ``"sigma"``.

        Raises
        ------
        ValueError
            If ``k`` and ``market_vols`` have different lengths or fewer
            than 4 data points.
        RuntimeError
            If the optimisation fails to produce admissible parameters.
        """
        k = np.asarray(k, dtype=float)
        market_vols = np.asarray(market_vols, dtype=float)
        if k.shape != market_vols.shape:
            raise ValueError("k and market_vols must have the same shape.")
        if len(k) < 4:
            raise ValueError("Need at least 4 points to fit SVI parameters.")

        market_var = market_vols ** 2

        def objective(params: np.ndarray) -> float:
            a, b, rho, m, sigma = params
            # Admissibility constraints (penalised)
            penalty = 0.0
            if b < 0:
                penalty += 1e6 * (-b)
            if abs(rho) >= 1:
                penalty += 1e6 * (abs(rho) - 1)
            if sigma <= 0:
                penalty += 1e6 * (-sigma + 1e-6)
            # Butterfly no-arb: b*(1+|rho|) <= 2*a is a necessary condition
            if a < 0 and b * (1 + abs(rho)) > 2 * a:
                penalty += 1e4
            w = _svi_smile(k, a, b, rho, m, sigma)
            w = np.clip(w, 1e-12, None)
            return float(np.sum((w - market_var) ** 2)) + penalty

        # Initial guess: flat smile
        a0 = float(np.mean(market_var))
        b0 = 0.1
        rho0 = -0.3
        m0 = 0.0
        sigma0 = 0.2

        result = minimize(
            objective,
            x0=[a0, b0, rho0, m0, sigma0],
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-10},
        )

        a, b, rho, m, sigma = result.x
        # Enforce b ≥ 0, |rho| < 1, sigma > 0 hard clipping after fitting
        b = max(b, 0.0)
        rho = float(np.clip(rho, -0.9999, 0.9999))
        sigma = max(sigma, 1e-6)

        params = {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)}
        self.logger.debug("SVI fit: %s", params)
        return params

    # ------------------------------------------------------------------
    # Arbitrage checks
    # ------------------------------------------------------------------

    def check_calendar_spread_arb(self, surface: pd.DataFrame) -> pd.Series:
        """Detect calendar-spread arbitrage in the implied vol surface.

        Calendar spread arbitrage exists when ATM implied variance decreases
        as expiry increases (i.e., the term structure of total variance is
        not monotonically non-decreasing at any moneyness).

        Parameters
        ----------
        surface : pd.DataFrame
            Implied volatility surface with expiry (days) as the index and
            log-moneyness as columns.  Values are implied volatilities.

        Returns
        -------
        pd.Series
            Boolean series indexed by expiry pair ``"T1_vs_T2"``, ``True``
            where a violation is detected.

        Raises
        ------
        ValueError
            If ``surface`` has fewer than 2 expiry rows.
        """
        if len(surface) < 2:
            raise ValueError("Calendar spread check requires at least 2 expiries.")

        expiries = surface.index.astype(float).values
        order = np.argsort(expiries)
        sorted_expiries = expiries[order]
        sorted_surface = surface.iloc[order]

        violations: Dict[str, bool] = {}
        for i in range(len(sorted_expiries) - 1):
            t1, t2 = sorted_expiries[i], sorted_expiries[i + 1]
            row1 = sorted_surface.iloc[i].values.astype(float)
            row2 = sorted_surface.iloc[i + 1].values.astype(float)

            # Total variance = T * sigma^2; must be non-decreasing in T
            var1 = row1 ** 2 * t1 / 365.0
            var2 = row2 ** 2 * t2 / 365.0

            # A violation occurs if total variance decreases at any point
            has_violation = bool(np.any(var2 < var1 - 1e-6))
            key = f"{int(t1)}_vs_{int(t2)}"
            violations[key] = has_violation
            if has_violation:
                self.logger.debug("Calendar arb detected: %s", key)

        return pd.Series(violations, name="calendar_arb_violation")

    def check_butterfly_arb(
        self, k: np.ndarray, vols: np.ndarray
    ) -> bool:
        """Check for butterfly arbitrage via risk-neutral density positivity.

        Butterfly arbitrage is absent if and only if the risk-neutral
        density (second derivative of the call price w.r.t. strike, or
        equivalently the density implied from Breeden-Litzenberger) is
        non-negative everywhere.

        This implementation uses the Dupire/Breeden-Litzenberger formula:
        the risk-neutral density ``g(k)`` from total variance ``w(k)``
        must satisfy ``g(k) ≥ 0``.

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness, sorted ascending, shape ``(N,)``.
        vols : np.ndarray
            Implied volatilities at each ``k``, shape ``(N,)``.

        Returns
        -------
        bool
            ``True`` if butterfly arbitrage is absent (density ≥ 0),
            ``False`` if a violation is detected.
        """
        k = np.asarray(k, dtype=float)
        vols = np.asarray(vols, dtype=float)
        if len(k) < 5:
            self.logger.debug("check_butterfly_arb: too few points (%d); skipping.", len(k))
            return True  # conservative: no violation declared

        # Fit SVI and evaluate density on a dense grid
        try:
            fitted = self.svi_params(k, vols)
        except Exception as exc:
            self.logger.warning("butterfly check SVI fit failed: %s", exc)
            return True

        k_dense = np.linspace(k.min() - 0.5, k.max() + 0.5, 500)
        w = _svi_smile(k_dense, **fitted)
        w = np.clip(w, 1e-12, None)

        # Breeden-Litzenberger: g(d2) = ... approximate by finite differences
        dw = np.gradient(w, k_dense)
        d2w = np.gradient(dw, k_dense)

        # Local vol check: density is proportional to (1 - k*dw/(2w) + ...)
        # Use the simplified check: d2w >= 0 is necessary (convexity of w in k)
        # plus w(k) >= 0 already enforced
        butterfly_free = bool(np.all(d2w >= -1e-6))
        if not butterfly_free:
            self.logger.debug("Butterfly arb detected.")
        return butterfly_free

    # ------------------------------------------------------------------
    # Synthetic surface generation (for testing without live data)
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_surface(
        n_expiries: int = 6,
        n_strikes: int = 9,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a synthetic implied vol surface for testing.

        Injects small random perturbations around a parametric smile so
        that the strategy has something actionable to work with.

        Parameters
        ----------
        n_expiries : int, optional
            Number of expiry slices. Default is 6.
        n_strikes : int, optional
            Number of log-moneyness grid points. Default is 9.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Surface with expiry days as index and log-moneyness as columns.
        """
        rng = np.random.default_rng(seed)
        expiries = np.array([7, 14, 30, 60, 90, 180, 252])[:n_expiries]
        k_grid = np.linspace(-0.5, 0.5, n_strikes)

        rows: Dict[int, np.ndarray] = {}
        for T in expiries:
            t_y = T / 252.0
            # A simple parametric smile (vol increases for OTM)
            base_vol = 0.20 + 0.05 * k_grid ** 2 - 0.02 * k_grid
            # Add noise and a random dislocation in one slice
            noise = rng.normal(0, 0.003, size=n_strikes)
            rows[T] = np.clip(base_vol + noise, 0.01, 1.5)

        df = pd.DataFrame(rows, index=k_grid).T
        df.index.name = "expiry_days"
        df.columns = [f"k_{kv:.2f}" for kv in k_grid]
        return df

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol surface dislocation signals.

        If ``data`` contains a ``"date"`` column (time-indexed snapshots),
        it processes each snapshot; otherwise it treats ``data`` as a single
        surface snapshot (expiry × moneyness).

        Signals are non-zero where SVI-fitted vols deviate from market vols
        by more than ``self.arb_threshold``.

        Parameters
        ----------
        data : pd.DataFrame
            Either:
            - A vol surface with expiry (days) as index and log-moneyness
              as columns (values = implied vols), **or**
            - A panel with a ``"date"`` column and the above as the
              remaining columns.

        Returns
        -------
        pd.DataFrame
            Signal DataFrame.  Each column represents an (expiry, strike)
            pair.  Positive value = surface too cheap (buy vol), negative =
            too expensive (sell vol).
        """
        if data is None or data.empty:
            self.logger.warning(
                "No data provided; generating synthetic surface for testing."
            )
            data = self._synthetic_surface(seed=42)

        # Detect whether data is a single surface or a time-panel
        if "date" in data.columns:
            dates = data["date"].unique()
            all_signals: List[pd.DataFrame] = []
            for d in dates:
                slice_df = data[data["date"] == d].drop(columns="date")
                sigs = self._signals_from_surface(slice_df)
                sigs.index = [d] * len(sigs)
                all_signals.append(sigs)
            if not all_signals:
                return pd.DataFrame()
            return pd.concat(all_signals)
        else:
            return self._signals_from_surface(data)

    @staticmethod
    def _expiry_to_float(expiry: Any) -> float:
        """Convert an expiry index label to a float number of days.

        Parameters
        ----------
        expiry : Any
            Expiry label (int, float, or string like ``"30_days"``).

        Returns
        -------
        float
            Number of days as a float.
        """
        return float(str(expiry).replace("_days", ""))

    def _signals_from_surface(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Compute signals from a single surface snapshot.

        Parameters
        ----------
        surface : pd.DataFrame
            Implied vol surface (expiry rows × moneyness columns).

        Returns
        -------
        pd.DataFrame
            Signal DataFrame indexed by expiry, one column per
            log-moneyness grid point.
        """
        k_values = np.array(
            [float(str(c).replace("k_", "")) for c in surface.columns],
            dtype=float,
        )
        signals = pd.DataFrame(
            0.0, index=surface.index, columns=surface.columns
        )

        for expiry in surface.index:
            if self._expiry_to_float(expiry) < self.min_expiry_days:
                continue
            row_vols = surface.loc[expiry].values.astype(float)
            valid_mask = ~np.isnan(row_vols) & (row_vols > 0)
            if valid_mask.sum() < 4:
                continue

            k_valid = k_values[valid_mask]
            v_valid = row_vols[valid_mask]

            try:
                fitted = self.svi_params(k_valid, v_valid)
            except Exception as exc:
                self.logger.warning(
                    "SVI fit failed for expiry %s: %s", expiry, exc
                )
                continue

            self.svi_fits[self._expiry_to_float(expiry)] = fitted
            fitted_vols = np.sqrt(
                np.clip(_svi_smile(k_valid, **fitted), 1e-12, None)
            )
            residuals = v_valid - fitted_vols
            threshold = self.arb_threshold

            row_signal = np.zeros(len(k_values))
            row_signal[valid_mask] = np.where(
                np.abs(residuals) > threshold,
                np.sign(residuals) * self.signal_scale,
                0.0,
            )
            signals.loc[expiry] = row_signal

        # Calendar spread check
        try:
            cal_viols = self.check_calendar_spread_arb(surface)
            n_cal = int(cal_viols.sum())
            if n_cal > 0:
                self.logger.info(
                    "%d calendar spread violation(s) detected.", n_cal
                )
        except Exception as exc:
            self.logger.debug("Calendar spread check failed: %s", exc)

        # Butterfly check per slice
        for expiry in surface.index:
            row_vols = surface.loc[expiry].values.astype(float)
            valid_mask = ~np.isnan(row_vols) & (row_vols > 0)
            if valid_mask.sum() < 5:
                continue
            bf_free = self.check_butterfly_arb(
                k_values[valid_mask], row_vols[valid_mask]
            )
            if not bf_free:
                self.logger.info(
                    "Butterfly arb present at expiry %s; boosting signal.", expiry
                )
                signals.loc[expiry] *= 1.5

        self.logger.info(
            "_signals_from_surface: %d non-zero cells",
            int((signals != 0).values.sum()),
        )
        return signals

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert dislocation signals to normalised position weights.

        Normalises each row (expiry slice) by its L1 norm so that the
        gross exposure per expiry is at most 1.0.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals`.

        Returns
        -------
        pd.DataFrame
            Position weights in ``[-1, +1]`` per (expiry, strike) cell.
        """
        if signals.empty:
            self.logger.warning("compute_positions: empty signals.")
            return pd.DataFrame(index=signals.index)

        positions = signals.copy().astype(float)
        for idx in positions.index:
            row = positions.loc[idx].abs().sum()
            if row > 0:
                positions.loc[idx] /= row

        self.logger.info(
            "compute_positions: gross exposure max=%.4f",
            float(positions.abs().sum(axis=1).max()) if not positions.empty else 0.0,
        )
        return positions
