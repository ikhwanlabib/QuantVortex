"""
Execution engine module for QuantVortex trading engine.

Implements optimal execution scheduling (Almgren-Chriss, TWAP, VWAP),
market impact models (linear, square-root), Kyle's lambda estimation,
transaction cost analysis (TCA), and order fill simulation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Production-grade optimal execution and transaction cost analysis engine.

    Provides analytical and model-based execution scheduling, market impact
    estimation, TCA reporting, and fill simulation for equities.

    Parameters
    ----------
    lot_size : int, optional
        Minimum share lot size for rounding. Default 1 (no rounding).
    commission_per_share : float, optional
        Fixed commission per share in currency units. Default 0.0.

    Examples
    --------
    >>> engine = ExecutionEngine(lot_size=100, commission_per_share=0.005)
    >>> schedule = engine.twap_schedule(10000, n_slices=20)
    """

    def __init__(
        self,
        lot_size: int = 1,
        commission_per_share: float = 0.0,
    ) -> None:
        if lot_size < 1:
            raise ValueError("lot_size must be a positive integer.")
        self.lot_size = lot_size
        self.commission_per_share = commission_per_share
        logger.info(
            "ExecutionEngine initialised with lot_size=%d, commission=%.5f",
            lot_size,
            commission_per_share,
        )

    # ------------------------------------------------------------------
    # Almgren-Chriss Optimal Execution Schedule
    # ------------------------------------------------------------------

    def almgren_chriss_schedule(
        self,
        shares: float,
        T: float,
        sigma: float,
        eta: float,
        gamma: float,
        urgency: float = 0.5,
        n_slices: int = 10,
    ) -> np.ndarray:
        """Almgren-Chriss optimal execution trajectory.

        Computes the optimal number of shares to trade in each time slice by
        minimising the mean-variance cost functional:

        ``E[cost] + lambda * Var[cost]``

        where ``lambda`` (urgency) governs the risk-aversion of the trader.

        Parameters
        ----------
        shares : float
            Total shares to liquidate (positive).
        T : float
            Total execution horizon in days.
        sigma : float
            Daily volatility of the asset (annualised sigma / sqrt(252)).
        eta : float
            Temporary market-impact coefficient (linear in trading rate).
        gamma : float
            Permanent market-impact coefficient (linear in trade size).
        urgency : float, optional
            Risk-aversion parameter ``lambda`` in ``[0, inf)``. Higher values
            favour faster execution. Default 0.5.
        n_slices : int, optional
            Number of equal-sized time intervals. Default 10.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` containing shares to trade per slice,
            summing approximately to ``shares``.

        Raises
        ------
        ValueError
            If ``shares``, ``T``, ``sigma``, ``eta``, or ``n_slices`` are non-positive.
        """
        if shares <= 0:
            raise ValueError("shares must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if n_slices < 1:
            raise ValueError("n_slices must be at least 1.")

        tau = T / n_slices  # length of each slice

        # Kappa: characteristic decay rate from AC model
        kappa_sq = urgency * sigma ** 2 / eta
        kappa = np.sqrt(max(kappa_sq, 1e-14))

        # Closed-form optimal trajectory: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
        times = np.linspace(0, T, n_slices + 1)
        sinh_kT = np.sinh(kappa * T)

        if abs(sinh_kT) < 1e-14:
            # Near-zero kappa: fall back to linear (TWAP) schedule
            logger.debug("almgren_chriss: kappa~0, falling back to TWAP.")
            return self.twap_schedule(shares, n_slices)

        holdings = shares * np.sinh(kappa * (T - times)) / sinh_kT
        # Trade sizes are differences in the holdings trajectory
        schedule = np.diff(-holdings)  # positive = buying; negative = liquidating
        schedule = np.abs(schedule)

        # Rescale to match total shares exactly
        total = schedule.sum()
        if total > 1e-14:
            schedule = schedule * shares / total

        schedule = self._apply_lot_rounding(schedule, shares)
        logger.info(
            "almgren_chriss_schedule: %d slices, total_shares=%.0f", n_slices, schedule.sum()
        )
        return schedule

    # ------------------------------------------------------------------
    # TWAP Schedule
    # ------------------------------------------------------------------

    def twap_schedule(
        self,
        total_shares: float,
        n_slices: int = 10,
    ) -> np.ndarray:
        """Time-Weighted Average Price (TWAP) execution schedule.

        Divides ``total_shares`` equally across ``n_slices`` time intervals.

        Parameters
        ----------
        total_shares : float
            Total number of shares to trade.
        n_slices : int, optional
            Number of equal time slices. Default 10.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` with equal share quantities.

        Raises
        ------
        ValueError
            If ``total_shares <= 0`` or ``n_slices < 1``.
        """
        if total_shares <= 0:
            raise ValueError("total_shares must be positive.")
        if n_slices < 1:
            raise ValueError("n_slices must be at least 1.")

        base = total_shares / n_slices
        schedule = np.full(n_slices, base)
        schedule = self._apply_lot_rounding(schedule, total_shares)
        logger.debug("twap_schedule: %d slices of %.4f shares each", n_slices, base)
        return schedule

    # ------------------------------------------------------------------
    # VWAP Schedule
    # ------------------------------------------------------------------

    def vwap_schedule(
        self,
        total_shares: float,
        volume_profile: np.ndarray,
    ) -> np.ndarray:
        """Volume-Weighted Average Price (VWAP) execution schedule.

        Allocates ``total_shares`` proportionally to the intraday volume profile.

        Parameters
        ----------
        total_shares : float
            Total number of shares to trade.
        volume_profile : np.ndarray
            Expected volume fractions per time slice (need not sum to 1;
            normalisation is applied internally). Shape ``(n_slices,)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_slices,)`` with share quantities proportional
            to ``volume_profile``.

        Raises
        ------
        ValueError
            If ``total_shares <= 0`` or ``volume_profile`` is empty or non-positive.
        """
        if total_shares <= 0:
            raise ValueError("total_shares must be positive.")
        vp = np.asarray(volume_profile, dtype=float)
        if vp.size == 0:
            raise ValueError("volume_profile must not be empty.")
        if np.any(vp < 0):
            raise ValueError("volume_profile must contain non-negative values.")
        if vp.sum() < 1e-14:
            raise ValueError("volume_profile must have a positive total.")

        weights = vp / vp.sum()
        schedule = weights * total_shares
        schedule = self._apply_lot_rounding(schedule, total_shares)
        logger.debug("vwap_schedule: %d slices, max_slice=%.4f", len(schedule), schedule.max())
        return schedule

    # ------------------------------------------------------------------
    # Linear Slippage / Market Impact
    # ------------------------------------------------------------------

    def slippage_linear(
        self,
        shares: float,
        adv: float,
        impact_coeff: float = 0.1,
    ) -> float:
        """Linear market-impact (slippage) model.

        Estimates execution slippage as a fraction of price proportional to
        the participation rate:

        ``slippage = impact_coeff * (shares / adv)``

        Parameters
        ----------
        shares : float
            Number of shares to trade.
        adv : float
            Average daily volume (shares).
        impact_coeff : float, optional
            Proportionality constant. Default 0.1.

        Returns
        -------
        float
            Slippage as a fraction of price (e.g., 0.001 = 10 bps).

        Raises
        ------
        ValueError
            If ``adv <= 0``.
        """
        if adv <= 0:
            raise ValueError("adv must be positive.")

        participation = abs(shares) / adv
        slip = impact_coeff * participation
        logger.debug("slippage_linear: participation=%.4f, slippage=%.6f", participation, slip)
        return float(slip)

    # ------------------------------------------------------------------
    # Square-Root Slippage / Market Impact
    # ------------------------------------------------------------------

    def slippage_sqrt(
        self,
        shares: float,
        adv: float,
        sigma: float,
        impact_coeff: float = 0.1,
    ) -> float:
        """Square-root market-impact model.

        Uses the empirical square-root law for price impact:

        ``slippage = impact_coeff * sigma * sqrt(shares / adv)``

        Parameters
        ----------
        shares : float
            Number of shares to trade.
        adv : float
            Average daily volume (shares).
        sigma : float
            Daily return volatility of the asset.
        impact_coeff : float, optional
            Proportionality constant. Default 0.1.

        Returns
        -------
        float
            Slippage as a fraction of price.

        Raises
        ------
        ValueError
            If ``adv <= 0`` or ``sigma < 0``.
        """
        if adv <= 0:
            raise ValueError("adv must be positive.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")

        participation = abs(shares) / adv
        slip = impact_coeff * sigma * np.sqrt(participation)
        logger.debug("slippage_sqrt: participation=%.4f, sigma=%.4f, slippage=%.6f",
                     participation, sigma, slip)
        return float(slip)

    # ------------------------------------------------------------------
    # Kyle's Lambda
    # ------------------------------------------------------------------

    def kyle_lambda(
        self,
        prices: pd.Series,
        volumes: pd.Series,
    ) -> float:
        """Estimate Kyle's lambda (price-impact coefficient) via OLS regression.

        Regresses signed price changes on signed order flow (volume proxy):

        ``delta_p = lambda * sign(delta_v) * |delta_v| + epsilon``

        Parameters
        ----------
        prices : pd.Series
            Transaction or mid-quote price time series.
        volumes : pd.Series
            Signed trade volume time series (positive = buy, negative = sell).
            Must have the same index as ``prices``.

        Returns
        -------
        float
            Kyle's lambda estimate (in price units per share).

        Raises
        ------
        ValueError
            If series lengths do not match or fewer than 2 observations exist.
        """
        if len(prices) != len(volumes):
            raise ValueError("prices and volumes must have the same length.")
        if len(prices) < 2:
            raise ValueError("At least 2 observations required.")

        delta_p = prices.diff().dropna().values
        delta_v = volumes.diff().dropna().values

        # Align lengths (both should be same after diff)
        min_len = min(len(delta_p), len(delta_v))
        delta_p = delta_p[:min_len]
        delta_v = delta_v[:min_len]

        # Signed order flow
        signed_flow = np.sign(delta_v) * np.sqrt(np.abs(delta_v) + 1e-14)

        # OLS: delta_p = lambda * signed_flow
        numerator = float(np.sum(signed_flow * delta_p))
        denominator = float(np.sum(signed_flow ** 2))

        if abs(denominator) < 1e-14:
            logger.warning("kyle_lambda: near-zero denominator; returning 0.")
            return 0.0

        lam = numerator / denominator
        logger.debug("kyle_lambda: lambda=%.8f", lam)
        return float(lam)

    # ------------------------------------------------------------------
    # Transaction Cost Analysis
    # ------------------------------------------------------------------

    def transaction_cost_analysis(
        self,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute standard TCA metrics for a set of orders and fills.

        Parameters
        ----------
        orders : pd.DataFrame
            Order records with columns:
            ``['order_id', 'symbol', 'side', 'quantity', 'arrival_price',
               'decision_price', 'benchmark_vwap']``.
        fills : pd.DataFrame
            Fill records with columns:
            ``['order_id', 'fill_price', 'fill_quantity', 'fill_time']``.
            Multiple fills per order are aggregated.

        Returns
        -------
        pd.DataFrame
            TCA report with columns:
            ``['order_id', 'symbol', 'side', 'quantity', 'arrival_price',
               'avg_fill_price', 'implementation_shortfall_bps',
               'arrival_slippage_bps', 'vwap_slippage_bps',
               'total_cost_bps', 'fill_rate']``.

        Raises
        ------
        ValueError
            If required columns are missing from ``orders`` or ``fills``.
        """
        required_order_cols = {
            "order_id", "symbol", "side", "quantity",
            "arrival_price", "decision_price", "benchmark_vwap",
        }
        required_fill_cols = {"order_id", "fill_price", "fill_quantity"}

        missing_order = required_order_cols - set(orders.columns)
        if missing_order:
            raise ValueError(f"orders missing columns: {missing_order}")
        missing_fill = required_fill_cols - set(fills.columns)
        if missing_fill:
            raise ValueError(f"fills missing columns: {missing_fill}")

        # Aggregate fills per order: volume-weighted average fill price
        fill_agg = (
            fills.groupby("order_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "avg_fill_price": (
                            (g["fill_price"] * g["fill_quantity"]).sum()
                            / g["fill_quantity"].sum()
                        )
                        if g["fill_quantity"].sum() > 0
                        else np.nan,
                        "filled_quantity": g["fill_quantity"].sum(),
                    }
                )
            )
            .reset_index()
        )

        report = orders.merge(fill_agg, on="order_id", how="left")

        def _bps(px_a: float, px_b: float, side: str) -> float:
            """Return signed basis-point cost (positive = adverse)."""
            if px_b < 1e-14 or np.isnan(px_a) or np.isnan(px_b):
                return np.nan
            direction = 1.0 if str(side).lower() == "buy" else -1.0
            return direction * (px_a - px_b) / px_b * 1e4

        report["implementation_shortfall_bps"] = report.apply(
            lambda r: _bps(r["avg_fill_price"], r["decision_price"], r["side"]), axis=1
        )
        report["arrival_slippage_bps"] = report.apply(
            lambda r: _bps(r["avg_fill_price"], r["arrival_price"], r["side"]), axis=1
        )
        report["vwap_slippage_bps"] = report.apply(
            lambda r: _bps(r["avg_fill_price"], r["benchmark_vwap"], r["side"]), axis=1
        )
        report["commission_bps"] = (self.commission_per_share / report["avg_fill_price"]) * 1e4
        report["total_cost_bps"] = report["arrival_slippage_bps"] + report["commission_bps"]
        report["fill_rate"] = report["filled_quantity"] / report["quantity"]

        output_cols = [
            "order_id", "symbol", "side", "quantity", "arrival_price",
            "avg_fill_price", "implementation_shortfall_bps",
            "arrival_slippage_bps", "vwap_slippage_bps",
            "total_cost_bps", "fill_rate",
        ]
        # Keep only columns that exist (commission_bps is internal)
        output_cols = [c for c in output_cols if c in report.columns]
        result = report[output_cols].copy()
        logger.info("transaction_cost_analysis: processed %d orders", len(result))
        return result

    # ------------------------------------------------------------------
    # Simulate Fill
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        order_shares: float,
        bid: float,
        ask: float,
        adv: float,
        sigma: float,
    ) -> dict:
        """Simulate an order fill with realistic market-impact slippage.

        Uses the square-root impact model to estimate execution slippage and
        adjusts the mid-price accordingly.  Partial fills are generated when
        the order exceeds a fraction of ADV.

        Parameters
        ----------
        order_shares : float
            Number of shares in the order (positive = buy, negative = sell).
        bid : float
            Current best bid price.
        ask : float
            Current best ask price.
        adv : float
            Average daily volume in shares.
        sigma : float
            Daily return volatility of the asset.

        Returns
        -------
        dict
            Keys:
            ``'fill_price'`` – volume-weighted average execution price,
            ``'fill_shares'`` – actual shares filled (may be less than ``|order_shares|``),
            ``'slippage_bps'`` – execution slippage in basis points,
            ``'commission'``  – total commission in currency units,
            ``'side'``        – ``'buy'`` or ``'sell'``.

        Raises
        ------
        ValueError
            If ``bid`` or ``ask`` are non-positive, or ``bid >= ask``.
        """
        if bid <= 0 or ask <= 0:
            raise ValueError("bid and ask must be positive.")
        if bid >= ask:
            raise ValueError("bid must be strictly less than ask.")
        if adv <= 0:
            raise ValueError("adv must be positive.")

        mid = (bid + ask) / 2.0
        half_spread = (ask - bid) / 2.0
        is_buy = order_shares >= 0
        side = "buy" if is_buy else "sell"
        abs_shares = abs(order_shares)

        # Participation cap: max 20% of ADV per fill
        max_fill = 0.20 * adv
        fill_shares = min(abs_shares, max_fill)

        # Market impact via sqrt model
        impact_frac = self.slippage_sqrt(fill_shares, adv, sigma, impact_coeff=0.1)
        spread_cost = half_spread / mid  # as fraction

        total_slippage_frac = impact_frac + spread_cost
        direction = 1.0 if is_buy else -1.0
        fill_price = mid * (1.0 + direction * total_slippage_frac)
        fill_price = max(fill_price, 1e-6)  # guard against negative price

        slippage_bps = total_slippage_frac * 1e4
        commission = fill_shares * self.commission_per_share

        result = {
            "fill_price": float(fill_price),
            "fill_shares": float(fill_shares),
            "slippage_bps": float(slippage_bps),
            "commission": float(commission),
            "side": side,
        }
        logger.debug(
            "simulate_fill: side=%s, fill_price=%.4f, fill_shares=%.0f, slippage_bps=%.2f",
            side, fill_price, fill_shares, slippage_bps,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_lot_rounding(
        self,
        schedule: np.ndarray,
        total_shares: float,
    ) -> np.ndarray:
        """Round each slice to the nearest lot size and correct rounding error.

        The last slice absorbs any residual from rounding so that the total
        always sums to ``total_shares`` (within floating-point tolerance).

        Parameters
        ----------
        schedule : np.ndarray
            Raw share quantities per slice.
        total_shares : float
            Target total that the schedule must sum to.

        Returns
        -------
        np.ndarray
            Lot-rounded schedule with corrected total.
        """
        if self.lot_size <= 1:
            return schedule.copy()

        rounded = (np.round(schedule / self.lot_size) * self.lot_size).astype(float)
        # Assign remaining shares to the largest-slice bucket
        residual = total_shares - rounded.sum()
        if len(rounded) > 0 and abs(residual) > 0:
            idx = int(np.argmax(rounded))
            rounded[idx] += round(residual / self.lot_size) * self.lot_size

        return rounded
