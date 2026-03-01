"""Interactive Plotly Dash dashboard for the QuantVortex trading engine.

This module provides ``QuantVortexDashboard``, a self-contained Dash application
that visualises portfolio NAV, strategy allocations, risk metrics, regime
detection, correlation matrices, drawdowns, and rolling Sharpe ratios.

Examples
--------
>>> from visualization.dashboard import QuantVortexDashboard
>>> dash = QuantVortexDashboard()
>>> dash.run(debug=False, port=8050)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark colour palette
# ---------------------------------------------------------------------------
_BG_COLOR = "#0d1117"
_PAPER_COLOR = "#161b22"
_GRID_COLOR = "#30363d"
_TEXT_COLOR = "#c9d1d9"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#d29922"
_PURPLE = "#bc8cff"

_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": _PAPER_COLOR,
        "plot_bgcolor": _BG_COLOR,
        "font": {"color": _TEXT_COLOR, "family": "Inter, Roboto, sans-serif", "size": 12},
        "xaxis": {"gridcolor": _GRID_COLOR, "zerolinecolor": _GRID_COLOR},
        "yaxis": {"gridcolor": _GRID_COLOR, "zerolinecolor": _GRID_COLOR},
        "legend": {"bgcolor": _PAPER_COLOR, "bordercolor": _GRID_COLOR},
        "colorway": [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE,
                     "#e3b341", "#79c0ff", "#a5d6ff"],
    }
}


class QuantVortexDashboard:
    """Interactive Dash dashboard for the QuantVortex trading engine.

    Attributes
    ----------
    app : dash.Dash
        The underlying Dash application instance.

    Parameters
    ----------
    title : str, optional
        Browser tab title. Default ``"QuantVortex"``.

    Examples
    --------
    >>> from visualization.dashboard import QuantVortexDashboard
    >>> dashboard = QuantVortexDashboard()
    >>> dashboard.run(debug=True, port=8050)
    """

    def __init__(self, title: str = "QuantVortex") -> None:
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.CYBORG],
            title=title,
            suppress_callback_exceptions=True,
        )
        self._sample: dict[str, Any] = {}
        logger.info("QuantVortexDashboard initialised.")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def create_layout(self) -> html.Div:
        """Build the full dashboard layout.

        Returns
        -------
        html.Div
            The root Dash component containing all panels.
        """
        data = self.generate_sample_data()
        self._sample = data

        nav_fig = self.create_nav_chart(data["equity_df"], regimes=data["regimes"])
        alloc_fig = self.create_allocation_pie(data["allocations"])
        corr_fig = self.create_correlation_matrix(data["returns_df"])
        dd_sharpe_fig = self.create_drawdown_sharpe_combined(
            data["equity_curve"], data["returns"]
        )
        risk_fig = self.create_risk_heatmap(data["risk_metrics"])
        ohlc_fig = self.create_ohlc_chart(data["ohlc_df"])
        cum_ret_fig = self.create_cumulative_returns_chart(data["returns_df"])
        var_fig = self.create_var_histogram(data["returns"])

        _card_style = {"border": f"1px solid {_GRID_COLOR}"}

        header = dbc.Navbar(
            dbc.Container(
                [
                    html.Span(
                        "⚡ QuantVortex Trading Dashboard",
                        style={
                            "color": _ACCENT,
                            "fontSize": "1.4rem",
                            "fontWeight": "700",
                            "letterSpacing": "0.04em",
                        },
                    ),
                    html.Span(
                        id="live-clock",
                        style={"color": _TEXT_COLOR, "fontSize": "0.85rem"},
                    ),
                ],
                fluid=True,
                style={"display": "flex", "justifyContent": "space-between"},
            ),
            color="dark",
            dark=True,
            style={"borderBottom": f"1px solid {_GRID_COLOR}"},
        )

        # Row 0 — KPI ticker strip
        row0 = self.create_kpi_strip(data["kpi"])

        # Row 1 — NAV + Allocation
        row1 = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Portfolio NAV Curve", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=nav_fig,
                                    id="nav-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "380px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Strategy Allocation", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=alloc_fig,
                                    id="alloc-pie",
                                    config={"displayModeBar": False},
                                    style={"height": "380px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
            ],
            className="mt-3",
        )

        # Row 2 — OHLC | Cumulative Returns | VaR Histogram
        row2 = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("OHLC Candlestick + Volume", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=ohlc_fig,
                                    id="ohlc-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Strategy Cumulative Returns", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=cum_ret_fig,
                                    id="cum-returns-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Return Distribution & VaR", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=var_fig,
                                    id="var-histogram",
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
            ],
            className="mt-3",
        )

        # Row 3 — Risk Table | Correlation Matrix | Drawdown+Sharpe
        row3 = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Risk Metrics", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=risk_fig,
                                    id="risk-table",
                                    config={"displayModeBar": False},
                                    style={"height": "360px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Correlation Matrix", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=corr_fig,
                                    id="corr-matrix",
                                    config={"displayModeBar": False},
                                    style={"height": "360px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Drawdown & Rolling Sharpe", style={"color": _ACCENT}),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=dd_sharpe_fig,
                                    id="drawdown-sharpe",
                                    config={"displayModeBar": False},
                                    style={"height": "360px"},
                                )
                            ),
                        ],
                        style=_card_style,
                    ),
                    width=4,
                ),
            ],
            className="mt-3 mb-2",
        )

        footer = html.Div(
            "QuantVortex v1.0.0 • Data is synthetic for demonstration",
            style={
                "color": _GRID_COLOR,
                "textAlign": "center",
                "padding": "12px",
                "fontSize": "0.75rem",
                "borderTop": f"1px solid {_GRID_COLOR}",
                "marginTop": "12px",
            },
        )

        layout = html.Div(
            [
                header,
                dbc.Container([row0, row1, row2, row3, footer], fluid=True),
                dcc.Interval(id="interval", interval=5_000, n_intervals=0),
            ],
            style={"backgroundColor": _BG_COLOR, "minHeight": "100vh"},
        )
        logger.debug("Dashboard layout created.")
        return layout

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def create_nav_chart(
        self,
        equity_data: pd.DataFrame,
        regimes: pd.Series | None = None,
    ) -> go.Figure:
        """Build a line chart of portfolio NAV (and optional benchmark).

        Parameters
        ----------
        equity_data : pd.DataFrame
            DataFrame with a DatetimeIndex and at least one column ``"portfolio"``.
            Optional additional columns are plotted as overlays.
        regimes : pd.Series, optional
            Integer regime labels (0=low-vol, 1=normal, 2=high-vol) with the
            same DatetimeIndex as *equity_data*.  When provided, background
            shading is added for regimes 0 and 2.

        Returns
        -------
        go.Figure
            Plotly figure with filled area traces.
        """
        fig = go.Figure()

        palette = [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE]
        for i, col in enumerate(equity_data.columns):
            is_portfolio = col.lower() == "portfolio"
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data[col],
                    name=col,
                    mode="lines",
                    line={"color": color, "width": 2},
                    fill="tozeroy" if is_portfolio else "none",
                    fillcolor=f"rgba(88,166,255,0.08)" if is_portfolio else None,
                    hovertemplate="%{x|%Y-%m-%d}<br>NAV: %{y:,.2f}<extra></extra>",
                )
            )

        # Regime background shading
        if regimes is not None:
            regime_colors = {
                0: "rgba(63,185,80,0.05)",   # low vol — subtle green
                2: "rgba(248,81,73,0.08)",   # high vol — subtle red
            }
            dates = regimes.index
            for target_regime, fill_color in regime_colors.items():
                in_regime = False
                start_date = None
                for i, (date, reg) in enumerate(zip(dates, regimes)):
                    if reg == target_regime and not in_regime:
                        in_regime = True
                        start_date = date
                    elif reg != target_regime and in_regime:
                        fig.add_vrect(
                            x0=start_date,
                            x1=date,
                            fillcolor=fill_color,
                            line_width=0,
                            layer="below",
                        )
                        in_regime = False
                if in_regime and start_date is not None:
                    fig.add_vrect(
                        x0=start_date,
                        x1=dates[-1],
                        fillcolor=fill_color,
                        line_width=0,
                        layer="below",
                    )

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=380,
            margin={"l": 55, "r": 20, "t": 20, "b": 40},
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="NAV",
            showlegend=len(equity_data.columns) > 1,
        )
        logger.debug("create_nav_chart: %d series, %d points", len(equity_data.columns), len(equity_data))
        return fig

    def create_allocation_pie(self, allocations: dict) -> go.Figure:
        """Build a donut chart showing strategy allocation.

        Parameters
        ----------
        allocations : dict
            Mapping of strategy name → weight (values need not sum to 1).

        Returns
        -------
        go.Figure
            Plotly donut chart.
        """
        labels = list(allocations.keys())
        values = [max(float(v), 0.0) for v in allocations.values()]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1%}<extra></extra>",
                marker={
                    "colors": [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE,
                               "#e3b341", "#79c0ff"],
                    "line": {"color": _BG_COLOR, "width": 2},
                },
            )
        )
        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=380,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            showlegend=True,
        )
        logger.debug("create_allocation_pie: %d strategies", len(labels))
        return fig

    def create_risk_heatmap(self, risk_metrics: pd.DataFrame) -> go.Figure:
        """Build a colour-coded table of risk metrics.

        Parameters
        ----------
        risk_metrics : pd.DataFrame
            DataFrame with columns ``["Metric", "Value", "Status"]``.

        Returns
        -------
        go.Figure
            Plotly table figure with conditional cell colouring.
        """
        status_colors = {
            "Good": _GREEN,
            "Warning": _ORANGE,
            "Bad": _RED,
            "Neutral": _TEXT_COLOR,
        }

        cell_colors = [
            [_PAPER_COLOR] * len(risk_metrics),
            [_PAPER_COLOR] * len(risk_metrics),
            [status_colors.get(str(s), _TEXT_COLOR) for s in risk_metrics["Status"]],
        ]

        fig = go.Figure(
            go.Table(
                header={
                    "values": ["<b>Metric</b>", "<b>Value</b>", "<b>Status</b>"],
                    "fill_color": _GRID_COLOR,
                    "font": {"color": _ACCENT, "size": 12},
                    "align": "left",
                },
                cells={
                    "values": [
                        risk_metrics["Metric"].tolist(),
                        risk_metrics["Value"].tolist(),
                        risk_metrics["Status"].tolist(),
                    ],
                    "fill_color": cell_colors,
                    "font": {"color": _TEXT_COLOR, "size": 11},
                    "align": ["left", "right", "center"],
                    "height": 28,
                },
            )
        )
        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=360,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        logger.debug("create_risk_heatmap: %d rows", len(risk_metrics))
        return fig

    def create_correlation_matrix(self, returns: pd.DataFrame) -> go.Figure:
        """Build a correlation-matrix heatmap.

        Parameters
        ----------
        returns : pd.DataFrame
            Returns DataFrame (DatetimeIndex × asset columns).

        Returns
        -------
        go.Figure
            Plotly heatmap with diverging colour scale.
        """
        corr = returns.corr()
        mask = np.tril(np.ones_like(corr, dtype=bool))
        corr_masked = corr.where(mask)

        fig = go.Figure(
            go.Heatmap(
                z=corr_masked.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.where(
                    mask, corr.round(2).values.astype(str), ""
                ),
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
                colorbar={"title": {"text": "ρ", "font": {"color": _TEXT_COLOR}}},
            )
        )
        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=360,
            margin={"l": 60, "r": 20, "t": 20, "b": 60},
        )
        fig.update_xaxes(tickangle=-45)
        logger.debug("create_correlation_matrix: %d assets", len(corr.columns))
        return fig

    def create_drawdown_chart(self, equity_curve: pd.Series) -> go.Figure:
        """Build a filled-area drawdown chart.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity indexed by date.

        Returns
        -------
        go.Figure
            Plotly figure showing underwater equity curve.
        """
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100.0

        fig = go.Figure(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                fillcolor=f"rgba(248,81,73,0.25)",
                line={"color": _RED, "width": 1.5},
                name="Drawdown",
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=280,
            margin={"l": 55, "r": 20, "t": 20, "b": 40},
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        fig.update_yaxes(ticksuffix="%")
        logger.debug("create_drawdown_chart: max_dd=%.2f%%", float(drawdown.min()))
        return fig

    def create_rolling_sharpe_chart(
        self, returns: pd.Series, window: int = 252
    ) -> go.Figure:
        """Build a rolling Sharpe ratio chart.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
        window : int, optional
            Rolling window in trading days. Default 252.

        Returns
        -------
        go.Figure
            Plotly line chart with a zero reference line.
        """
        ann_factor = np.sqrt(252)
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std(ddof=1)
        ) * ann_factor

        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color=_GRID_COLOR)
        fig.add_hline(y=1, line_dash="dot", line_color=_GREEN, opacity=0.4)

        # Colour positive / negative regions differently
        pos = rolling_sharpe.clip(lower=0)
        neg = rolling_sharpe.clip(upper=0)

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=pos,
                fill="tozeroy",
                fillcolor=f"rgba(63,185,80,0.15)",
                line={"color": _GREEN, "width": 1.5},
                name="Sharpe > 0",
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=neg,
                fill="tozeroy",
                fillcolor=f"rgba(248,81,73,0.15)",
                line={"color": _RED, "width": 1.5},
                name="Sharpe < 0",
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=280,
            margin={"l": 55, "r": 20, "t": 20, "b": 40},
            xaxis_title="Date",
            yaxis_title=f"Sharpe ({window}d)",
            showlegend=False,
        )
        logger.debug(
            "create_rolling_sharpe_chart: window=%d, mean=%.3f",
            window,
            float(rolling_sharpe.dropna().mean()),
        )
        return fig

    def create_drawdown_sharpe_combined(
        self, equity_curve: pd.Series, returns: pd.Series, window: int = 252
    ) -> go.Figure:
        """Build a combined Drawdown / Rolling-Sharpe subplot figure.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity indexed by date.
        returns : pd.Series
            Daily returns series.
        window : int, optional
            Rolling window for Sharpe calculation.  Default 252.

        Returns
        -------
        go.Figure
            Plotly figure with two vertically stacked subplots sharing the
            x-axis.
        """
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100.0

        ann_factor = np.sqrt(252)
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std(ddof=1)
        ) * ann_factor
        pos = rolling_sharpe.clip(lower=0)
        neg = rolling_sharpe.clip(upper=0)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
        )

        # -- Drawdown (top subplot) ----------------------------------------
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.25)",
                line={"color": _RED, "width": 1.5},
                name="Drawdown",
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # -- Rolling Sharpe (bottom subplot) ----------------------------------
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=pos,
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.15)",
                line={"color": _GREEN, "width": 1.5},
                name="Sharpe > 0",
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=neg,
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.15)",
                line={"color": _RED, "width": 1.5},
                name="Sharpe < 0",
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color=_GRID_COLOR, row=2, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color=_GREEN, opacity=0.4, row=2, col=1)

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=360,
            margin={"l": 55, "r": 20, "t": 10, "b": 40},
            hovermode="x unified",
            showlegend=False,
        )
        fig.update_yaxes(title_text="DD (%)", ticksuffix="%", row=1, col=1)
        fig.update_yaxes(title_text=f"Sharpe ({window}d)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        logger.debug(
            "create_drawdown_sharpe_combined: max_dd=%.2f%%, mean_sharpe=%.3f",
            float(drawdown.min()),
            float(rolling_sharpe.dropna().mean()),
        )
        return fig

    def create_kpi_strip(self, kpi: dict) -> dbc.Row:
        """Build the KPI ticker strip row.

        Parameters
        ----------
        kpi : dict
            Mapping of KPI name → dict with ``value`` (display string) and
            ``color`` (hex colour string).

        Returns
        -------
        dbc.Row
            A Bootstrap row of equal-width KPI cards.
        """
        cols = []
        for label, meta in kpi.items():
            color = meta["color"]
            card = dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            label,
                            style={
                                "color": _GRID_COLOR,
                                "fontSize": "0.7rem",
                                "marginBottom": "2px",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.05em",
                            },
                        ),
                        html.Div(
                            meta["value"],
                            style={
                                "color": color,
                                "fontSize": "1.1rem",
                                "fontWeight": "700",
                            },
                        ),
                    ],
                    style={"padding": "8px 12px"},
                ),
                style={
                    "backgroundColor": _PAPER_COLOR,
                    "border": f"1px solid {_GRID_COLOR}",
                    "borderLeft": f"3px solid {color}",
                },
            )
            cols.append(dbc.Col(card, className="px-1"))

        return dbc.Row(cols, className="mt-2 g-1")

    def create_ohlc_chart(self, ohlc_df: pd.DataFrame) -> go.Figure:
        """Build a candlestick chart with volume bars.

        Parameters
        ----------
        ohlc_df : pd.DataFrame
            DataFrame with columns Open, High, Low, Close, Volume and a
            DatetimeIndex.

        Returns
        -------
        go.Figure
            Plotly figure with candlestick (top) and volume bars (bottom).
        """
        df = ohlc_df.tail(126).copy()
        up_mask = df["Close"] >= df["Open"]
        vol_colors = [_GREEN if u else _RED for u in up_mask]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=_GREEN,
                decreasing_line_color=_RED,
                name="OHLC",
                hovertext=[
                    f"O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}"
                    for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
                ],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=vol_colors,
                name="Volume",
                hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:,.0f}<extra></extra>",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=320,
            margin={"l": 55, "r": 20, "t": 10, "b": 40},
            hovermode="x unified",
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        logger.debug("create_ohlc_chart: %d candles", len(df))
        return fig

    def create_cumulative_returns_chart(self, returns_df: pd.DataFrame) -> go.Figure:
        """Build a multi-line cumulative returns chart.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily returns DataFrame with strategy columns.

        Returns
        -------
        go.Figure
            Plotly figure showing each strategy's cumulative return.
        """
        colorway = [_ACCENT, _GREEN, _ORANGE, _RED, _PURPLE,
                    "#e3b341", "#79c0ff", "#a5d6ff"]
        cum_ret = (1 + returns_df).cumprod() - 1

        # Identify best-performing strategy for filled area
        best_col = cum_ret.iloc[-1].idxmax()

        fig = go.Figure()
        for i, col in enumerate(cum_ret.columns):
            color = colorway[i % len(colorway)]
            fig.add_trace(
                go.Scatter(
                    x=cum_ret.index,
                    y=cum_ret[col] * 100,
                    name=col,
                    mode="lines",
                    line={"color": color, "width": 1.8},
                    fill="tozeroy" if col == best_col else "none",
                    fillcolor=f"rgba(88,166,255,0.06)" if col == best_col else None,
                    hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}<br>%{{y:+.2f}}%<extra></extra>",
                )
            )

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=320,
            margin={"l": 55, "r": 20, "t": 10, "b": 40},
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
        )
        fig.update_layout(
            legend={
                "x": 0.01,
                "y": 0.99,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": _PAPER_COLOR,
                "bordercolor": _GRID_COLOR,
            }
        )
        fig.update_yaxes(ticksuffix="%")
        logger.debug("create_cumulative_returns_chart: %d strategies", len(cum_ret.columns))
        return fig

    def create_var_histogram(self, returns: pd.Series) -> go.Figure:
        """Build a return distribution histogram with VaR/CVaR overlays.

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.

        Returns
        -------
        go.Figure
            Plotly figure with histogram, normal-fit overlay, and VaR lines.
        """
        ret_vals = returns.dropna().values
        var_95 = float(-np.percentile(ret_vals, 5))
        cvar_mask = ret_vals < -var_95
        cvar_95 = float(-ret_vals[cvar_mask].mean()) if cvar_mask.any() else var_95

        mu, sigma = float(ret_vals.mean()), float(ret_vals.std(ddof=1))
        x_range = np.linspace(ret_vals.min(), ret_vals.max(), 300)
        pdf_vals = norm.pdf(x_range, mu, sigma)

        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=ret_vals * 100,
                nbinsx=80,
                name="Daily Returns",
                marker_color=f"rgba(88,166,255,0.5)",
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
            )
        )

        # Scale pdf to histogram counts
        bin_width = (ret_vals.max() - ret_vals.min()) / 80 * 100
        pdf_scaled = pdf_vals * len(ret_vals) * bin_width

        fig.add_trace(
            go.Scatter(
                x=x_range * 100,
                y=pdf_scaled,
                name="Normal fit",
                mode="lines",
                line={"color": _ORANGE, "width": 2, "dash": "dot"},
                hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.1f}<extra></extra>",
            )
        )

        # VaR line
        fig.add_vline(
            x=-var_95 * 100,
            line_dash="dash",
            line_color=_RED,
            annotation_text=f"VaR 95%: {var_95:.2%}",
            annotation_font_color=_RED,
            annotation_position="top right",
        )

        # CVaR line
        fig.add_vline(
            x=-cvar_95 * 100,
            line_dash="dash",
            line_color=_ORANGE,
            annotation_text=f"CVaR 95%: {cvar_95:.2%}",
            annotation_font_color=_ORANGE,
            annotation_position="top left",
        )

        fig.update_layout(
            **_DARK_TEMPLATE["layout"],
            height=320,
            margin={"l": 55, "r": 20, "t": 30, "b": 40},
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            showlegend=True,
            barmode="overlay",
        )
        fig.update_layout(legend={"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"})
        fig.update_xaxes(ticksuffix="%")
        logger.debug("create_var_histogram: VaR=%.3f%% CVaR=%.3f%%", var_95 * 100, cvar_95 * 100)
        return fig

    # ------------------------------------------------------------------
    # Sample data generator
    # ------------------------------------------------------------------

    def generate_sample_data(self) -> dict:
        """Generate synthetic demo data for all dashboard panels.

        Uses a seeded random number generator so the output is
        deterministic across runs.

        Returns
        -------
        dict
            Keys:
            - ``"equity_curve"`` : pd.Series (portfolio NAV)
            - ``"equity_df"``    : pd.DataFrame (portfolio + benchmark NAV)
            - ``"returns"``      : pd.Series (daily returns)
            - ``"returns_df"``   : pd.DataFrame (multi-asset returns)
            - ``"allocations"``  : dict (strategy weights)
            - ``"risk_metrics"`` : pd.DataFrame (metric table)
            - ``"regimes"``      : pd.Series (integer regime labels)
            - ``"ohlc_df"``      : pd.DataFrame (OHLC + Volume data)
            - ``"kpi"``          : dict (KPI ticker strip values)
        """
        rng = np.random.default_rng(42)
        n = 756  # ~3 years daily
        dates = pd.bdate_range("2021-01-04", periods=n)

        # --- Portfolio NAV -----------------------------------------------
        daily_ret = rng.normal(0.0005, 0.012, n)
        equity_curve = pd.Series(
            100.0 * np.cumprod(1 + daily_ret), index=dates, name="portfolio"
        )

        # Benchmark (S&P-like drift)
        bm_ret = rng.normal(0.0003, 0.010, n)
        bm_curve = pd.Series(
            100.0 * np.cumprod(1 + bm_ret), index=dates, name="benchmark"
        )
        equity_df = pd.concat([equity_curve, bm_curve], axis=1)
        returns = pd.Series(daily_ret, index=dates, name="returns")

        # --- Multi-asset returns ------------------------------------------
        asset_names = ["StatArb", "Momentum", "MeanRev", "MLAlpha", "Vol Arb"]
        corr_base = np.array(
            [
                [1.00, 0.15, 0.20, 0.10, -0.05],
                [0.15, 1.00, 0.35, 0.25, 0.00],
                [0.20, 0.35, 1.00, 0.30, 0.05],
                [0.10, 0.25, 0.30, 1.00, 0.08],
                [-0.05, 0.00, 0.05, 0.08, 1.00],
            ]
        )
        L = np.linalg.cholesky(corr_base)
        z = rng.standard_normal((n, 5))
        multi_ret = z @ L.T * 0.012
        multi_ret[:, 0] += 0.0006
        multi_ret[:, 1] += 0.0008
        multi_ret[:, 2] += 0.0004
        multi_ret[:, 3] += 0.0007
        multi_ret[:, 4] += 0.0002
        returns_df = pd.DataFrame(multi_ret, index=dates, columns=asset_names)

        # --- Strategy allocations ----------------------------------------
        allocations = {
            "StatArb": 0.30,
            "Momentum": 0.25,
            "MeanReversion": 0.20,
            "MLAlpha": 0.15,
            "VolArb": 0.10,
        }

        # --- Risk metrics ------------------------------------------------
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        sortino_denom = returns[returns < 0].std(ddof=1)
        sortino = float(returns.mean() / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else np.nan
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        var_95 = float(-np.percentile(daily_ret, 5))
        cvar_mask = daily_ret < -var_95
        cvar_95 = float(-daily_ret[cvar_mask].mean()) if cvar_mask.any() else var_95

        # Additional institutional metrics
        trading_days = n
        years = trading_days / 252.0
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        cagr = float((1 + total_return) ** (1.0 / years) - 1)
        ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
        win_rate = float((returns > 0).mean())
        pos_sum = float(returns[returns > 0].sum())
        neg_sum = float(abs(returns[returns < 0].sum()))
        profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.nan
        skewness = float(returns.skew())
        kurtosis = float(returns.kurt())

        def _status(metric: str, val: float) -> str:
            thresholds: dict[str, tuple] = {
                "Sharpe": (1.0, 0.5),
                "Sortino": (1.5, 0.8),
                "Max Drawdown": (-0.10, -0.20),
                "VaR 95%": (0.015, 0.025),
                "CVaR 95%": (0.020, 0.035),
                "CAGR": (0.10, 0.05),
                "Calmar Ratio": (1.0, 0.5),
                "Win Rate": (0.55, 0.45),
                "Profit Factor": (1.5, 1.0),
            }
            if metric not in thresholds:
                return "Neutral"
            good, warn = thresholds[metric]
            if metric in ("Max Drawdown",):
                return "Good" if val > good else ("Warning" if val > warn else "Bad")
            return "Good" if val >= good else ("Warning" if val >= warn else "Bad")

        rows = [
            ("Sharpe Ratio", f"{sharpe:.3f}", _status("Sharpe", sharpe)),
            ("Sortino Ratio", f"{sortino:.3f}", _status("Sortino", sortino)),
            ("Max Drawdown", f"{max_dd:.2%}", _status("Max Drawdown", max_dd)),
            ("VaR 95%", f"{var_95:.3%}", _status("VaR 95%", var_95)),
            ("CVaR 95%", f"{cvar_95:.3%}", _status("CVaR 95%", cvar_95)),
            ("CAGR", f"{cagr:.2%}", _status("CAGR", cagr)),
            ("Ann. Volatility", f"{ann_vol:.2%}", "Neutral"),
            ("Calmar Ratio", f"{calmar:.3f}", _status("Calmar Ratio", calmar)),
            ("Win Rate", f"{win_rate:.1%}", _status("Win Rate", win_rate)),
            ("Profit Factor", f"{profit_factor:.3f}", _status("Profit Factor", profit_factor)),
            ("Skewness", f"{skewness:.3f}", "Neutral"),
            ("Kurtosis", f"{kurtosis:.3f}", "Neutral"),
        ]
        risk_metrics = pd.DataFrame(rows, columns=["Metric", "Value", "Status"])

        # --- Regime labels (simple volatility-based) ---------------------
        vol_21 = returns.rolling(21).std()
        q33, q66 = vol_21.quantile([0.33, 0.66])
        regimes = pd.cut(
            vol_21,
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
        ).astype(float)
        regimes = regimes.fillna(1).astype(int)

        # --- OHLC data ---------------------------------------------------
        close_prices = equity_curve.values.copy()
        open_prices = np.empty(n)
        open_prices[0] = close_prices[0]
        open_prices[1:] = close_prices[:-1]
        _high_raw = close_prices * (1 + np.abs(rng.normal(0, 0.005, n)))
        _low_raw = close_prices * (1 - np.abs(rng.normal(0, 0.005, n)))
        # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
        high_prices = np.maximum(_high_raw, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(_low_raw, np.minimum(open_prices, close_prices))
        volumes = rng.integers(1_000_000, 50_000_000, n)
        ohlc_df = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
            },
            index=dates,
        )

        # --- KPI dict ----------------------------------------------------
        nav_val = float(equity_curve.iloc[-1])
        daily_pnl = float(daily_ret[-1])
        kpi = {
            "Portfolio NAV": {
                "value": f"${nav_val:,.2f}",
                "color": _ACCENT,
            },
            "Daily P&L": {
                "value": f"{daily_pnl:+.2%}",
                "color": _GREEN if daily_pnl >= 0 else _RED,
            },
            "Sharpe Ratio": {
                "value": f"{sharpe:.3f}",
                "color": _GREEN if sharpe >= 1.0 else (_ORANGE if sharpe >= 0.5 else _RED),
            },
            "Max Drawdown": {
                "value": f"{max_dd:.2%}",
                "color": _GREEN if max_dd > -0.10 else (_ORANGE if max_dd > -0.20 else _RED),
            },
            "VaR 95%": {
                "value": f"{var_95:.2%}",
                "color": _GREEN if var_95 < 0.015 else (_ORANGE if var_95 < 0.025 else _RED),
            },
            "Volatility (Ann.)": {
                "value": f"{ann_vol:.1%}",
                "color": _TEXT_COLOR,
            },
            "Win Rate": {
                "value": f"{win_rate:.1%}",
                "color": _GREEN if win_rate >= 0.55 else (_ORANGE if win_rate >= 0.45 else _RED),
            },
        }

        logger.info(
            "generate_sample_data: n=%d, sharpe=%.3f, max_dd=%.2f%%",
            n, sharpe, max_dd * 100,
        )
        return {
            "equity_curve": equity_curve,
            "equity_df": equity_df,
            "returns": returns,
            "returns_df": returns_df,
            "allocations": allocations,
            "risk_metrics": risk_metrics,
            "regimes": regimes,
            "ohlc_df": ohlc_df,
            "kpi": kpi,
        }

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def run(self, debug: bool = False, port: int = 8050) -> None:
        """Start the Dash development server.

        Parameters
        ----------
        debug : bool, optional
            Enable hot-reload and debug toolbar. Default ``False``.
        port : int, optional
            TCP port to listen on. Default 8050.
        """
        self.app.layout = self.create_layout()
        self._register_callbacks()
        logger.info("Starting QuantVortex dashboard on http://0.0.0.0:%d", port)
        self.app.run(debug=debug, host="0.0.0.0", port=port)

    def _register_callbacks(self) -> None:
        """Register all Dash callbacks."""

        @self.app.callback(
            Output("live-clock", "children"),
            Input("interval", "n_intervals"),
        )
        def update_clock(n_intervals: int) -> str:  # noqa: ARG001
            now = datetime.now(timezone.utc)
            ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            return f"🟢 LIVE  •  {ts}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = QuantVortexDashboard()
    dashboard.run(debug=True)
