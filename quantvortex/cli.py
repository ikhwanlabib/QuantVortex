"""Command-line interface for the QuantVortex trading engine.

Usage
-----
    quantvortex dashboard [--port PORT] [--debug]
    quantvortex backtest [--config CONFIG]

Examples
--------
    quantvortex dashboard --port 8050
    quantvortex backtest --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the interactive Dash dashboard."""
    from visualization.dashboard import QuantVortexDashboard

    dashboard = QuantVortexDashboard()
    dashboard.run(debug=args.debug, port=args.port)


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run the multi-strategy backtesting engine."""
    import yaml
    from backtesting.engine import BacktestEngine
    from backtesting.performance import PerformanceAnalytics

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    bt_cfg = cfg.get("backtesting", {})
    engine = BacktestEngine(
        initial_capital=bt_cfg.get("initial_capital", 10_000_000),
        commission_bps=bt_cfg.get("commission_bps", 5),
        slippage_bps=bt_cfg.get("slippage_bps", 10),
    )
    results = engine.run()
    analytics = PerformanceAnalytics(results)
    summary = analytics.compute_metrics()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``quantvortex`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="quantvortex",
        description="QuantVortex quantitative trading engine",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Launch the interactive dashboard")
    dash_parser.add_argument("--port", type=int, default=8050, help="TCP port (default: 8050)")
    dash_parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")

    # --- backtest ---
    bt_parser = subparsers.add_parser("backtest", help="Run the backtesting engine")
    bt_parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml)",
    )

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    if args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "backtest":
        _cmd_backtest(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
