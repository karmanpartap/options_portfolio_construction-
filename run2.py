#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# run_heston_backtest.py (no catalog)
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import (
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
    ImportableStrategyConfig,
    LoggingConfig,
)

def main():
    strategies = [
        ImportableStrategyConfig(
            strategy_path="backtest_strategy:HestonMispricingStrategy",
            config_path="backtest_strategy:HestonMispricingStrategyConfig",
            config={
                "strategy_id": "HESTON-BACKTEST-001",
                "instrument_symbol": "TSLA",
                "databento_api_key": "db-DEG5Vr3msMNj7AijJ9TLrR8UkW75w", 
                "data_date": "2025-04-04",
                "risk_free": 0.0435,
                "max_days": 60,
                "write_csvs": True,
            },
        )
    ]

    engine_cfg = BacktestEngineConfig(
        logging=LoggingConfig(log_level="INFO", use_pyo3=False),
        strategies=strategies,
    )

    venues = [
        BacktestVenueConfig(
            name="SIM",
            oms_type="HEDGING",
            account_type="CASH",
            base_currency="USD",
            starting_balances=["1_000_000 USD"],
        )
    ]

    run_cfg = BacktestRunConfig(
        engine=engine_cfg,
        venues=venues,
        data=[],                     # <-- no catalog/data
        dispose_on_completion=True,
        raise_exception=True,
    )

    node = BacktestNode(configs=[run_cfg])
    try:
        node.run()   # This fires Strategy.on_start(), which runs your full pipeline
    finally:
        node.dispose()

if __name__ == "__main__":
    main()
