# Heston Mispricing Backtest

This repository contains a **quantitative backtesting pipeline** for detecting and exploiting option mispricings using the **Heston stochastic volatility model**, integrated with **Databento market data** and the **NautilusTrader** backtesting engine.

## Repository Structure

- **`backtest_strategy.py`**  
  Defines the **HestonMispricingStrategy**, which:
  - Fetches options data (definitions, trades, BBO) from Databento.  
  - Computes implied volatility and Greeks.  
  - Calibrates the Heston model parameters to option prices.  
  - Identifies mispriced contracts using multi-stage filtering.  
  - Runs portfolio optimization with constraints (budget, strike diversification, call/put ratio, etc.).  

- **`run2.py`**  
  Provides a **runner script** that sets up a `BacktestNode` with:
  - Strategy import from `backtest_strategy`.  
  - Configurable parameters (symbol, date, API key, horizon, etc.).  
  - A simulated venue with $1,000,000 starting balance.  
  - No catalog/data feed required (data is fetched directly from Databento).  

---

## Getting Started

### 1. Requirements
- Python 3.10+
- Dependencies:
  - `numpy`, `pandas`, `scipy`, `numba`
  - `databento`
  - `pulp`
  - `nautilus-trader`

Install with:

```bash
pip install numpy pandas scipy numba databento pulp nautilus-trader
```

### 2. Environment Variables
Set your **Databento API key**:

```bash
export DATABENTO_API_KEY="your_api_key_here"
```

Alternatively, provide it directly in the config inside `run2.py`.

---

## Usage

### Run Backtest
```bash
python run2.py
```

This will:
1. Trigger the strategy pipeline (`on_start`).  
2. Fetch TSLA option data for the configured date.  
3. Output Greeks and calibration CSVs (if `write_csvs=True`).  
4. Print optimization results (selected contracts, costs, premiums, scores).  

### Customize Parameters
Modify `run2.py`:
```python
config={
    "strategy_id": "HESTON-BACKTEST-001",
    "instrument_symbol": "TSLA",
    "databento_api_key": "your_api_key",
    "data_date": "2025-04-04",
    "risk_free": 0.0435,
    "max_days": 60,
    "write_csvs": True,
}
```

---

## Outputs
- `*_otm_with_bbo_greeks_<date>.csv` → computed Greeks.  
- `heston_calibration_from_last_trade_<date>.csv` → Heston parameter calibration.  
- `top10_otm_contracts_filtered-good.csv` → best mispriced contracts.  
- Console logs with optimization results (status, allocation, costs, premiums).  

---

##  Notes
- Default setup targets **TSLA options on 2025-04-04**.  
- Budget is **$1,000,000** with diversification and risk constraints.  
- Strategy can be reused with different symbols/dates by editing the config.  

---

## License
This project is released under the MIT License.
