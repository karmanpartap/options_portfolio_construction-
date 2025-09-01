#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 23:40:03 2025

@author: karmanpartap singh sidhu 
"""

# backtest_strategy.py
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import databento as db

from datetime import datetime
from typing import Tuple, Dict, Any

from scipy.stats import norm
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.integrate import quad

import numba as nb
from pulp import (
    LpProblem, LpVariable, lpSum,
    LpMaximize, LpInteger, LpBinary,
    LpStatus,
)

# Nautilus
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy


# ───────────────────────────────────────────────────────────
# 1) FETCH & COMPUTE GREEKS (adapted to accept symbol/date)
# ───────────────────────────────────────────────────────────
def fetch_and_compute_greeks(
    api_key: str,
    date: str,
    risk_free: float,
    underlying_symbol: str,
    output_csv: str | None = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Pull OPRA option defs + last trades + EOD BBO for `underlying_symbol`,
    compute IV/Greeks on the mid, and return (df, spot_price).
    """
    client = db.Historical(api_key)
    end_date = pd.to_datetime(date) + pd.Timedelta(days=1)

    # a) Option definitions
    defs = (
        client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            start=date,
            symbols=[f"{underlying_symbol}.OPT"],
            stype_in="parent",
            stype_out="instrument_id",
        )
        .to_df()
    )
    defs["expiration"] = pd.to_datetime(defs["expiration"], utc=True).dt.tz_localize(None)
    defs = defs[defs["expiration"] > pd.Timestamp(date)].copy()

    # b) Spot (close→last tick in window)
    spot_df = (
        client.timeseries.get_range(
            dataset="XNAS.BASIC",
            schema="trades",
            start=date,
            end=end_date,
            symbols=[underlying_symbol],
            stype_in="raw_symbol",
        )
        .to_df()
    )
    spot_price = (
        spot_df.sort_values("ts_recv")
        .groupby("symbol", as_index=False)
        .last()
        .iloc[0]["price"]
    )

    # c) OTM filter
    is_call = defs["instrument_class"] == "C"
    is_put = defs["instrument_class"] == "P"
    otm = defs[
        (is_call & (defs["strike_price"] > spot_price)) |
        (is_put  & (defs["strike_price"] < spot_price))
    ].copy()
    otm_ids = otm["instrument_id"].tolist()

    # d) Last trades
    trades = (
        client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="trades",
            start=date,
            end=end_date,
            symbols=[f"{underlying_symbol}.OPT"],
            stype_in="parent",
            stype_out="instrument_id",
        )
        .to_df()
    )
    last_trades = (
        trades.sort_values("ts_recv")
        .groupby("instrument_id", as_index=False)
        .last()[["instrument_id", "price"]]
        .rename(columns={"price": "last_trade_price"})
    )

    # e) End-of-day BBO (16:00:00–16:00:01)
    cbbo = (
        client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="cbbo-1s",
            start=f"{date}T16:00:00",
            end=f"{date}T16:00:01",
            symbols=[f"{underlying_symbol}.OPT"],
            stype_in="parent",
            stype_out="instrument_id",
        )
        .to_df()
    )
    eod_bbo = (
        cbbo[cbbo["instrument_id"].isin(otm_ids)]
        .sort_values("ts_recv")
        .groupby("instrument_id", as_index=False)
        .last()[["instrument_id", "bid_px_00", "ask_px_00"]]
        .rename(columns={"bid_px_00": "bid_price", "ask_px_00": "ask_price"})
    )
    eod_bbo["bid_ask_spread"] = eod_bbo["ask_price"] - eod_bbo["bid_price"]

    # f) Merge
    df = (
        otm
        .merge(last_trades, on="instrument_id", how="left")
        .merge(eod_bbo, on="instrument_id", how="left")
    )

    # g) Time to expiration (years)
    df["time_to_exp"] = (
        (df["expiration"] - pd.to_datetime(date))
        .dt.total_seconds() / (365 * 24 * 3600)
    )

    # h) Black–Scholes & Greeks (your original formulae)
    def bs_price(opt, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if opt == "C":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def bs_delta(opt, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return norm.cdf(d1) if opt == "C" else norm.cdf(d1) - 1

    def bs_gamma(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def implied_vol(mkt, opt, S, K, T, r):
        if np.isnan(mkt) or mkt <= 0 or T <= 0:
            return np.nan
        f = lambda vol: bs_price(opt, S, K, T, r, vol) - mkt
        try:
            return brentq(f, 1e-6, 5.0, maxiter=200)
        except ValueError:
            return np.nan

    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["iv"] = df.apply(lambda row: implied_vol(
        row["mid_price"],
        row["instrument_class"],
        spot_price,
        row["strike_price"],
        row["time_to_exp"],
        risk_free,
    ), axis=1)

    df["delta"] = df.apply(lambda row: bs_delta(
        row["instrument_class"],
        spot_price,
        row["strike_price"],
        row["time_to_exp"],
        risk_free,
        row["iv"],
    ), axis=1)

    df["gamma"] = df.apply(lambda row: bs_gamma(
        spot_price,
        row["strike_price"],
        row["time_to_exp"],
        risk_free,
        row["iv"],
    ), axis=1)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Wrote Greeks & IV → {output_csv}")

    return df, float(spot_price)


# ─────────────────────────────
# 2) HESTON CALIBRATION
# ─────────────────────────────
@nb.njit
def heston_char_func(u, T, kappa, theta, sigma, rho, v0):
    sigma = max(sigma, 0.01); v0 = max(v0, 0.01)
    a = kappa * theta
    b = kappa
    rho_sigma_u = rho * sigma * u * 1j
    d = np.sqrt((rho_sigma_u - b)**2 + sigma**2*(u*1j + u**2))
    g = (b - rho_sigma_u - d)/(b - rho_sigma_u + d + 1e-12)
    exp_term = g * np.exp(-d * T)
    log_exp  = np.log((1 - exp_term)/(1 - g + 1e-12))
    # NOTE: your original used 0.03 here (kept intact)
    C = 0.03*u*1j*T + (a/(sigma**2+1e-12))*((b - rho_sigma_u - d)*T - 2*log_exp)
    D = ((b - rho_sigma_u - d)/(sigma**2+1e-12))*(1 - np.exp(-d*T))/(1 - exp_term + 1e-12)
    return np.exp(C + D*v0)

def heston_price(S, K, T, r, kappa, theta, sigma, rho, v0, otype="call"):
    if T <= 1/365:
        return max(0, S-K) if otype=="call" else max(0, K-S)
    def integrand(u):
        k = np.log(S/K) + r*T
        cf = heston_char_func(u-0.5j, T, kappa, theta, sigma, rho, v0)
        return np.real(np.exp(-1j*u*k)*cf/(u**2 + 0.25))
    integral = quad(integrand, 1e-8, 100, limit=2000, epsabs=1e-8, epsrel=1e-8)[0]
    call_price = S - np.sqrt(S*K)*np.exp(-r*T/2)*integral/np.pi
    call_price = max(1e-10, min(call_price, S))
    if otype == "call":
        return call_price
    return call_price + K*np.exp(-r*T) - S

_global_S = None
_global_r = None
_global_data = None

def heston_objective(params):
    global _global_S, _global_r, _global_data
    kappa, theta, sigma, rho, v0 = params
    total_err = 0.0; cnt = 0
    # Feller penalty
    feller = 2*kappa*theta - sigma**2
    if feller < 0:
        total_err += 10 * abs(feller)
    for _, row in _global_data.iterrows():
        mkt = row["market_price"]
        model = heston_price(
            _global_S, row["strike"], row["T"],
            _global_r, *params, row["type"]
        )
        moneyness = abs(np.log(row["strike"]/_global_S))
        pw = 1/(mkt + 0.001)
        tw = 1/(row["T"] + 0.01)
        w  = min(500, max(5, moneyness*20 + pw*2 + tw))
        err = (model - mkt)/(mkt + 0.1)
        total_err += w * err * err
        cnt       += 1
    return total_err/cnt if cnt>0 else 1e12



def run_calibration(
    df: pd.DataFrame,
    spot_price: float,
    risk_free: float,
    cal_date: pd.Timestamp,
    max_days: int,
    output_csv: str | None = None,
):
    # rename & prepare
    df_cal = df.rename(columns={
        "strike_price":    "strike",
        "instrument_class":"iclass",
        "last_trade_price":"market_price",
        "instrument_id":   "ticker",   # ensure ticker exists
    })
    df_cal["type"] = df_cal["iclass"].map({"C":"call","P":"put"})
    df_cal["expiration_date"] = pd.to_datetime(df_cal["expiration"])
    df_cal["T"] = (df_cal["expiration_date"] - cal_date).dt.days / 365.0

    mask = (
        (df_cal["T"] > 0) &
        (df_cal["T"] <= max_days) &
        (df_cal["market_price"] > 0)
    )
    df_cal = df_cal.loc[mask].reset_index(drop=True)

    print(f"Calibrating on {len(df_cal)} short-dated OTM options…")
    params = calibrate_heston(spot_price, risk_free, df_cal)

    df_cal["model_price"] = df_cal.apply(
        lambda row: heston_price(
            spot_price, row["strike"], row["T"],
            risk_free, *params, row["type"]
        ), axis=1
    )
    df_cal["abs_err"] = (df_cal["model_price"] - df_cal["market_price"]).abs()
    df_cal["rel_err"] = df_cal["abs_err"] / df_cal["market_price"].clip(lower=0.01)

    if output_csv:
        df_cal.to_csv(output_csv, index=False)
        print(f"Wrote calibration results → {output_csv}")

    return df_cal, params


# ───────────────────────────────────────────────
# 3) MISPRICING METRICS & FILTERING (unchanged)
# ───────────────────────────────────────────────
MAX_DAYS_DEFAULT = 60  # used in filters as a fallback

def compute_and_filter_mispricing(df_cal, spot_price, risk_free, reference_dt, MAX_DAYS=MAX_DAYS_DEFAULT):
    df = df_cal.copy()
    df["days_to_expiration"] = (df["expiration_date"] - reference_dt).dt.days
    df["T_bs"]               = df["days_to_expiration"] / 252.0
    df["bid_ask_spread_pct"] = ((df["ask_price"] - df["bid_price"]).abs() / df["mid_price"]) * 100

    def bs_price_bs(S, K, T, r, vol, opt_type):
        if T <= 0 or vol <= 0:
            return max(0.0, (S-K) if opt_type=="call" else (K-S))
        d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol*math.sqrt(T))
        d2 = d1 - vol*math.sqrt(T)
        if opt_type=="call":
            return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        else:
            return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def implied_vol_from_price(price, S, K, T, r, opt_type):
        try:
            return brentq(
                lambda v: bs_price_bs(S, K, T, r, v, opt_type) - price,
                1e-6, 5.0
            )
        except ValueError:
            return np.nan

    df["heston_iv"] = df.apply(
        lambda row: implied_vol_from_price(
            row["model_price"], spot_price, row["strike"],
            row["T_bs"], risk_free, row["type"]
        ),
        axis=1
    )

    df["iv_difference"]    = df["iv"] - df["heston_iv"]
    df["overpricing"]      = df["market_price"] - df["model_price"]
    df["mispricing_score"] = df["iv_difference"] / df["bid_ask_spread_pct"]

    # Stage 2
    df2 = df[
        (df["market_price"] > df["model_price"]) &
        (df["market_price"] > 0.5) &
        (df["delta"].abs() < 0.10) &
        (df["bid_ask_spread_pct"] > 2) &
        (df["bid_ask_spread_pct"] < 7) &
        (df["days_to_expiration"] > 0) &
        (df["days_to_expiration"] <= MAX_DAYS)
    ].copy()
    top2 = df2.sort_values("overpricing", ascending=False).head(10)
    top2.to_csv("top_10_mispriced_contracts_filtered.csv", index=False)
    print("\n→ Top 10 “rich” per overpricing (stage 2):")
    print(top2[[
        "ticker","type","strike","expiration_date",
        "iv","heston_iv","iv_difference","market_price",
        "delta","bid_ask_spread_pct","overpricing","mispricing_score"
    ]].round(4).to_string(index=False))

    # Stage 3
    df3 = df[
        (df["market_price"] > df["model_price"]) &
        (df["delta"].abs() < 0.10) &
        (df["days_to_expiration"] <= 30) &
        (df["bid_ask_spread_pct"] > 0.1) &
        (df["bid_ask_spread_pct"] < 10) &
        (df["market_price"] > 1) &
        (df["iv"] > df["heston_iv"])
    ].copy()
    df3["mispricing"]   = df3["market_price"] - df3["model_price"]
    df3["profit_score"] = df3["iv"] * df3["mispricing"]
    top3 = df3.sort_values("profit_score", ascending=False).head(10).reset_index(drop=True)
    top3.to_csv("top10_otm_contracts_filtered-good.csv", index=False)
    print("\n→ Top 10 by profit_score (stage 3):")
    print(top3[[
        "ticker","type","strike","expiration_date",
        "delta","market_price","model_price",
        "iv","heston_iv","bid_ask_spread_pct",
        "mispricing","profit_score"
    ]].round(4).to_string(index=False))

    return top3


# ─────────────────────────────
# 4) PORTFOLIO OPTIMIZATION
# ─────────────────────────────
def optimize_portfolio(df_top):
    df = df_top.copy()
    SPOT_PRICE = float(df["strike"].mean())  
    CALL_RATIO = 0.75
    PUT_RATIO  = 0.25
    MIN_DISTINCT_STRIKES     = 4
    MIN_CONTRACTS_PER_STRIKE = 4
    MIN_CALL_CONTRACTS       = 2
    RISK_WEIGHT  = 20
    DIST_WEIGHT  = 50

    df["premium"]  = df["market_price"].astype(float)
    df["risk"]     = df["delta"].abs()
    df["distance"] = (df["strike"] - SPOT_PRICE).abs() / SPOT_PRICE

    def compute_cost(r):
        if r["type"].lower() == "call":
            return 100 * SPOT_PRICE
        else:
            return 100 * (0.10 * r["strike"] + r["premium"])
    df["cost"]  = df.apply(compute_cost, axis=1)
    df["score"] = df["premium"] - RISK_WEIGHT*df["risk"] + DIST_WEIGHT*df["distance"]

    n = len(df)
    model = LpProblem("Option_Alloc", LpMaximize)
    x = [LpVariable(f"x_{i}", lowBound=0, cat=LpInteger) for i in range(n)]
    y = [LpVariable(f"y_{i}", cat=LpBinary)  for i in range(n)]

    M = int(BUDGET / df["cost"].min())
    for i in range(n):
        model += x[i] <= M * y[i]
        model += x[i] >= MIN_CONTRACTS_PER_STRIKE * y[i]

    model += lpSum(df.loc[i,"cost"] * x[i] for i in range(n)) <= BUDGET

    call_idxs = [i for i,r in df.iterrows() if r["type"].lower()=="call"]
    put_idxs  = [i for i,r in df.iterrows() if r["type"].lower()=="put"]

    if call_idxs and put_idxs:
        model += lpSum(df.loc[i,"cost"] * x[i] for i in call_idxs) <= CALL_RATIO * BUDGET
        model += lpSum(df.loc[i,"cost"] * x[i] for i in put_idxs)  <= PUT_RATIO  * BUDGET
    elif call_idxs:
        model += lpSum(df.loc[i,"cost"] * x[i] for i in call_idxs) <= BUDGET

    model += lpSum(y[i] for i in range(n)) >= MIN_DISTINCT_STRIKES
    if call_idxs:
        model += lpSum(x[i] for i in call_idxs) >= MIN_CALL_CONTRACTS

    model += lpSum(df.loc[i,"score"] * 100 * x[i] for i in range(n))

    model.solve()
    print(f"\nStatus: {LpStatus[model.status]}\n")
    print("Selected contracts:")
    print("-"*90)
    print(f"{'Type':<6}{'Strike':>8}{'Δ':>8}{'Qty':>6}{'Cost':>12}{'Prem*100':>12}{'Score':>8}")
    print("-"*90)

    tot_cost = tot_prem = 0.0
    for i in range(n):
        qty = int(x[i].value() or 0)
        if qty > 0:
            r    = df.loc[i]
            cost = r["cost"] * qty
            prem = r["premium"] * 100 * qty
            tot_cost += cost
            tot_prem += prem
            print(f"{r['type']:<6}{r['strike']:>8.1f}{r['delta']:>8.2f}"
                  f"{qty:>6}{cost:>12,.0f}{prem:>12,.0f}{r['score']:>8.2f}")
    print("-"*90)
    print(f"{'TOTAL':<20}{tot_cost:>12,.0f} cost{tot_prem:>13,.0f} premium")


# ─────────────────────────────
# 5) STRATEGY + CONFIG
# ─────────────────────────────
class HestonMispricingStrategyConfig(StrategyConfig):
    # You can override these from the runner
    strategy_id: str = "HESTON-BACKTEST-001"
    instrument_symbol: str = "TSLA"         # used for Databento queries
    databento_api_key: str = ""             # env fallback if empty
    data_date: str = "2025-04-04"           # YYYY-MM-DD
    risk_free: float = 0.0435
    max_days: int = 60
    write_csvs: bool = True
    output_greeks: str = ""                 # if empty → auto name
    output_cal: str = ""                    # if empty → auto name

class HestonMispricingStrategy(Strategy):
    def __init__(self, config: HestonMispricingStrategyConfig) -> None:
        super().__init__(config)
        self.results: Dict[str, Any] = {}

    def on_start(self) -> None:
        cfg = self.config
        self.log.info(f"START Heston pipeline for {cfg.instrument_symbol} @ {cfg.data_date}")

        api_key = cfg.databento_api_key or os.getenv("DATABENTO_API_KEY", "")
        if not api_key:
            raise RuntimeError("DATABENTO_API_KEY is required for this strategy.")

        # Filenames
        out_greeks = cfg.output_greeks or f"{cfg.instrument_symbol.lower()}_otm_with_bbo_greeks_{cfg.data_date}.csv"
        out_cal    = cfg.output_cal or f"heston_calibration_from_last_trade_{cfg.data_date}.csv"

        # 1) Greeks & IV
        df_greeks, spot = fetch_and_compute_greeks(
            api_key=api_key,
            date=cfg.data_date,
            risk_free=cfg.risk_free,
            underlying_symbol=cfg.instrument_symbol,
            output_csv=out_greeks if cfg.write_csvs else None,
        )
        self.log.info(f"Spot={spot:.4f}, rows={len(df_greeks)}")

        # 2) Heston calibration
        cal_date = pd.Timestamp(cfg.data_date)
        df_cal, params = run_calibration(
            df=df_greeks,
            spot_price=spot,
            risk_free=cfg.risk_free,
            cal_date=cal_date,
            max_days=cfg.max_days,
            output_csv=out_cal if cfg.write_csvs else None,
        )
        self.log.info(f"Heston params={params}")

        # 3) Mispricing & filters
        top_contracts = compute_and_filter_mispricing(
            df_cal=df_cal,
            spot_price=spot,
            risk_free=cfg.risk_free,
            reference_dt=cal_date,
            MAX_DAYS=cfg.max_days,
        )

        # 4) Portfolio optimization
        optimize_portfolio(top_contracts)

        # Store in-memory in case you want to inspect after run
        self.results = {
            "spot": spot,
            "heston_params": params,
            "top_contracts": top_contracts,
        }
        self.log.info("Pipeline complete.")

    def on_stop(self) -> None:
        self.log.info("STOP Heston strategy")


def build(config_dict: dict) -> HestonMispricingStrategy:
    cfg = HestonMispricingStrategyConfig(**config_dict)
    return HestonMispricingStrategy(cfg)
