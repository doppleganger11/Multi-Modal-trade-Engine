# portfolio_engine.py
# Build today's target portfolio using last close, sentiment "alpha",
# and weekly liquidity guardrails (ADV/ADVVAL).

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
PRICES = DATA / "prices_daily.csv"
ADV = DATA / "weekly_adv.csv"
SCORES = DATA / "scores_today.csv"
OUT = DATA / "portfolio_today.csv"

# ---------- helpers ----------
def no_trade_today() -> bool:
    """
    Return True if we should skip trading today:
      - prices_daily.csv missing/empty
      - latest price date < today (IST)
    """
    try:
        p = pd.read_csv(PRICES)
    except FileNotFoundError:
        return True

    if p.empty or "date" not in p.columns:
        return True

    # Parse as UTC-aware directly; DO NOT tz_localize afterwards.
    s = pd.to_datetime(p["date"], utc=True, errors="coerce")
    if s.isna().all():
        return True

    last_ist = s.max().tz_convert("Asia/Kolkata").date()
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").date()
    return last_ist < today_ist


def load_latest_adv(adv_csv: Path) -> pd.DataFrame:
    """
    Return latest weekly ADV shares & ADVVAL per firm.

    Works even if there is NO 'week' column. We simply take the last row
    per firm in file order. If 'week'/'date'/'week_end' exists, we pick max.
    """
    if not adv_csv.exists():
        return pd.DataFrame(columns=["firm_name", "adv_share", "advval"])

    w = pd.read_csv(adv_csv)

    if "firm_name" not in w.columns:
        raise ValueError(f"{adv_csv} missing 'firm_name' column")

    # normalise adv/advval names
    if "adv_share" not in w.columns:
        for alt in ("adv", "adv_shares"):
            if alt in w.columns:
                w = w.rename(columns={alt: "adv_share"})
                break
    if "advval" not in w.columns:
        # compute if price is present
        if "close_last" in w.columns and "adv_share" in w.columns:
            w["advval"] = pd.to_numeric(w["close_last"], errors="coerce") * pd.to_numeric(
                w["adv_share"], errors="coerce"
            )
        else:
            w["advval"] = 0.0

    # try to use a date column if available
    date_candidates = [c for c in ("week", "date", "week_end", "week_start") if c in w.columns]
    if date_candidates:
        dc = date_candidates[0]
        w[dc] = pd.to_datetime(w[dc], utc=True, errors="coerce").dt.date
        idx = w.groupby("firm_name")[dc].idxmax()
        out = w.loc[idx, ["firm_name", "adv_share", "advval"]].reset_index(drop=True)
    else:
        # no date col → last occurrence per firm
        out = (w.groupby("firm_name", as_index=False)
                 .tail(1)[["firm_name", "adv_share", "advval"]]
                 .reset_index(drop=True))

    out["adv_share"] = pd.to_numeric(out.get("adv_share", 0), errors="coerce").fillna(0).astype(int)
    out["advval"] = pd.to_numeric(out.get("advval", 0.0), errors="coerce").fillna(0.0)
    return out

def load_last_closes(prices_csv: Path) -> pd.DataFrame:
    if not prices_csv.exists():
        raise FileNotFoundError(f"{prices_csv} not found")
    df = pd.read_csv(prices_csv, usecols=["date", "firm_name", "close"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    idx = df.groupby("firm_name")["date"].idxmax()
    out = df.loc[idx, ["firm_name", "close"]].rename(columns={"close": "close_last"}).reset_index(drop=True)
    return out

def load_sentiment(scores_csv: Path) -> pd.DataFrame:
    if not scores_csv.exists():
        return pd.DataFrame(columns=["firm_name", "alpha"])
    s = pd.read_csv(scores_csv, usecols=["firm_name", "score"]).rename(columns={"score": "alpha"})
    s["alpha"] = pd.to_numeric(s["alpha"], errors="coerce").fillna(0.0)
    return s

# ---------- main ----------
def build_portfolio(capital: float = 10_000.0) -> None:
    # Seatbelt: if obviously holiday/weekend, write empty suggestion
    if no_trade_today():
        df = pd.DataFrame(columns=["firm_name","close_last","alpha","sigma","w_norm",
                                   "target_we","max_share","cap_share","adv_share","advval","note"])
        df.to_csv(OUT, index=False, encoding="utf-8")
        print(f"[OK] Holiday/weekend detected — wrote empty portfolio to {OUT}")
        return

    # Inputs
    last = load_last_closes(PRICES)             # close_last
    sent = load_sentiment(SCORES)               # alpha
    adv  = load_latest_adv(ADV)                 # adv_share, advval

    # Merge universe (inner join on firms we have all pieces for)
    df = last.merge(sent, on="firm_name", how="left").merge(adv, on="firm_name", how="left")
    df["alpha"] = df["alpha"].fillna(0.0)

    # simple risk proxy
    df["sigma"] = 0.05  # placeholder until you compute rolling vol

    # mean-variance toy weight (maximize alpha/variance with unit sum)
    # w_i ∝ alpha_i / sigma_i^2 ; then normalize to 1
    raw = df["alpha"] / (df["sigma"] ** 2)
    raw = raw.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if raw.abs().sum() == 0:
        w_norm = np.zeros(len(df))
    else:
        w_norm = (raw / raw.abs().sum()).to_numpy()
    df["w_norm"] = w_norm

    # desired notional per name
    df["target_we"] = capital * df["w_norm"]

    # Liquidity caps -------------------------------------------------------
    # Participation rules:
    # - daily participation cap = 20% of 1/5th of weekly ADV (i.e., 4% of ADV shares)
    # - also cap notional by 5% of ADVVAL per week -> daily approx 1% of ADVVAL
    adv_shares = pd.to_numeric(df["adv_share"], errors="coerce").fillna(0).to_numpy()
    advval = pd.to_numeric(df["advval"], errors="coerce").fillna(0.0).to_numpy()
    price = pd.to_numeric(df["close_last"], errors="coerce").fillna(0.0).to_numpy()

    # shares you’re allowed to trade today
    daily_cap_shares = 0.2 * (adv_shares / 5.0)     # 20% of daily ADV proxy
    # fix: use numpy clip (no keyword 'lower')
    daily_cap_shares = np.clip(np.floor(np.nan_to_num(daily_cap_shares, nan=0.0)), 0, None).astype(int)

    # notional cap (1% of weekly ADVVAL)
    daily_cap_notional = 0.01 * advval
    # convert notional cap to shares via price
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_by_notional = np.where(price > 0, daily_cap_notional / price, 0.0)
    cap_by_notional = np.clip(np.floor(np.nan_to_num(cap_by_notional, nan=0.0)), 0, None).astype(int)

    # max shares permitted today is the minimum of the two caps
    max_shares_today = np.minimum(daily_cap_shares, cap_by_notional)

    # max shares needed to reach target notional (one-shot)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_shares_needed = np.where(price > 0, df["target_we"].to_numpy() / price, 0.0)
    target_shares_needed = np.clip(np.floor(np.nan_to_num(target_shares_needed, nan=0.0)), 0, None).astype(int)

    df["max_share"] = target_shares_needed
    df["cap_share"] = max_shares_today

    # notes / flags
    notes = np.full(len(df), "ok", dtype=object)
    notes[(adv_shares <= 0) | (advval <= 0)] = "no_liquidity_info"
    notes[max_shares_today == 0] = "cap_binds"
    df["note"] = notes

    # sort for readability
    df = df.sort_values(["w_norm"], ascending=False).reset_index(drop=True)

    # output
    cols = ["firm_name","close_last","alpha","sigma","w_norm","target_we",
            "max_share","cap_share","adv_share","advval","note"]
    df[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] wrote portfolio to {OUT}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=10_000.0)
    args = parser.parse_args()
    build_portfolio(capital=args.capital)
