# signals_pipeline.py
# Build per-firm features from prices + sentiment + weekly ADV.
# Inputs:
#   data/prices_daily.csv     (date, firm_name, ticker, close, volume)
#   data/scores_today.csv     (firm_name, score)
#   data/weekly_adv.csv       (week_end, firm_name, adv_shares, advval)
# Output:
#   data/factor_snapshot.csv  (latest snapshot of features per firm)

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
PRICES = DATA / "prices_daily.csv"
SCORES = DATA / "scores_today.csv"
WADV   = DATA / "weekly_adv.csv"
OUT    = DATA / "factor_snapshot.csv"

def _read_prices(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    # normalize dtypes
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    df["firm_name"] = df["firm_name"].astype(str)
    if "ticker" not in df.columns:
        df["ticker"] = ""
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date","firm_name","close"])
    return df

def _read_scores(p: Path) -> pd.DataFrame:
    if not p.exists():
        # no sentiment today -> zeros
        return pd.DataFrame({"firm_name": [], "score": []})
    s = pd.read_csv(p)
    s["firm_name"] = s["firm_name"].astype(str)
    # standardize to 'score' column if different name was used
    if "score" not in s.columns:
        if "headline_score" in s.columns:
            s = s.rename(columns={"headline_score":"score"})
        else:
            s["score"] = 0.0
    s["score"] = pd.to_numeric(s["score"], errors="coerce").fillna(0.0)
    # keep last value per firm if duplicates
    s = s.groupby("firm_name", as_index=False)["score"].last()
    return s

def _read_weekly_adv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame({"firm_name": [], "adv_shares": [], "advval": []})

    w = pd.read_csv(p)

    # Normalize column names from variants
    cols_lower = {c.lower(): c for c in w.columns}
    if "adv_share" in cols_lower and "adv_shares" not in w.columns:
        w = w.rename(columns={cols_lower["adv_share"]: "adv_shares"})
    if "advshares" in cols_lower and "adv_shares" not in w.columns:
        w = w.rename(columns={cols_lower["advshares"]: "adv_shares"})
    if "adv_val" in cols_lower and "advval" not in w.columns:
        w = w.rename(columns={cols_lower["adv_val"]: "advval"})
    if "advvalue" in cols_lower and "advval" not in w.columns:
        w = w.rename(columns={cols_lower["advvalue"]: "advval"})
    if "week_end" in cols_lower:
        w = w.rename(columns={cols_lower["week_end"]: "week_end"})

    # Types
    if "week_end" in w.columns:
        w["week_end"] = pd.to_datetime(w["week_end"], utc=True, errors="coerce").dt.date
    w["firm_name"] = w["firm_name"].astype(str)
    if "adv_shares" not in w.columns:
        w["adv_shares"] = np.nan
    if "advval" not in w.columns:
        w["advval"] = np.nan

    w["adv_shares"] = pd.to_numeric(w["adv_shares"], errors="coerce")
    w["advval"] = pd.to_numeric(w["advval"], errors="coerce")

    # Latest week per firm
    if "week_end" in w.columns:
        idx = w.groupby("firm_name")["week_end"].idxmax()
        w_last = w.loc[idx, ["firm_name","adv_shares","advval"]].copy()
    else:
        w_last = w.groupby("firm_name", as_index=False)[["adv_shares","advval"]].last()

    return w_last


def _features_for_firm(g: pd.DataFrame) -> pd.Series:
    # g is one firm's history sorted by date
    g = g.sort_values("date").copy()
    g["ret1"] = g["close"].pct_change()
    # momentum windows
    def total_return(n):
        if len(g) < n+1: return np.nan
        c0 = g["close"].iloc[-(n+1)]
        c1 = g["close"].iloc[-1]
        return (c1 / c0) - 1.0 if c0 and pd.notna(c0) else np.nan

    ret20  = total_return(20)
    ret60  = total_return(60)
    ret120 = total_return(120)

    # 20D volatility (daily)
    vol20 = g["ret1"].rolling(20).std().iloc[-1] if len(g) >= 20 else np.nan

    # Sharpe-ish
    sharpe20 = (ret20 / (vol20 * np.sqrt(20))) if (pd.notna(ret20) and pd.notna(vol20) and vol20 > 0) else np.nan

    # Mean reversion (z-score of last 5D returns)
    if len(g) >= 25:
        r = g["ret1"].tail(25).iloc[:-1]  # prior 24 days
        mu, sd = r.mean(), r.std()
        last5 = g["ret1"].tail(5).sum()
        z5 = (last5 - mu*5) / (sd*np.sqrt(5)) if sd and sd > 1e-12 else np.nan
    else:
        z5 = np.nan

    # closing price & date
    close_last = g["close"].iloc[-1]
    date_last  = g["date"].iloc[-1]

    return pd.Series({
        "date": date_last,
        "close_last": close_last,
        "ret20": ret20,
        "ret60": ret60,
        "ret120": ret120,
        "vol20": vol20,
        "sharpe20": sharpe20,
        "zscore5": z5
    })

def build_snapshot():
    prices = _read_prices(PRICES)
    # compute features per firm
    feats = prices.groupby(["firm_name","ticker"]).apply(_features_for_firm).reset_index()
    # bring in sentiment
    scores = _read_scores(SCORES)
    feats = feats.merge(scores, on="firm_name", how="left")
    feats["score"] = pd.to_numeric(feats["score"], errors="coerce").fillna(0.0)
    # bring in liquidity (weekly ADV)
    wlast = _read_weekly_adv(WADV)
    feats = feats.merge(wlast, on="firm_name", how="left")

    # ---- ADV sanity: backfill advval if missing but adv_shares + price exist ----
    feats["adv_shares"] = pd.to_numeric(feats.get("adv_shares"), errors="coerce")
    feats["advval"] = pd.to_numeric(feats.get("advval"), errors="coerce")
    feats["close_last"] = pd.to_numeric(feats.get("close_last"), errors="coerce")

    need_val = feats["advval"].isna() & feats["adv_shares"].notna() & feats["close_last"].notna()
    feats.loc[need_val, "advval"] = feats.loc[need_val, "adv_shares"] * feats.loc[need_val, "close_last"]

    # final clean (no negatives / NaNs)
    feats.loc[~np.isfinite(feats["adv_shares"]) | (feats["adv_shares"] < 0), "adv_shares"] = 0
    feats.loc[~np.isfinite(feats["advval"]) | (feats["advval"] < 0), "advval"] = 0


    # clean up numeric types and caps
    for c in ["ret20","ret60","ret120","vol20","sharpe20","zscore5","adv_shares","advval"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")

    # keep one row per firm
    feats = feats.drop_duplicates(subset=["firm_name"], keep="last")

    # write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] wrote factor snapshot for {len(feats)} firms -> {OUT}")

if __name__ == "__main__":
    build_snapshot()
