# prices_pipeline.py  — Incremental updater (append + same-day overwrite)
from __future__ import annotations
import os
import sys
import time
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import requests

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_CSV = DATA_DIR / "prices_daily.csv"
FIRMS_DEFAULT = (HERE.parent / "firms.csv").resolve()
IST = "Asia/Kolkata"


def atomic_write_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, out_path)


def load_firms(firms_csv: Path) -> pd.DataFrame:
    """
    Accepts firms.csv with columns:
      - firm_name
      - yf_ticker  (preferred)  OR 'ticker' (alias)
    """
    df = pd.read_csv(firms_csv)
    if "yf_ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "yf_ticker"})
    need = {"firm_name", "yf_ticker"}
    if not need.issubset(df.columns):
        raise ValueError(f"firms.csv must contain columns {need}, got {df.columns.tolist()}")
    df = df[["firm_name", "yf_ticker"]].copy()
    df["firm_name"] = df["firm_name"].astype(str)
    df["yf_ticker"] = df["yf_ticker"].astype(str)
    return df


def read_existing(out_csv: Path) -> pd.DataFrame:
    """
    Reads your stored schema:
      columns = [date, close, volume, firm_name, yf_ticker]
    date is tz-aware string like 'YYYY-mm-dd 00:00:00+05:30'.
    Returns a clean DataFrame with same columns; 'date' kept tz-aware (IST) at midnight.
    """
    cols = ["date", "close", "volume", "firm_name", "yf_ticker"]
    if not out_csv.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(out_csv)

    # Robust parse → IST midnight
    d = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    d = d.dt.tz_convert(IST)
    # normalize to midnight IST (so strings match your format)
    d = d.dt.floor("D")

    df = df.loc[d.notna()].copy()
    df["date"] = d

    # Types
    df["firm_name"] = df["firm_name"].astype(str)
    df["yf_ticker"] = df["yf_ticker"].astype(str)
    df["close"] = pd.to_numeric(df.get("close", 0), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

    # Keep schema + order
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in ("firm_name", "yf_ticker") else (0 if c in ("close", "volume") else pd.NaT)
    df = df[cols]

    # De-dup and sort
    df = df.drop_duplicates(subset=["firm_name", "date"], keep="last")
    df = df.sort_values(["firm_name", "date"]).reset_index(drop=True)
    return df


def last_ist_date_for(df: pd.DataFrame, firm: str) -> pd.Timestamp | None:
    sub = df.loc[df["firm_name"] == firm, "date"]
    if sub.empty:
        return None
    # date column is tz-aware IST; keep it that way
    return sub.max()


def to_ist_midnight(idx: pd.Index) -> pd.Series:
    # yfinance sometimes sets tz; sometimes not. Normalize to IST midnight.
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(IST).dt.floor("D")
    return dt


def fetch_tail(ticker: str, days: int = 7) -> pd.DataFrame:
    """
    Pull last N days from Yahoo for a ticker.
    Returns columns: date(IST tz-aware), close, volume
    """
    # IMPORTANT: do NOT pass a requests.Session() to yfinance on new versions
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{days}d", interval="1d", auto_adjust=False, actions=False)

    if df is None or df.empty:
        # secondary attempt with yf.download (also with no session)
        df = yf.download(
            ticker, period=f"{days}d", interval="1d",
            auto_adjust=False, progress=False, threads=False
        )

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = (
        df.reset_index()[["Date", "Close", "Volume"]]
          .rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    )
    out["date"] = to_ist_midnight(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["date"])
    return out[["date", "close", "volume"]]



def try_fetch_with_fallback(nse: str, bse: str | None = None, days: int = 7) -> pd.DataFrame:
    # 1) NSE ticker
    df = fetch_tail(nse, days=days)
    if not df.empty:
        return df
    # 2) BSE fallback if provided
    if bse:
        df = fetch_tail(bse, days=days)
        if not df.empty:
            return df
    return df  # empty


def merge_update(existing: pd.DataFrame, newrows: pd.DataFrame, firm: str, yf_ticker: str) -> pd.DataFrame:
    """
    existing: current stored table
    newrows: cols [date, close, volume] for one firm
    – Adds/updates rows for that firm; same-day values overwrite.
    """
    if newrows.empty:
        return existing

    add = newrows.copy()
    add["firm_name"] = firm
    add["yf_ticker"] = yf_ticker

    # concat + last wins per key
    comb = pd.concat([existing, add], ignore_index=True)
    comb = comb.sort_values(["firm_name", "date"])  # ensure deterministic
    comb = comb.drop_duplicates(subset=["firm_name", "date"], keep="last")
    comb = comb.sort_values(["firm_name", "date"]).reset_index(drop=True)
    return comb


def run(firms_csv: Path, out_csv: Path = OUT_CSV, tail_days: int = 7) -> None:
    print(f"[BOOT] Using firms.csv at: {firms_csv}")
    firms = load_firms(firms_csv)

    # existing table (your schema)
    existing = read_existing(out_csv)

    # Map of NSE->BSE fallback if your tickers follow *.NS/*.BO
    def bse_fallback(t: str) -> str | None:
        return (t[:-3] + ".BO") if t.endswith(".NS") else None

    for _, row in firms.iterrows():
        firm = row["firm_name"]
        tk = row["yf_ticker"].strip()

        # If we already have data, pull only last few days (and overwrite same-day if changed)
        last_dt = last_ist_date_for(existing, firm)
        print(f"[FETCH] {firm} ({tk}) last {tail_days}d → update ...")

        df_new = try_fetch_with_fallback(tk, bse_fallback(tk), days=tail_days)
        if df_new.empty:
            print(f"[INFO] {firm}: no data fetched")
            continue

        # Keep only rows >= (last_dt - a buffer) to minimize volume of merge
        if last_dt is not None:
            cutoff = (pd.Timestamp(last_dt).tz_convert(IST) - pd.Timedelta(days=tail_days + 1))
            df_new = df_new[df_new["date"] >= cutoff]

        if df_new.empty:
            print(f"[INFO] {firm}: nothing new after cutoff")
            continue

        existing = merge_update(existing, df_new, firm, tk)

    if existing.empty:
        print("[WARN] Nothing to write.")
        return

    # Ensure column order and formatting like your file
    existing = existing[["date", "close", "volume", "firm_name", "yf_ticker"]].copy()

    # Write using ISO + timezone string (keeps your +05:30 midnight format)
    # pandas will emit RFC 3339 strings for tz-aware datetimes
    atomic_write_df(existing, out_csv)
    print(f"[OK] wrote {len(existing)} rows to {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--firms", type=str, default=str(FIRMS_DEFAULT))
    parser.add_argument("--out", type=str, default=str(OUT_CSV))
    parser.add_argument("--tail_days", type=int, default=7, help="Days to refetch for overwrite/append.")
    args = parser.parse_args()

    run(Path(args.firms), Path(args.out), tail_days=args.tail_days)
