# order_build.py
# Build executable integer-share orders from portfolio_today.csv for a small capital.
# - No dependency on a 'ticker' column (adds it if firms.csv has it)
# - Respects cap_share and a per-name max fraction of capital
# - Falls back to greedy 1-share rounds if all weighted targets round to 0

import argparse, math
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
PORT = HERE / "data" / "portfolio_today.csv"
FIRMS = HERE.parent / "firms.csv"          # optional (to add ticker)
OUT  = HERE / "data" / "orders_today.csv"

def _merge_ticker_if_available(df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" in df.columns:
        return df
    if FIRMS.exists():
        try:
            f = pd.read_csv(FIRMS)
            # accept yf_ticker or ticker
            if "ticker" not in f.columns and "yf_ticker" in f.columns:
                f = f.rename(columns={"yf_ticker": "ticker"})
            if {"firm_name","ticker"}.issubset(f.columns):
                f = f[["firm_name","ticker"]].copy()
                return df.merge(f, on="firm_name", how="left")
        except Exception:
            pass
    df["ticker"] = ""  # keep column for convenience
    return df

def build_orders(capital: float, max_name_frac: float = 0.4):
    df = pd.read_csv(PORT)
    df = _merge_ticker_if_available(df)

    # normalize/safety
    for c in ("close_last","w_norm","target_we","cap_share"):
        if c not in df.columns:
            df[c] = 0
    df["close_last"] = pd.to_numeric(df["close_last"], errors="coerce").fillna(0.0)
    df["w_norm"]     = pd.to_numeric(df["w_norm"], errors="coerce").fillna(0.0)
    df["target_we"]  = pd.to_numeric(df["target_we"], errors="coerce").fillna(0.0)
    df["cap_share"]  = pd.to_numeric(df["cap_share"], errors="coerce").fillna(0).astype(int)
    note = df.get("note", "ok").astype(str).str.lower()
    trad = df[(note == "ok") & (df["cap_share"] > 0) & (df["close_last"] > 0)].copy()

    if trad.empty:
        print("[INFO] no tradable names today (after caps/notes/prices).")
        out = df.copy()
        out["order_shares"] = 0
        out["order_value"] = 0.0
        out.to_csv(OUT, index=False)
        return

    # target shares from weights
    trad["float_sh"]  = trad["target_we"] / trad["close_last"]
    trad["floor_sh"]  = trad["float_sh"].apply(math.floor).clip(lower=0).astype(int)
    trad["target_shares"] = trad[["floor_sh","cap_share"]].min(axis=1)

    # If everyone is 0 → greedy 1-share rounds by weight
    if trad["target_shares"].sum() == 0:
        trad = trad.sort_values("w_norm", ascending=False).reset_index(drop=True)
        remaining = float(capital)
        per_name_cap = max_name_frac * capital
        trad["order_shares"] = 0

        # Round 1: one share each in weight order
        for i, r in trad.iterrows():
            px = float(r["close_last"])
            if px <= 0 or r["cap_share"] < 1:
                continue
            if remaining >= px and px <= per_name_cap:
                trad.loc[i, "order_shares"] = 1
                remaining -= px

        # Additional rounds while cash remains
        progressed = True
        while progressed and remaining > 0:
            progressed = False
            for i, r in trad.iterrows():
                px = float(r["close_last"])
                cur = int(trad.loc[i, "order_shares"])
                if px <= 0 or cur >= int(r["cap_share"]):
                    continue
                if (cur + 1) * px <= per_name_cap and remaining >= px:
                    trad.loc[i, "order_shares"] = cur + 1
                    remaining -= px
                    progressed = True
    else:
        # Start from floors, then distribute leftover rupees to largest fractions
        trad["order_shares"] = trad["target_shares"].astype(int)
        spent = (trad["order_shares"] * trad["close_last"]).sum()
        remaining = max(0.0, capital - float(spent))
        per_name_cap = max_name_frac * capital

        trad["frac"] = trad["float_sh"] - trad["floor_sh"]
        trad = trad.sort_values("frac", ascending=False).reset_index(drop=True)
        for i, r in trad.iterrows():
            px = float(r["close_last"])
            cur = int(trad.loc[i, "order_shares"])
            if px <= 0 or remaining < px:
                continue
            if cur >= int(r["cap_share"]):
                continue
            if (cur + 1) * px <= per_name_cap:
                trad.loc[i, "order_shares"] = cur + 1
                remaining -= px

    trad["order_value"] = trad["order_shares"] * trad["close_last"]

    cols = ["firm_name","ticker","close_last","cap_share","w_norm","order_shares","order_value"]
    orders = trad[cols].copy()

    # merge back onto the full portfolio for visibility
    out = df.merge(orders, on=["firm_name","ticker","close_last","cap_share","w_norm"], how="left")
    out["order_shares"] = out["order_shares"].fillna(0).astype(int)
    out["order_value"]  = out["order_value"].fillna(0.0).round(2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] wrote {OUT} | spend ≈ ₹{out['order_value'].sum():,.2f} | cash left ≈ ₹{capital - out['order_value'].sum():,.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--max_name_frac", type=float, default=0.4,
                    help="max fraction of capital per single name (default 0.4)")
    args = ap.parse_args()
    build_orders(args.capital, args.max_name_frac)
