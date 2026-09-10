"""
Pull Sofascore's own 1X2 odds and attach them to nln_xg.csv by event_id.

Gives opening (initialFractionalValue) and closing (fractionalValue) prices
for every season with coverage — no name matching, no date matching.

Run after `sofascore_nln.py backfill`.  Output: nln_merged.csv
"""

from fractions import Fraction

import numpy as np
import pandas as pd

from sofascore_nln import get, seasons, finished_events


def to_decimal(frac):
    """'23/10' -> 3.30.  Sofascore quotes net fractional odds."""
    if not frac:
        return np.nan
    try:
        return float(Fraction(frac)) + 1.0
    except (ValueError, ZeroDivisionError):
        return np.nan


def match_odds(event_id):
    d = get(f"event/{event_id}/odds/1/all")
    if not d:
        return {}
    for mk in d.get("markets", []):
        if mk.get("marketGroup") != "1X2" or mk.get("marketPeriod") != "Full-time":
            continue
        out = {}
        for ch in mk.get("choices", []):
            key = {"1": "home", "X": "draw", "2": "away"}.get(ch.get("name"))
            if not key:
                continue
            out[f"open_{key}"] = to_decimal(ch.get("initialFractionalValue"))
            out[f"close_{key}"] = to_decimal(ch.get("fractionalValue"))
        return out
    return {}


rows = []
for sid, year in seasons()[:3]:
    evs = finished_events(sid)
    print(f"{year}: {len(evs)} matches")
    for i, e in enumerate(evs, 1):
        o = match_odds(e["id"])
        if o:
            rows.append({"event_id": e["id"], **o})
        if i % 50 == 0:
            print(f"  {i}/{len(evs)}")

odds = pd.DataFrame(rows)
print(f"\n{len(odds)} matches with 1X2 odds")

xg = pd.read_csv("nln_xg.csv")
m = xg.merge(odds, on="event_id", how="left")

# De-vig both books proportionally.
for stage in ("open", "close"):
    cols = [f"{stage}_{k}" for k in ("home", "draw", "away")]
    imp = 1 / m[cols]
    tot = imp.sum(axis=1)
    m[f"{stage}_overround"] = tot - 1
    for k in ("home", "draw", "away"):
        m[f"p_{stage}_{k}"] = imp[f"{stage}_{k}"] / tot

final = m.dropna(subset=["p_close_home", "home_xg"])
final.to_csv("nln_merged.csv", index=False)

print(f"wrote nln_merged.csv — {len(final)} matches with xG and closing odds")
print(f"\nmean overround  open: {m['open_overround'].mean():.2%}"
      f"   close: {m['close_overround'].mean():.2%}")
print(final.groupby("season").size().to_string())

# Sanity: does the closing price actually predict the result?
res = np.where(final["home_goals"] > final["away_goals"], "home",
               np.where(final["home_goals"] < final["away_goals"], "away", "draw"))
for k in ("home", "draw", "away"):
    hit = (res == k).mean()
    print(f"  {k}: implied {final[f'p_close_{k}'].mean():.3f}  actual {hit:.3f}")
