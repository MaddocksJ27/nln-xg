"""
Would backing the better opponent-adjusted team have been profitable?

Sweeps rating-gap thresholds 0.5 / 0.75 / 1.0 / 1.25 / 1.5.

Ratings are strictly pre-match (walk-forward fit on prior matches only), so
there is no lookahead. Settled at closing price, flat stakes.

READ THIS BEFORE READING THE NUMBERS
------------------------------------
This is a 5-threshold x 3-side x 3-season sweep. Under a pure null with the
vig applied, several cells WILL print positive — that is arithmetic, not
evidence. The output therefore shows:
  - every cell, never a selected subset
  - the standard error next to every ROI
  - 24/25 and 25/26 separately, so replication is visible
A cell is only interesting if it is positive in BOTH seasons by more than
its own standard error. Nothing in this project has yet cleared that bar.

    python rating_strategy.py
"""

import numpy as np
import pandas as pd

THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5]

od = pd.read_csv("nln_merged.csv")
od["date"] = pd.to_datetime(od.kickoff).dt.normalize()

try:
    r = pd.read_csv("nln_ratings_history.csv")
except FileNotFoundError:
    raise SystemExit("run `python ratings.py --history` first")
r["date"] = pd.to_datetime(r.date).dt.normalize()
r = r[["season", "date", "team", "rating"]]

m = (od.merge(r, left_on=["season", "date", "home"],
              right_on=["season", "date", "team"], how="inner")
       .rename(columns={"rating": "r_home"}).drop(columns="team")
       .merge(r, left_on=["season", "date", "away"],
              right_on=["season", "date", "team"], how="inner")
       .rename(columns={"rating": "r_away"}).drop(columns="team"))

print(f"{len(m)} matches with pre-match ratings for both teams")
print(f"by season:\n{m.groupby('season').size().to_string()}\n")

m["gap"] = m.r_home - m.r_away          # positive = home rated higher
m["home_won"] = (m.home_goals > m.away_goals).astype(int)
m["away_won"] = (m.away_goals > m.home_goals).astype(int)


def leg(sel, side):
    if side == "home":
        ret = np.where(sel.home_won == 1, sel.close_home - 1, -1.0)
        price = sel.close_home
    else:
        ret = np.where(sel.away_won == 1, sel.close_away - 1, -1.0)
        price = sel.close_away
    return ret, price


def cell(d, thr, mode):
    """mode: 'home' back home when gap>=thr, 'away' back away when gap<=-thr,
       'both' back whichever side is better rated by >= thr."""
    if mode == "home":
        sel = d[d.gap >= thr]
        ret, price = leg(sel, "home")
    elif mode == "away":
        sel = d[d.gap <= -thr]
        ret, price = leg(sel, "away")
    else:
        hs, aw = d[d.gap >= thr], d[d.gap <= -thr]
        rh, ph = leg(hs, "home")
        ra, pa = leg(aw, "away")
        ret = np.concatenate([rh, ra])
        price = pd.concat([ph, pa])
        sel = pd.concat([hs, aw])

    n = len(ret)
    if n == 0:
        return n, np.nan, np.nan, np.nan
    return n, ret.mean(), ret.std() / np.sqrt(n), price.mean()


for mode in ("both", "home", "away"):
    print("=" * 86)
    print(f"BACK THE BETTER-RATED SIDE — {mode}")
    print("=" * 86)
    print(f"{'thr':>5} | {'all seasons':^26} | {'24/25':^26} | {'25/26':^26}")
    print(f"{'':>5} | {'n    ROI      se':^26} | {'n    ROI      se':^26} "
          f"| {'n    ROI      se':^26}")
    print("-" * 86)
    for thr in THRESHOLDS:
        parts = []
        for d in (m, m[m.season == "24/25"], m[m.season == "25/26"]):
            n, roi, se, avg = cell(d, thr, mode)
            parts.append("  n/a" if n == 0 else
                         f"{n:>4} {roi:+7.2%} {se:>7.2%}")
        print(f"{thr:>5} | {parts[0]:^26} | {parts[1]:^26} | {parts[2]:^26}")
    print()

print("Interpretation: with ~11% overround on 1X2, a cell needs to beat 0%")
print("by clearly more than one standard error, in BOTH seasons, to mean")
print("anything. One good season out of two is the pattern that has already")
print("appeared twice in this project and reversed both times.")
