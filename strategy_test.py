"""
The discrete strategy, run on both xG measurements.

RULE (fixed, as originally specified — nothing tuned here):
    back AWAY when, over the last 10 matches each:
      away xG differential  >  home xG differential
      away points per game  <  home points per game
      away closing price    >  2.0

Form windows reset each season. Flat stakes, settled at closing price.

Run on Sofascore xG and on the recalibrated NLN xG, so the ONLY thing
that changes is the measurement.

CAVEAT, stated up front: this specification emerged from a search over
~11,000 variants and was already identified as roughly a 1-in-10 draw.
Swapping the xG input does not undo that. At ~100-150 bets the standard
error on flat-stakes ROI is 12-15pp, so anything under about +25% is
indistinguishable from zero. The informative outcome here is a clean
negative.

    python strategy_test.py
"""

import numpy as np
import pandas as pd

WINDOW = 10
MIN_ODDS = 2.0

df = pd.read_csv("nln_merged.csv").sort_values("kickoff").reset_index(drop=True)
nln = pd.read_csv("nln_match_xg_nln.csv")

w = nln.pivot(index="event_id", columns="is_home", values="xg_nln")
w.columns = ["away_xg_nln", "home_xg_nln"]
df = df.merge(w.reset_index(), on="event_id", how="inner")
print(f"{len(df)} matches with odds and both xG versions\n")


def forms(d, hcol, acol):
    """Rolling xG differential and points per game, shifted, per team."""
    def side(us, them, ucol, tcol):
        o = pd.DataFrame({
            "event_id": d.event_id, "season": d.season, "kickoff": d.kickoff,
            "team": d[us], "xgd": d[ucol] - d[tcol],
            "gf": d[f"{us}_goals"], "ga": d[f"{them}_goals"]})
        o["pts"] = np.where(o.gf > o.ga, 3, np.where(o.gf == o.ga, 1, 0))
        return o

    long = pd.concat([side("home", "away", hcol, acol),
                      side("away", "home", acol, hcol)]).sort_values("kickoff")
    g = long.groupby(["season", "team"])
    long["xgf"] = g["xgd"].transform(lambda s: s.shift().rolling(WINDOW).mean())
    long["ptf"] = g["pts"].transform(lambda s: s.shift().rolling(WINDOW).mean())

    k = ["event_id", "team", "xgf", "ptf"]
    return (d.merge(long[k], left_on=["event_id", "home"],
                    right_on=["event_id", "team"])
             .merge(long[k], left_on=["event_id", "away"],
                    right_on=["event_id", "team"], suffixes=("_h", "_a")))


def run(m, label):
    m = m.dropna(subset=["xgf_h", "xgf_a", "close_away"]).copy()
    sel = m[(m.xgf_a > m.xgf_h) &
            (m.ptf_a < m.ptf_h) &
            (m.close_away > MIN_ODDS)].copy()

    if len(sel) < 10:
        print(f"  {label:<26} n={len(sel)} — too few to read")
        return

    sel["won"] = (sel.away_goals > sel.home_goals).astype(int)
    ret = np.where(sel.won == 1, sel.close_away - 1, -1.0)

    roi, se = ret.mean(), ret.std() / np.sqrt(len(ret))
    strike = sel.won.mean()
    avg = sel.close_away.mean()
    # what the market says the strike rate should be, vig included
    implied = (1 / sel.close_away).mean()

    print(f"  {label:<26} n={len(sel):<4} "
          f"ROI={roi:+7.2%} (se {se:.2%})  "
          f"strike={strike:.3f} vs implied {implied:.3f}  "
          f"avg price {avg:.2f}  P/L={ret.sum():+.1f}u")


for tag, hc, ac in (("Sofascore xG", "home_xg", "away_xg"),
                    ("Recalibrated xG", "home_xg_nln", "away_xg_nln")):
    print(f"=== {tag} ===")
    m = forms(df, hc, ac)
    run(m, "all seasons")
    run(m[m.season == "24/25"], "24/25")
    run(m[m.season == "25/26"], "25/26")
    run(m[m.season == "26/27"], "26/27 (held out)")
    print()

print("Compare ROI against its own standard error before anything else.")
print("Compare strike rate against implied: implied already contains the vig,")
print("so beating it by less than ~3pp is not an edge.")
