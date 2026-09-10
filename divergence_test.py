"""
Divergence test on Sofascore xG.

Hypothesis: a team whose xG form outruns its points form has been
under-rewarded, the market anchors on results, and the gap predicts.

Model:  awaywin ~ divergence,  offset = logit(market p_away)
        divergence = away_div - home_div
        div  = xg_form - 0.5 * pts_form
        xg_form  = mean(xG for - xG against) over last N
        pts_form = mean(points) over last N, 0-3 scale

Form windows reset each season (squad turnover).

CHECK THE SCALING against your FootyStats run before comparing coefficients.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

WINDOWS = [5, 10]
df = pd.read_csv("nln_merged.csv").sort_values("kickoff").reset_index(drop=True)

# ---- long format: one row per team-match -------------------------------
def side(d, us, them):
    o = pd.DataFrame({
        "event_id": d["event_id"], "season": d["season"], "kickoff": d["kickoff"],
        "team": d[us], "xg_for": d[f"{us}_xg"], "xg_ag": d[f"{them}_xg"],
        "gf": d[f"{us}_goals"], "ga": d[f"{them}_goals"],
    })
    o["pts"] = np.where(o.gf > o.ga, 3, np.where(o.gf == o.ga, 1, 0))
    o["xgd"] = o.xg_for - o.xg_ag
    return o

long = pd.concat([side(df, "home", "away"), side(df, "away", "home")],
                 ignore_index=True).sort_values("kickoff")

# ---- rolling form, shifted so the current match is excluded -------------
g = long.groupby(["season", "team"])
for n in WINDOWS:
    long[f"xgf{n}"] = g["xgd"].transform(lambda s: s.shift().rolling(n).mean())
    long[f"ptf{n}"] = g["pts"].transform(lambda s: s.shift().rolling(n).mean())
    long[f"div{n}"] = long[f"xgf{n}"] - 0.0 * long[f"ptf{n}"]

# ---- back to wide, build away-minus-home divergence --------------------
keep = ["event_id", "team"] + [f"div{n}" for n in WINDOWS]
m = df.merge(long[keep], left_on=["event_id", "home"], right_on=["event_id", "team"]) \
      .merge(long[keep], left_on=["event_id", "away"], right_on=["event_id", "team"],
             suffixes=("_h", "_a"))

m["awaywin"] = (m.away_goals > m.home_goals).astype(int)
m["offset"] = np.log(m.p_close_away / (1 - m.p_close_away))

for n in WINDOWS:
    m[f"D{n}"] = m[f"div{n}_a"] - m[f"div{n}_h"]

# ---- fit ---------------------------------------------------------------
def run(data, n, label):
    d = data.dropna(subset=[f"D{n}", "offset"])
    if len(d) < 60:
        print(f"{label:<22} n={len(d):<5} too few")
        return
    X = sm.add_constant(d[[f"D{n}"]])
    fit = sm.Logit(d["awaywin"], X, offset=d["offset"]).fit(disp=0)
    b = fit.params[f"D{n}"]
    se = fit.bse[f"D{n}"]
    print(f"{label:<22} n={len(d):<5} beta={b:+.4f}  se={se:.4f}  "
          f"z={b/se:+.2f}  95% CI [{b-1.96*se:+.3f}, {b+1.96*se:+.3f}]")

for n in WINDOWS:
    print(f"\n--- window {n} ---")
    run(m, n, "all seasons")
    run(m[m.season == "24/25"], n, "24/25 (in-sample era)")
    run(m[m.season != "24/25"], n, "25/26+ (out of sample)")
