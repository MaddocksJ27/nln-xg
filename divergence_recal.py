"""
Divergence test, re-run on the recalibrated NLN xG.

Identical specification to divergence_test.py. The ONLY change is the xG
input: nln_match_xg_nln.csv (NLN-calibrated) instead of Sofascore's figure.
Anything else held fixed, so any difference is attributable to the
measurement change and nothing else.

  window     : 5 and 10, reset each season
  predictor  : (away_div - home_div), div = xg_form - 0.5 * pts_form
  model      : logit(awaywin) ~ D, offset = logit(market p_away)
  splits     : all / 24-25 / 25-26+ , and Sofascore vs recalibrated side by side

Prior: probably null. Shot-level calibration error largely averages out when
twelve shots are summed into a match total. But it is a clean single question.

    python divergence_recal.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("nln_merged.csv").sort_values("kickoff").reset_index(drop=True)
nln = pd.read_csv("nln_match_xg_nln.csv")

# recalibrated xG back to wide
w = nln.pivot(index="event_id", columns="is_home", values="xg_nln")
w.columns = ["away_xg_nln", "home_xg_nln"]
df = df.merge(w.reset_index(), on="event_id", how="inner")
print(f"{len(df)} matches with odds and recalibrated xG\n")

print("match-level xG, Sofascore vs recalibrated:")
print(f"  home  {df.home_xg.mean():.3f} -> {df.home_xg_nln.mean():.3f}")
print(f"  away  {df.away_xg.mean():.3f} -> {df.away_xg_nln.mean():.3f}")
print(f"  corr(sofa, recal) home = "
      f"{df[['home_xg','home_xg_nln']].corr().iloc[0,1]:.4f}\n")


def build(d, hcol, acol, n):
    def side(us, them, ucol, tcol):
        o = pd.DataFrame({
            "event_id": d.event_id, "season": d.season, "kickoff": d.kickoff,
            "team": d[us], "xgf": d[ucol], "xga": d[tcol],
            "gf": d[f"{us}_goals"], "ga": d[f"{them}_goals"]})
        o["pts"] = np.where(o.gf > o.ga, 3, np.where(o.gf == o.ga, 1, 0))
        o["xgd"] = o.xgf - o.xga
        return o

    long = pd.concat([side("home", "away", hcol, acol),
                      side("away", "home", acol, hcol)]).sort_values("kickoff")
    g = long.groupby(["season", "team"])
    long["xgf_r"] = g["xgd"].transform(lambda s: s.shift().rolling(n).mean())
    long["pts_r"] = g["pts"].transform(lambda s: s.shift().rolling(n).mean())
    long["div"] = long.xgf_r - 0.5 * long.pts_r

    k = ["event_id", "team", "div"]
    m = (d.merge(long[k], left_on=["event_id", "home"],
                 right_on=["event_id", "team"])
           .merge(long[k], left_on=["event_id", "away"],
                  right_on=["event_id", "team"], suffixes=("_h", "_a")))
    m["D"] = m.div_a - m.div_h
    m["awaywin"] = (m.away_goals > m.home_goals).astype(int)
    m["off"] = np.log(m.p_close_away / (1 - m.p_close_away))
    return m.dropna(subset=["D", "off"])


def fit(m, label):
    if len(m) < 60:
        print(f"    {label:<14} n={len(m):<5} too few")
        return
    f = sm.Logit(m.awaywin, sm.add_constant(m[["D"]]), offset=m.off).fit(disp=0)
    b, se = f.params["D"], f.bse["D"]
    sd = m.D.std()
    print(f"    {label:<14} n={len(m):<5} beta={b:+.4f} se={se:.4f} "
          f"z={b/se:+.2f}  per-SD={b*sd:+.4f}  "
          f"CI[{(b-1.96*se)*sd:+.3f},{(b+1.96*se)*sd:+.3f}]")


for n in (5, 10):
    print(f"{'='*78}\nWINDOW {n}\n{'='*78}")
    for tag, hc, ac in (("Sofascore   ", "home_xg", "away_xg"),
                        ("Recalibrated", "home_xg_nln", "away_xg_nln")):
        m = build(df, hc, ac, n)
        print(f"  {tag}")
        fit(m, "all")
        fit(m[m.season == "24/25"], "24/25")
        fit(m[m.season != "24/25"], "25/26+")
    print()

print("CI shown per standard deviation, so the two xG versions are directly")
print("comparable. Read 25/26+ — it is the only split neither the original")
print("hypothesis nor the FootyStats estimate ever saw.")
