"""
Does the xG-differential tactic beat the OPENING price?

Pre-committed spec — four numbers, no variants:
  window        : 10 matches
  side          : away
  tests         : (a) continuous logit, offset = logit(market p_away)
                  (b) flat-stakes ROI on the top 19% of divergence
  benchmark     : each run against opening and closing price
  sample        : 25/26 onward only

24/25 is excluded because 41% of its "opening" prices are identical to the
close — Sofascore recorded one price and stored it as both. Including them
makes the opening test partly a closing test. Do NOT filter to moved-only
rows instead: that selects on the outcome.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

WINDOW = 10
TOP_Q = 0.81          # top 19%

df = pd.read_csv("nln_merged.csv").sort_values("kickoff").reset_index(drop=True)


def side(d, us, them):
    o = pd.DataFrame({
        "event_id": d["event_id"], "season": d["season"], "kickoff": d["kickoff"],
        "team": d[us], "xg_for": d[f"{us}_xg"], "xg_ag": d[f"{them}_xg"],
        "gf": d[f"{us}_goals"], "ga": d[f"{them}_goals"],
    })
    o["xgd"] = o.xg_for - o.xg_ag
    return o


long = pd.concat([side(df, "home", "away"), side(df, "away", "home")],
                 ignore_index=True).sort_values("kickoff")

g = long.groupby(["season", "team"])
long["form"] = g["xgd"].transform(lambda s: s.shift().rolling(WINDOW).mean())

keep = ["event_id", "team", "form"]
m = (df.merge(long[keep], left_on=["event_id", "home"], right_on=["event_id", "team"])
       .merge(long[keep], left_on=["event_id", "away"], right_on=["event_id", "team"],
              suffixes=("_h", "_a")))

m["D"] = m["form_a"] - m["form_h"]
m["awaywin"] = (m.away_goals > m.home_goals).astype(int)

m = m[m.season != "24/25"].dropna(subset=["D"]).copy()
print(f"sample: {len(m)} matches, seasons {sorted(m.season.unique())}")
print(f"D: sd={m.D.std():.4f}\n")

thresh = m.D.quantile(TOP_Q)
sel = m[m.D >= thresh]
print(f"threshold D >= {thresh:.3f} -> {len(sel)} selections\n")

for stage in ("open", "close"):
    p = m[f"p_{stage}_away"]
    fit = sm.Logit(m["awaywin"], sm.add_constant(m[["D"]]),
                   offset=np.log(p / (1 - p))).fit(disp=0)
    b, se = fit.params["D"], fit.bse["D"]

    price = sel[f"{stage}_away"]
    ret = np.where(sel.awaywin == 1, price - 1, -1)
    roi = ret.mean()
    roi_se = ret.std() / np.sqrt(len(ret))

    print(f"--- vs {stage} price ---")
    print(f"  continuous  beta={b:+.4f}  se={se:.4f}  z={b/se:+.2f}"
          f"   per-SD={b*m.D.std():+.4f}")
    print(f"  ROI top19%  {roi:+.3%}  (se {roi_se:.3%})  "
          f"strike {sel.awaywin.mean():.3f}  avg price {price.mean():.2f}")
    print(f"              breakeven needs {1/price.mean():.3f}\n")

print("Read the ROI standard error before the ROI. At n~120 with prices near 3,")
print("one se is roughly 15pp — anything under +15% is indistinguishable from zero.")
