"""
Does combined xG form beat the Over 2.5 line?

PRE-COMMITTED — one specification, decided before looking:
  predictor : combined last-5 xG total (both teams' xG-for, summed, shifted)
  market    : Over 2.5 goals, full time
  model     : logit(over25) ~ T,  offset = logit(market p_over)
  fit on    : 25/26 only
  held out  : 26/27 — NOT touched here

Run once. If it comes back null, that is the answer. Do not sweep 1.5/3.5
or the 10-match window afterwards looking for a better cell.

Step 1: python over25_test.py fetch    (pulls totals odds, ~30 min)
Step 2: python over25_test.py test
"""

import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sofascore_nln import get, seasons, finished_events

LINE = "2.5"
WINDOW = 5


def dec(f):
    try:
        return float(Fraction(f)) + 1.0
    except (ValueError, ZeroDivisionError, TypeError):
        return np.nan


def cmd_fetch():
    rows = []
    for sid, year in seasons()[:3]:
        evs = finished_events(sid)
        print(f"{year}: {len(evs)}")
        for i, e in enumerate(evs, 1):
            d = get(f"event/{e['id']}/odds/1/all")
            if not d:
                continue
            for mk in d.get("markets", []):
                if (mk.get("marketGroup") == "Match goals"
                        and mk.get("marketPeriod") == "Full-time"
                        and mk.get("choiceGroup") == LINE):
                    c = {ch["name"]: ch for ch in mk.get("choices", [])}
                    if "Over" in c and "Under" in c:
                        rows.append({
                            "event_id": e["id"],
                            "open_over": dec(c["Over"]["initialFractionalValue"]),
                            "close_over": dec(c["Over"]["fractionalValue"]),
                            "open_under": dec(c["Under"]["initialFractionalValue"]),
                            "close_under": dec(c["Under"]["fractionalValue"]),
                        })
                    break
            if i % 100 == 0:
                print(f"  {i}/{len(evs)}")
    o = pd.DataFrame(rows)
    o.to_csv("nln_totals.csv", index=False)
    print(f"\nwrote nln_totals.csv — {len(o)} matches with the {LINE} line")


def cmd_test():
    df = pd.read_csv("nln_xg.csv").sort_values("kickoff").reset_index(drop=True)
    od = pd.read_csv("nln_totals.csv")

    # combined xG total per match, then each team's rolling form
    long = pd.concat([
        df[["event_id", "season", "kickoff", "home", "home_xg"]]
          .rename(columns={"home": "team", "home_xg": "xgf"}),
        df[["event_id", "season", "kickoff", "away", "away_xg"]]
          .rename(columns={"away": "team", "away_xg": "xgf"}),
    ]).sort_values("kickoff")

    long["form"] = (long.groupby(["season", "team"])["xgf"]
                        .transform(lambda s: s.shift().rolling(WINDOW).mean()))

    k = ["event_id", "team", "form"]
    m = (df.merge(long[k], left_on=["event_id", "home"], right_on=["event_id", "team"])
           .merge(long[k], left_on=["event_id", "away"], right_on=["event_id", "team"],
                  suffixes=("_h", "_a"))
           .merge(od, on="event_id"))

    m["T"] = m.form_h + m.form_a                       # expected combined xG
    m["over25"] = ((m.home_goals + m.away_goals) > 2.5).astype(int)

    for s in ("open", "close"):
        imp_o, imp_u = 1 / m[f"{s}_over"], 1 / m[f"{s}_under"]
        m[f"p_{s}"] = imp_o / (imp_o + imp_u)          # two-way de-vig

    d = m[m.season == "24/25"].dropna(subset=["T", "p_close"])
    print(f"n = {len(d)}   T: mean={d['T'].mean():.2f} sd={d['T'].std():.2f}")
    print(f"over 2.5 rate: {d.over25.mean():.3f}   "
          f"market implied: {d.p_close.mean():.3f}")
    print(f"mean overround: {(1/d.close_over + 1/d.close_under - 1).mean():.2%}\n")

    for s in ("open", "close"):
        p = d[f"p_{s}"].clip(0.01, 0.99)
        fit = sm.Logit(d["over25"], sm.add_constant(d[["T"]]),
                       offset=np.log(p / (1 - p))).fit(disp=0)
        b, se = fit.params["T"], fit.bse["T"]
        print(f"vs {s}:  beta={b:+.4f}  se={se:.4f}  z={b/se:+.2f}  "
              f"per-SD={b*d['T'].std():+.4f}  "
              f"95% CI [{b-1.96*se:+.3f}, {b+1.96*se:+.3f}]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"fetch": cmd_fetch, "test": cmd_test}[cmd]()
