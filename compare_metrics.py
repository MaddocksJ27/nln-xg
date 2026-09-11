"""
Opponent-adjusted ratings vs raw xG differential — like for like.

Two questions, in order of how much they matter:

  1. AGREEMENT. Do the two metrics rank teams the same way and pick the same
     matches? If they agree ~90% of the time, the opponent adjustment is
     cosmetic and the movers table is prettier than it is useful.

  2. SELECTION. At matched percentile thresholds (so the same NUMBER of bets
     under each metric, on the same matches), does either produce a better
     return? The scales differ — raw xGD is goals per match, the rating is a
     log-scale attack-minus-defence — so percentile cuts are the only fair
     comparison. Thresholding both at "0.5" would compare nothing.

Expect both to be null. Question 1 is the one worth the run.

    python compare_metrics.py
"""

import numpy as np
import pandas as pd

PCTILES = [0.30, 0.20, 0.10, 0.05]
WINDOW = 10

od = pd.read_csv("nln_merged.csv")
od["date"] = pd.to_datetime(od.kickoff).dt.normalize()

r = pd.read_csv("nln_ratings_history.csv")
r["date"] = pd.to_datetime(r.date).dt.normalize()
r = (r.sort_values("date")
       .drop_duplicates(["season", "date", "team"], keep="last")
       [["season", "date", "team", "rating"]])

# ---- raw xGD form over the last WINDOW, same shift discipline ----------
xg = pd.read_csv("nln_xg.csv").sort_values("kickoff")
try:
    n = pd.read_csv("nln_match_xg_nln.csv")
    w = n.pivot(index="event_id", columns="is_home", values="xg_nln")
    w.columns = ["a_n", "h_n"]
    xg = xg.merge(w.reset_index(), on="event_id", how="left")
    xg["home_xg"] = xg.h_n.fillna(xg.home_xg)
    xg["away_xg"] = xg.a_n.fillna(xg.away_xg)
    print("raw-xGD arm also uses recalibrated xG (fair comparison)\n")
except FileNotFoundError:
    pass

long = pd.concat([
    pd.DataFrame({"event_id": xg.event_id, "season": xg.season,
                  "kickoff": xg.kickoff, "team": xg.home,
                  "xgd": xg.home_xg - xg.away_xg}),
    pd.DataFrame({"event_id": xg.event_id, "season": xg.season,
                  "kickoff": xg.kickoff, "team": xg.away,
                  "xgd": xg.away_xg - xg.home_xg}),
]).sort_values("kickoff")
long["form"] = (long.groupby(["season", "team"])["xgd"]
                    .transform(lambda s: s.shift().rolling(WINDOW).mean()))

k = ["event_id", "team", "form"]
m = (od.merge(long[k], left_on=["event_id", "home"],
              right_on=["event_id", "team"]).rename(columns={"form": "f_home"})
       .drop(columns="team")
       .merge(long[k], left_on=["event_id", "away"],
              right_on=["event_id", "team"]).rename(columns={"form": "f_away"})
       .drop(columns="team")
       .merge(r, left_on=["season", "date", "home"],
              right_on=["season", "date", "team"], how="inner")
       .rename(columns={"rating": "r_home"}).drop(columns="team")
       .merge(r, left_on=["season", "date", "away"],
              right_on=["season", "date", "team"], how="inner")
       .rename(columns={"rating": "r_away"}).drop(columns="team")
       .dropna(subset=["f_home", "f_away", "r_home", "r_away"]))

m["gap_rat"] = m.r_home - m.r_away
m["gap_raw"] = m.f_home - m.f_away
m["home_won"] = (m.home_goals > m.away_goals).astype(int)
m["away_won"] = (m.away_goals > m.home_goals).astype(int)

print(f"{len(m)} matches with BOTH metrics available")
print(m.groupby("season").size().to_string())

# ---------------------------------------------------------------- Q1
print("\n" + "=" * 72)
print("Q1 — DO THE TWO METRICS AGREE?")
print("=" * 72)

print(f"\nrank correlation (Spearman): "
      f"{m.gap_rat.corr(m.gap_raw, method='spearman'):.4f}")
print(f"sign agreement (same team favoured): "
      f"{(np.sign(m.gap_rat) == np.sign(m.gap_raw)).mean():.4f}")

for p in PCTILES:
    kq = int(len(m) * p)
    top_rat = set(m.nlargest(kq, "gap_rat").index) | set(m.nsmallest(kq, "gap_rat").index)
    top_raw = set(m.nlargest(kq, "gap_raw").index) | set(m.nsmallest(kq, "gap_raw").index)
    ov = len(top_rat & top_raw) / len(top_rat)
    print(f"  top {int(p*100):>2}% by |gap|: {ov:.1%} of selections shared")

print("\nHigh overlap => the adjustment mostly reorders teams it already")
print("agreed about. Low overlap => it genuinely changes the picture.")

# ---------------------------------------------------------------- Q2
def bet(d, col, p):
    """Back the better-rated side, top p by |gap| in each direction."""
    kq = max(int(len(d) * p / 2), 1)
    hs = d.nlargest(kq, col)
    aw = d.nsmallest(kq, col)
    ret = np.concatenate([
        np.where(hs.home_won == 1, hs.close_home - 1, -1.0),
        np.where(aw.away_won == 1, aw.close_away - 1, -1.0)])
    return len(ret), ret.mean(), ret.std() / np.sqrt(len(ret))


print("\n" + "=" * 72)
print("Q2 — MATCHED-PERCENTILE RETURNS")
print("=" * 72)
print(f"\n{'top %':>6} | {'adjusted rating':^34} | {'raw xGD form':^34}")
print(f"{'':>6} | {'all':^10} {'24/25':^11} {'25/26':^11} "
      f"| {'all':^10} {'24/25':^11} {'25/26':^11}")
print("-" * 80)

for p in PCTILES:
    cells = []
    for col in ("gap_rat", "gap_raw"):
        for d in (m, m[m.season == "24/25"], m[m.season == "25/26"]):
            n, roi, se = bet(d, col, p)
            cells.append(f"{roi:+.1%}({n})")
    print(f"{int(p*100):>5}% | {cells[0]:^10} {cells[1]:^11} {cells[2]:^11} "
          f"| {cells[3]:^10} {cells[4]:^11} {cells[5]:^11}")

print("\nSame bet count under each metric by construction, so any difference")
print("is the metric. Standard errors at these sample sizes run 8-25pp —")
print("differences smaller than that are noise, not a better metric.")
