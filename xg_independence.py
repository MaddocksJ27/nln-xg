"""
Is Sofascore's NLN xG independent information, or a shot-count transform?

Two tests, same shape as the FootyStats run:
  1. xG ~ shots + shots_on_target.  R^2 near 0.90 => it's a transform.
  2. Does the residual from (1) predict goals, conditional on the fitted part?
     If not, the "extra" in xG carries no signal about scoring.

Input: nln_xg.csv from `sofascore_nln.py backfill`.
"""

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("nln_xg.csv")

# Stack to one row per team-match. Home and away are separate observations.
home = df.rename(columns={
    "home_xg": "xg", "away_xg": "xg_opp",
    "home_shots": "shots", "home_sot": "sot",
    "home_goals": "goals",
})[["season", "event_id", "xg", "shots", "sot", "goals"]]
home["is_home"] = 1

away = df.rename(columns={
    "away_xg": "xg", "home_xg": "xg_opp",
    "away_shots": "shots", "away_sot": "sot",
    "away_goals": "goals",
})[["season", "event_id", "xg", "shots", "sot", "goals"]]
away["is_home"] = 0

t = pd.concat([home, away], ignore_index=True)
n_raw = len(t)
t = t.dropna(subset=["xg", "shots", "sot", "goals"])
print(f"{len(t)} team-matches ({n_raw - len(t)} dropped for missing data)")
print(f"seasons: {sorted(t['season'].unique())}\n")

# --- Test 1: how much of xG is just shot counts? -------------------------
X = sm.add_constant(t[["shots", "sot"]])
m1 = sm.OLS(t["xg"], X).fit()
print("xG ~ shots + SoT")
print(f"  R2        = {m1.rsquared:.4f}")
print(f"  shots     = {m1.params['shots']:+.4f}  (t={m1.tvalues['shots']:+.2f})")
print(f"  SoT       = {m1.params['sot']:+.4f}  (t={m1.tvalues['sot']:+.2f})")
print(f"  resid sd  = {m1.resid.std():.4f}   (xG sd = {t['xg'].std():.4f})\n")

# --- Test 2: does the leftover predict goals? ----------------------------
t["xg_fitted"] = m1.fittedvalues
t["xg_resid"] = m1.resid

X2 = sm.add_constant(t[["xg_fitted", "xg_resid"]])
m2 = sm.OLS(t["goals"], X2).fit()
print("goals ~ fitted(xG) + residual(xG)")
print(f"  fitted    = {m2.params['xg_fitted']:+.4f}  (t={m2.tvalues['xg_fitted']:+.2f})")
print(f"  residual  = {m2.params['xg_resid']:+.4f}  (t={m2.tvalues['xg_resid']:+.2f}, "
      f"p={m2.pvalues['xg_resid']:.3f})")

# --- Benchmark: does xG beat raw shot counts at predicting goals? --------
r_xg = t[["xg", "goals"]].corr().iloc[0, 1]
r_sot = t[["sot", "goals"]].corr().iloc[0, 1]
print(f"\ncorr(xG, goals)  = {r_xg:.4f}")
print(f"corr(SoT, goals) = {r_sot:.4f}")

print("\n---")
if m1.rsquared > 0.85:
    print("R2 > 0.85: same shape as FootyStats. Sofascore's aggregate xG is")
    print("adding little over shot counts. Go to the shotmap and fit your own.")
else:
    print("R2 materially below the FootyStats 0.90 — this is a genuinely")
    print("different measurement. The divergence test is worth running on it.")
