"""
An NLN-calibrated xG model.

Approach: keep Sofascore's chance ranking (it encodes defender positions and
pressure that the coordinates cannot reconstruct) and correct the two things
it demonstrably gets wrong at this level:
  - the shape of the calibration curve (a single slope was too rigid)
  - situation effects it has no feature for, above all long throws

Train on 24/25, validate on 25/26. 26/27 stays sealed.
Models are nested so each added block earns its place or does not.

    python xg_model.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-4

s = pd.read_csv("nln_shots.csv").merge(
    pd.read_csv("nln_xg.csv")[["event_id", "season"]], on="event_id", how="left")

s = s[(s.situation != "own-goal") & (s.situation != "penalty")].dropna(subset=["xg"])
s["goal"] = (s.shot_type == "goal").astype(int)
s["lg"] = np.log(s.xg.clip(EPS, 1 - EPS) / (1 - s.xg.clip(EPS, 1 - EPS)))

# geometry: x is distance from goal, y is lateral (50 = central)
s["dist"] = np.sqrt(s.x ** 2 + ((s.y - 50) * 0.7) ** 2)
s["lat"] = (s.y - 50).abs()
s["angle"] = np.degrees(np.arctan2(s.lat + EPS, s.x + EPS))

s["situ"] = s.situation.fillna("regular")
s["body"] = s.body_part.fillna("right-foot")

# ---------------------------------------------------------------- 0
print("=" * 64)
print("DIAGNOSTIC — what is 'set-piece' actually?")
print("=" * 64)
d = s.groupby("situ").agg(n=("goal", "size"), goal_rate=("goal", "mean"),
                          sofa_xg=("xg", "mean"), mean_dist=("dist", "mean"),
                          head_pct=("body", lambda c: (c == "head").mean()))
print(d.round(3).to_string())
print("\nIf 'set-piece' looks like corners on distance and headers but converts")
print("far worse, it is likely a residual bucket — treat with suspicion.\n")

train = s[s.season == "24/25"].copy()
test = s[s.season == "25/26"].copy()
print(f"train {len(train)} shots / {train.goal.sum()} goals   "
      f"test {len(test)} shots / {test.goal.sum()} goals\n")

# ---------------------------------------------------------------- models
SPECS = {
    "0 Sofascore raw":      None,                       # baseline, no fit
    "1 linear recal":       "goal ~ lg",
    "2 spline recal":       "goal ~ bs(lg, df=4)",
    "3 + situation":        "goal ~ bs(lg, df=4) + C(situ)",
    "4 + body part":        "goal ~ bs(lg, df=4) + C(situ) + C(body)",
    "5 + geometry":         "goal ~ bs(lg, df=4) + C(situ) + C(body) + dist + angle",
}

results = {}
for name, formula in SPECS.items():
    if formula is None:
        p_tr, p_te = train.xg.values, test.xg.values
        k = 0
    else:
        fit = smf.glm(formula, data=train,
                      family=sm.families.Binomial()).fit()
        p_tr = fit.predict(train).values
        p_te = fit.predict(test).values
        k = int(fit.df_model)
        results[name] = fit

    p_te = np.clip(p_te, EPS, 1 - EPS)
    print(f"{name:<20} params={k:<3} "
          f"test logloss={log_loss(test.goal, p_te):.5f}  "
          f"Brier={brier_score_loss(test.goal, p_te):.5f}  "
          f"sum={p_te.sum():.0f} vs {test.goal.sum()} goals")

print("\nLower logloss is better. A block that does not improve out-of-sample")
print("logloss is not earning its parameters — stop at the last one that does.\n")

# ---------------------------------------------------------------- detail
BEST = "3 + situation"
print("=" * 64)
print(f"COEFFICIENTS — {BEST}")
print("=" * 64)
f = results[BEST]
co = pd.DataFrame({"coef": f.params, "se": f.bse, "z": f.tvalues}).round(3)
print(co[~co.index.str.startswith("cr(")].to_string())
print("\n(situation coefficients are log-odds vs the 'corner' baseline)\n")

test = test.copy()
test["xg_nln"] = np.clip(f.predict(test).values, EPS, 1 - EPS)

BANDS = [0, .03, .05, .08, .12, .2, .35, .6, 1]


def calib(d, col):
    b = pd.cut(d[col], BANDS)
    t = d.groupby(b, observed=True).agg(n=("goal", "size"), pred=(col, "mean"),
                                        actual=("goal", "mean"))
    t["se"] = np.sqrt(t.pred * (1 - t.pred) / t.n)
    t["z"] = ((t.actual - t.pred) / t.se).round(2)
    return t.round(4)


print("Held-out calibration — Sofascore:")
print(calib(test, "xg").to_string())
print("\nHeld-out calibration — NLN model:")
print(calib(test, "xg_nln").to_string())

# ------------------------------------------------- redundancy check
print("\n" + "=" * 64)
print("REDUNDANCY — has it collapsed toward shot counts?")
print("=" * 64)
per = (test.groupby(["event_id", "is_home"])
           .agg(xg_s=("xg", "sum"), xg_n=("xg_nln", "sum"),
                shots=("goal", "size"),
                sot=("shot_type", lambda c: c.isin(["goal", "save"]).sum()))
           .reset_index())
X = sm.add_constant(per[["shots", "sot"]])
for lbl, col in (("Sofascore", "xg_s"), ("NLN model", "xg_n")):
    print(f"  {lbl:<12} R2 on shots+SoT = {sm.OLS(per[col], X).fit().rsquared:.4f}")

# ------------------------------------------------- ship it
full = smf.glm(SPECS[BEST], data=s[s.season != "26/27"],
               family=sm.families.Binomial()).fit()
s["xg_nln"] = np.clip(full.predict(s).values, EPS, 1 - EPS)
s.to_csv("nln_shots_model.csv", index=False)

match = (s.groupby(["event_id", "is_home"])
           .agg(xg_sofa=("xg", "sum"), xg_nln=("xg_nln", "sum"))
           .reset_index())
match.to_csv("nln_match_xg_nln.csv", index=False)
print("\nwrote nln_shots_model.csv and nln_match_xg_nln.csv")
print("(final model fitted on 24/25 + 25/26; 26/27 predictions are out-of-sample)")
