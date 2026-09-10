"""
An NLN-calibrated xG model.

Keeps Sofascore's chance ranking (it encodes defender position and pressure
that x/y cannot reconstruct) and corrects the two things it gets wrong here:
  - the shape of the calibration curve
  - situation effects it has no feature for, above all long throws

Splines are built manually as a fixed-knot linear basis, so there is no patsy
knot-boundary problem and no rank deficiency. Knots sit at quantiles of the
FULL dataset, so train and test share an identical basis.

Train 24/25, validate 25/26, 26/27 sealed.

    python xg_model2.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-4
BANDS = [0, .03, .05, .08, .12, .2, .35, .6, 1]

s = pd.read_csv("nln_shots.csv").merge(
    pd.read_csv("nln_xg.csv")[["event_id", "season"]], on="event_id", how="left")

s = s[~s.situation.isin(["own-goal", "penalty"])].dropna(subset=["xg"]).copy()
s["goal"] = (s.shot_type == "goal").astype(int)
p = s.xg.clip(EPS, 1 - EPS)
s["lg"] = np.log(p / (1 - p))
s["situ"] = s.situation.fillna("regular")
s["body"] = s.body_part.fillna("right-foot")
s["dist"] = np.sqrt(s.x ** 2 + ((s.y - 50) * 0.7) ** 2)
s["angle"] = np.degrees(np.arctan2((s.y - 50).abs() + EPS, s.x + EPS))

# ---- piecewise-linear spline basis on lg, knots from the full data ------
KNOTS = np.quantile(s.lg, [0.2, 0.4, 0.6, 0.8])
print(f"spline knots on logit(xg): {np.round(KNOTS, 3)}")
print(f"  = xG of {np.round(1/(1+np.exp(-KNOTS)), 4)}\n")


def basis(d):
    X = pd.DataFrame({"lg": d.lg.values}, index=d.index)
    for i, k in enumerate(KNOTS):
        X[f"lg_k{i}"] = np.maximum(d.lg.values - k, 0)
    return X


def design(d, blocks):
    parts = []
    if "spline" in blocks:
        parts.append(basis(d))
    else:
        parts.append(pd.DataFrame({"lg": d.lg.values}, index=d.index))
    if "situ" in blocks:
        parts.append(pd.get_dummies(d.situ, prefix="s", drop_first=True)
                       .astype(float).set_axis(d.index))
    if "body" in blocks:
        parts.append(pd.get_dummies(d.body, prefix="b", drop_first=True)
                       .astype(float).set_axis(d.index))
    if "geom" in blocks:
        parts.append(d[["dist", "angle"]])
    return sm.add_constant(pd.concat(parts, axis=1), has_constant="add")


def calib(d, col):
    b = pd.cut(d[col], BANDS)
    t = d.groupby(b, observed=True).agg(n=("goal", "size"), pred=(col, "mean"),
                                        actual=("goal", "mean"))
    t["se"] = np.sqrt(t.pred * (1 - t.pred) / t.n)
    t["z"] = ((t.actual - t.pred) / t.se).round(2)
    return t.round(4)


# ---- diagnostic --------------------------------------------------------
print("=" * 64)
print("DIAGNOSTIC — situation categories")
print("=" * 64)
print(s.groupby("situ").agg(n=("goal", "size"), goal_rate=("goal", "mean"),
                            sofa_xg=("xg", "mean"), mean_dist=("dist", "mean"),
                            head_pct=("body", lambda c: (c == "head").mean())
                            ).round(3).to_string())
print("\n'set-piece' predicting highest but converting poorly suggests a")
print("residual bucket rather than a real effect — do not build on it.\n")

train = s[s.season == "24/25"]
test = s[s.season == "25/26"]
print(f"train {len(train)} shots / {train.goal.sum()} goals   "
      f"test {len(test)} shots / {test.goal.sum()} goals\n")

# ---- nested ladder -----------------------------------------------------
SPECS = {
    "0 Sofascore raw":  None,
    "1 linear recal":   [],
    "2 spline recal":   ["spline"],
    "3 + situation":    ["spline", "situ"],
    "4 + body part":    ["spline", "situ", "body"],
    "5 + geometry":     ["spline", "situ", "body", "geom"],
}

fits = {}
print("=" * 64)
print("LADDER — evaluated on 25/26, never seen in training")
print("=" * 64)
for name, blocks in SPECS.items():
    if blocks is None:
        pt = test.xg.values
        k = 0
    else:
        Xtr, Xte = design(train, blocks), design(test, blocks)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
        f = sm.GLM(train.goal, Xtr, family=sm.families.Binomial()).fit()
        fits[name] = (f, blocks, Xtr.columns)
        pt = f.predict(Xte).values
        k = len(Xtr.columns) - 1
    pt = np.clip(pt, EPS, 1 - EPS)
    print(f"{name:<18} k={k:<3} logloss={log_loss(test.goal, pt):.5f}  "
          f"Brier={brier_score_loss(test.goal, pt):.5f}  "
          f"sum={pt.sum():.0f} vs {test.goal.sum()}")

print("\nStop at the last block that improves out-of-sample logloss.\n")

# ---- detail on the chosen model ---------------------------------------
BEST = "3 + situation"
f, blocks, cols = fits[BEST]
print("=" * 64)
print(f"COEFFICIENTS — {BEST}")
print("=" * 64)
co = pd.DataFrame({"coef": f.params, "se": f.bse, "z": f.tvalues}).round(3)
print(co[~co.index.str.startswith("lg")].to_string())
print("\n(situation dummies are log-odds vs the omitted 'corner' baseline)")
print(f"condition number: {np.linalg.cond(design(train, blocks).values):.1f}"
      "   (under ~1000 is fine)\n")

te = test.copy()
te["xg_nln"] = np.clip(
    f.predict(design(te, blocks).reindex(columns=cols, fill_value=0.0)).values,
    EPS, 1 - EPS)

print("Held-out calibration — Sofascore:")
print(calib(te, "xg").to_string())
print("\nHeld-out calibration — NLN model:")
print(calib(te, "xg_nln").to_string())

# ---- redundancy --------------------------------------------------------
print("\n" + "=" * 64)
print("REDUNDANCY — collapsed toward shot counts?")
print("=" * 64)
per = (te.groupby(["event_id", "is_home"])
         .agg(xg_s=("xg", "sum"), xg_n=("xg_nln", "sum"), shots=("goal", "size"),
              sot=("shot_type", lambda c: c.isin(["goal", "save"]).sum()))
         .reset_index())
X = sm.add_constant(per[["shots", "sot"]])
for lbl, c in (("Sofascore", "xg_s"), ("NLN model", "xg_n")):
    print(f"  {lbl:<12} R2 on shots+SoT = {sm.OLS(per[c], X).fit().rsquared:.4f}")

# ---- final fit on both full seasons, applied to everything -------------
fin_tr = s[s.season != "26/27"]
Xf = design(fin_tr, blocks)
ffin = sm.GLM(fin_tr.goal, Xf, family=sm.families.Binomial()).fit()
s["xg_nln"] = np.clip(
    ffin.predict(design(s, blocks).reindex(columns=Xf.columns, fill_value=0.0)).values,
    EPS, 1 - EPS)

s.to_csv("nln_shots_model.csv", index=False)
(s.groupby(["event_id", "is_home"])
   .agg(xg_sofa=("xg", "sum"), xg_nln=("xg_nln", "sum"))
   .reset_index()
   .to_csv("nln_match_xg_nln.csv", index=False))
print("\nwrote nln_shots_model.csv and nln_match_xg_nln.csv")
print("final model fitted on 24/25 + 25/26; 26/27 predictions are out-of-sample")
