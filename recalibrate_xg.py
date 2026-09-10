"""
Recalibrate Sofascore xG for National League North.

Step 1 (diagnostic): does the slope compression hold in every season?
        If it only shows in one, it is noise or a model change, not a
        property of the level. Stop there.

Step 2 (fit):  logit(P goal) = a + b * logit(xg_sofascore)
        b < 1 means their model is over-confident at this level:
        low chances convert more than stated, high chances less.

Step 3 (residual): does anything remain by situation or body part?
        Only add terms where the data says to.

Step 4 (sanity): is the recalibrated xG closer to raw shot counts than
        Sofascore's was? R^2 on shots+SoT was 0.589 for theirs. If the
        recalibrated version drifts materially above that, it is moving
        toward the metric that already showed no predictive power.

    python recalibrate_xg.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 1e-4

s = pd.read_csv("nln_shots.csv")
xg_m = pd.read_csv("nln_xg.csv")[["event_id", "season"]]
s = s.merge(xg_m, on="event_id", how="left")

print(f"{len(s)} shots loaded")

# --- clean -------------------------------------------------------------
s = s[s.situation != "own-goal"]
s = s.dropna(subset=["xg"])
pens = s[s.situation == "penalty"]
s = s[s.situation != "penalty"].copy()
print(f"{len(s)} open-play/set-piece shots after dropping "
      f"own goals and {len(pens)} penalties")
print(f"penalty conversion: {(pens.shot_type=='goal').mean():.3f} "
      f"(Sofascore says {pens.xg.mean():.3f})\n")

s["goal"] = (s.shot_type == "goal").astype(int)
s["lg"] = np.log(s.xg.clip(EPS, 1 - EPS) / (1 - s.xg.clip(EPS, 1 - EPS)))

BANDS = [0, .03, .05, .08, .12, .2, .35, .6, 1]


def calib(d):
    b = pd.cut(d.xg, BANDS)
    t = d.groupby(b, observed=True).agg(n=("goal", "size"),
                                        pred=("xg", "mean"),
                                        actual=("goal", "mean"))
    # standard error of the observed rate, and z vs prediction
    t["se"] = np.sqrt(t.pred * (1 - t.pred) / t.n)
    t["z"] = ((t.actual - t.pred) / t.se).round(2)
    return t.round(4)


# --- Step 1: per season ------------------------------------------------
print("=" * 62)
print("STEP 1 — calibration by season")
print("=" * 62)
for season, d in s.groupby("season"):
    print(f"\n--- {season}  (n={len(d)}, goals={d.goal.sum()}) ---")
    print(calib(d).to_string())

print("\n\n--- pooled ---")
print(calib(s).to_string())

# --- Step 2: recalibration fit -----------------------------------------
print("\n" + "=" * 62)
print("STEP 2 — logit(goal) ~ logit(xg)")
print("=" * 62)

fit = sm.Logit(s.goal, sm.add_constant(s[["lg"]])).fit(disp=0)
a, b = fit.params["const"], fit.params["lg"]
print(f"\nintercept a = {a:+.4f}  (se {fit.bse['const']:.4f})")
print(f"slope     b = {b:+.4f}  (se {fit.bse['lg']:.4f})")
print(f"            95% CI [{b-1.96*fit.bse['lg']:.4f}, "
      f"{b+1.96*fit.bse['lg']:.4f}]")
print("b = 1 and a = 0 would mean Sofascore is perfectly calibrated here.")

s["xg_cal"] = 1 / (1 + np.exp(-(a + b * s.lg)))

print(f"\ntotal goals            {s.goal.sum()}")
print(f"Sofascore xG sum       {s.xg.sum():.0f}")
print(f"recalibrated xG sum    {s.xg_cal.sum():.0f}")

print("\ncalibration after recalibration:")
t2 = s.copy()
t2["xg"] = t2.xg_cal
print(calib(t2).to_string())

# --- Step 3: residual structure ----------------------------------------
print("\n" + "=" * 62)
print("STEP 3 — residual by situation / body part")
print("=" * 62)
s["resid"] = s.goal - s.xg_cal
for col in ("situation", "body_part"):
    g = s.groupby(col).agg(n=("goal", "size"), pred=("xg_cal", "mean"),
                           actual=("goal", "mean"))
    g["se"] = np.sqrt(g.pred * (1 - g.pred) / g.n)
    g["z"] = ((g.actual - g.pred) / g.se).round(2)
    print(f"\n{col}:")
    print(g.round(4).to_string())
print("\n|z| > 2.5 on a decent n means that category needs its own term.")

# --- Step 4: has it drifted toward shot counts? ------------------------
print("\n" + "=" * 62)
print("STEP 4 — is recalibrated xG just shot counts again?")
print("=" * 62)

per = (s.groupby(["event_id", "is_home"])
         .agg(xg_s=("xg", "sum"), xg_c=("xg_cal", "sum"),
              shots=("goal", "size"),
              sot=("shot_type", lambda c: c.isin(["goal", "save"]).sum()))
         .reset_index())

X = sm.add_constant(per[["shots", "sot"]])
for lbl, col in (("Sofascore", "xg_s"), ("recalibrated", "xg_c")):
    r2 = sm.OLS(per[col], X).fit().rsquared
    print(f"  {lbl:<14} R2 on shots+SoT = {r2:.4f}")
print("\nSofascore's match-level figure gave 0.589 earlier. A materially")
print("higher number for the recalibrated version means it has moved toward")
print("the shot-count metric that showed no predictive power.")

s.to_csv("nln_shots_calibrated.csv", index=False)
print("\nwrote nln_shots_calibrated.csv")
