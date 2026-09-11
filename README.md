# National League North: xG calibration and market efficiency

Quantitative study of English tier-6 football using shot-level data for
1,144 matches across three seasons (2024/25 – 2026/27).

Two findings:

1. **Sofascore's xG model is miscalibrated at this level, and the error has
   structure.** Clear chances convert less often than predicted, and shots
   originating from long throw-ins convert roughly 37% more often than the
   model expects (+0.474 log-odds vs corners, z = 3.67, out of sample).
   Commercial xG models are trained on elite football, where a throw-in
   rarely produces a chance. In non-league it is a deliberate weapon.

2. **The market prices recent xG form efficiently.** Five pre-registered
   specifications, tested against both opening and closing prices, all null.
   This holds under Sofascore's xG *and* under a recalibrated model built
   here — so the null is not an artifact of a noisy instrument.

---

## The calibration problem

Sofascore's shot-level xG, evaluated against 25,870 non-penalty NLN shots:

| xG band | n | predicted | actual | z |
|---|---|---|---|---|
| 0.03 – 0.05 | 3,082 | 0.040 | 0.055 | **+4.38** |
| 0.05 – 0.08 | 3,411 | 0.064 | 0.086 | **+5.25** |
| 0.08 – 0.12 | 3,264 | 0.099 | 0.103 | +0.75 |
| 0.12 – 0.20 | 3,729 | 0.155 | 0.161 | +1.04 |
| 0.20 – 0.35 | 2,826 | 0.263 | 0.231 | **−3.85** |
| 0.35 – 0.60 | 1,339 | 0.448 | 0.382 | **−4.81** |
| 0.60 – 1.00 | 390 | 0.709 | 0.649 | **−2.64** |

The high-band under-conversion replicates in both full seasons. The
low-band over-conversion is largely a 2025/26 effect and is reported here
as the weaker of the two patterns.

Penalties convert at 0.774 against a predicted 0.760 — tier-6 penalty
taking is fine. The problem is open play.

## The model

Rather than fitting xG from scratch, the model **corrects** Sofascore's
figure. Their xG encodes defender positions, pressure and assist type,
none of which is reconstructable from the x/y coordinates available here;
discarding that ranking to fix calibration would be a bad trade.

`logit(P(goal)) = piecewise-linear spline on logit(xG_sofascore) + situation`

Trained on 2024/25, validated on 2025/26, with 2026/27 held out entirely.
Nested model selection by out-of-sample log-loss:

| model | k | test log-loss |
|---|---|---|
| Sofascore raw | 0 | 0.30120 |
| linear recalibration | 1 | 0.29952 |
| spline recalibration | 5 | 0.29892 |
| **+ situation** | **9** | **0.29838** |
| + body part | 11 | 0.29835 |
| + geometry | 13 | 0.29851 |

Geometry adds nothing beyond what Sofascore already encodes. Body part
does not earn its parameters. Model 3 ships.

Held-out calibration improves from four of eight bands inside ±2 to seven
of eight.

## Market tests

Every test uses de-vigged market probability as a fixed offset, so the
question is always *does this add information the price lacks*, never
*does the favourite win more often*.

| # | hypothesis | market | out-of-sample result |
|---|---|---|---|
| 1 | xG-vs-points divergence | 1X2 close | −0.000 per SD |
| 2 | pure xG differential | 1X2 close | +0.015 per SD |
| 3 | threshold selection (top 5/9/19%) | 1X2 close | negative in 11 of 12 cells |
| 4 | combined xG form | Over 2.5 | +0.31/SD in 25/26, **−0.10/SD in 24/25** |
| 5 | divergence on recalibrated xG | 1X2 close | +0.040 per SD |

Test 4 is the instructive one. It came back at z = 2.67 against the
opening price on 2025/26 — the first positive result in the project. Run
on 2024/25 as an independent replication, the sign reversed. 2025/26 saw
60.7% of matches go over 2.5 against 47.2% the previous season, so any
goal-correlated predictor would show a positive coefficient in that year
regardless of whether it carried information.

Test 5 exists because tests 1–4 used a measurement now known to be biased.
Recalibrated and Sofascore xG correlate at 0.943 at match level, so the
inputs genuinely differ — but the estimates are near identical
(+0.0417 vs +0.0398 per SD, window 10). Shot-level calibration error
averages out across ~12 shots per match. The nulls are real.

Across all tests, in-sample estimates on the discovery season run
+0.12 to +0.20 per SD and collapse to +0.02 to +0.04 out of sample —
an overfitting signature that appears under two independent instruments.

## Limitations

- **Redundancy drift.** Recalibration raises the R² of match xG on raw
  shots and shots-on-target from 0.606 to 0.655. The corrected metric is
  measurably closer to a shot count than the original, which is a cost
  worth naming rather than burying.
- **The intercept does not transfer across seasons.** Trained on a 47%-overs
  season, the model under-predicts a 61%-overs season by 38 goals. Use it
  for within-season comparison, not cross-season levels.
- **The lowest band remains miscalibrated** (z = +2.98 held out). The spline
  cannot reach that far.
- **Discrimination is barely improved.** Log-loss falls under 1%. The gain
  is calibration, not ranking — appropriate for aggregating to match xG,
  but this model does not rank chances better than Sofascore's.
- **`set-piece` is probably a residual bucket** (highest predicted xG of any
  category, near-worst conversion). The term is fitted but claimed as
  nothing.
- **Coverage gaps.** Farsley Celtic are missing ~15 fixtures, having played
  home games at borrowed venues. 5.5% overall attrition.
- 2026/27 is held out and remains unused.

## Files

| file | purpose |
|---|---|
| `sofascore_nln.py` | scraper — fixtures, team stats, shot maps |
| `fetch_odds.py` | 1X2 opening and closing prices, de-vigged |
| `xg_independence.py` | is vendor xG a shot-count transform? |
| `xg_model.py` | the calibration model |
| `divergence_test.py` | tests 1–3 |
| `over25_test.py` | test 4 |
| `divergence_recal.py` | test 5 |
| `xg_table.py` | live form tables, refreshed each run |

## Reproducing

```bash
pip install curl_cffi==0.13.0 pandas statsmodels scikit-learn
python sofascore_nln.py check      # verify coverage first
python sofascore_nln.py backfill
python sofascore_nln.py shots
python xg_model.py
```

Data is not redistributed here — Sofascore's terms prohibit it. The
scraper reproduces it in roughly two hours, caching every response so
re-runs are free.

## Method notes

Specifications were fixed in writing before each test after an early
finding that a promising strategy was a 1-in-10 draw from an 11,000-variant
search. Test 4 is the case where this mattered: it produced an encouraging
number that did not survive replication, and would have been reported as a
result under a less disciplined process.
## Opponent-adjusted ratings

Raw xG differential over a rolling window treats a result against the
leaders and one against the bottom club as identical. This model estimates
attack and defence for every team simultaneously, so each rating is fitted
in the context of who that team actually played.

```
log(xG_for) = mu + home·is_home + attack[team] + defence[opponent]
```

Ridge-penalised, with exponential time decay (60-day half-life). The
penalty matters: with five or six matches played, unpenalised attack and
defence estimates swing wildly, and shrinkage pulls a thin-sample team
toward league average instead. Ratings reset every season — squad turnover
at this level makes carryover meaningless — and are published only once a
team reaches five matches.

Fitting is walk-forward: the rating attached to a fixture uses only matches
played before it. Estimated home advantage is 1.22x.

### Does the adjustment change anything?

Compared against raw 10-match xG differential on the same fixtures, using
the same xG input for both:

| | agreement |
|---|---|
| rank correlation (Spearman) | 0.891 |
| same team favoured | 85.1% |
| shared selections, top 30% by gap | 80.3% |
| shared selections, top 20% by gap | 77.0% |
| shared selections, top 10% by gap | 69.4% |
| **shared selections, top 5% by gap** | **53.3%** |

The two metrics broadly agree on routine fixtures and diverge precisely
where the gap is largest — the matches you would look at hardest. At the
top 5%, they disagree about which fixture is the biggest mismatch roughly
half the time.

So the adjustment is not cosmetic. As an illustration from the live table,
Hereford sit 11th on raw xG differential and 5th adjusted, because their
flat raw numbers came against harder opposition than the leaders' better
ones.

**Which metric is more accurate is not settled here.** That requires
predicting results out of sample with more power than the 315 matches
carrying both metrics can provide. The matched-percentile return
comparison in `compare_metrics.py` reduces to six bets per cell at the
sharper thresholds, where standard errors run 25–30pp — it is reported for
completeness and reads as noise. Revisit once 2026/27 completes and a third
full season is available.

### Test 6: rating-gap thresholds

Backing the better-rated side at gap thresholds of 0.5 / 0.75 / 1.0 / 1.25 /
1.5, across home, away and both:

| threshold | all seasons | 24/25 | 25/26 |
|---|---|---|---|
| 0.50 | −1.25% (321) | +6.62% | −12.08% |
| 0.75 | −1.94% (136) | +2.45% | −3.46% |
| 1.00 | −1.57% (48) | −2.66% | +24.78% |
| 1.25 | −32.50% (15) | −19.68% | −35.50% |
| 1.50 | −59.74% (6) | −39.61% | n/a |

Null, and informatively so: returns get **worse** as the rating gap widens.
The hypothesis predicts the opposite — the largest mismatches should carry
the strongest signal. Instead the extremes are where the losses concentrate.
Sample sizes at the top two thresholds are too small to read on their own,
but nothing in the pattern points toward a hidden edge.

This is the sixth independent specification tested against the closing line,
and the sixth null.

### Files

| file | purpose |
|---|---|
| `ratings.py` | the ratings model; `--history` writes the walk-forward series |
| `rating_strategy.py` | test 6, the threshold sweep |
| `compare_metrics.py` | adjusted vs raw, agreement and returns |
