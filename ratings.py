"""
Opponent-adjusted xG ratings for National League North.

Raw xG differential treats beating the leaders and beating the bottom club
as identical. This fits attack and defence ratings for every team at once,
so each team's number is estimated in the context of who they actually
played.

Model, per season, on match-level xG:

    log(xG_for) = mu + home*is_home + attack[team] + defence[opponent]

Fitted by ridge regression with exponential time decay. Ridge matters here:
with five or six matches played, unpenalised attack/defence estimates swing
wildly, and the penalty shrinks a thin-sample team toward league average
instead. HALF_LIFE controls how fast old matches stop counting.

Ratings reset every season (squad turnover) and are only emitted once a team
has played MIN_GAMES.

Walk-forward: the rating shown for a fixture uses ONLY matches played before
it. No lookahead anywhere.

    python ratings.py              # current season table + next fixtures
    python ratings.py --history    # also write per-match ratings for all seasons
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HALF_LIFE = 60      # days; a match 60 days old counts half as much
ALPHA = 1.0         # ridge penalty; higher = more shrinkage to league average
MIN_GAMES = 5       # no rating published below this
XG_FLOOR = 0.15     # log() needs a floor

USE_NLN_XG = True   # use the recalibrated xG if available


def load():
    df = pd.read_csv("nln_xg.csv")
    df["date"] = pd.to_datetime(df.kickoff)

    if USE_NLN_XG:
        try:
            n = pd.read_csv("nln_match_xg_nln.csv")
            w = n.pivot(index="event_id", columns="is_home", values="xg_nln")
            w.columns = ["away_xg_n", "home_xg_n"]
            df = df.merge(w.reset_index(), on="event_id", how="left")
            hit = df.home_xg_n.notna().mean()
            print(f"using recalibrated xG ({hit:.0%} coverage)")
            df["home_xg"] = df.home_xg_n.fillna(df.home_xg)
            df["away_xg"] = df.away_xg_n.fillna(df.away_xg)
        except FileNotFoundError:
            print("recalibrated xG not found — using Sofascore")

    return df.dropna(subset=["home_xg", "away_xg"]).sort_values("date")


def to_long(d):
    """One row per team-match: who scored the xG, against whom, where."""
    h = pd.DataFrame({"date": d.date, "team": d.home, "opp": d.away,
                      "xg": d.home_xg, "is_home": 1.0})
    a = pd.DataFrame({"date": d.date, "team": d.away, "opp": d.home,
                      "xg": d.away_xg, "is_home": 0.0})
    return pd.concat([h, a], ignore_index=True)


def fit_ratings(long, as_of):
    """Ridge fit on everything strictly before as_of. Returns (att, dfc, meta)."""
    hist = long[long.date < as_of]
    if hist.empty:
        return None, None, None

    teams = sorted(set(hist.team) | set(hist.opp))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(hist), len(teams)

    # columns: [home, attack_1..k, defence_1..k]
    X = np.zeros((n, 1 + 2 * k))
    X[:, 0] = hist.is_home.values
    X[np.arange(n), 1 + hist.team.map(idx).values] = 1.0
    X[np.arange(n), 1 + k + hist.opp.map(idx).values] = 1.0

    y = np.log(hist.xg.clip(lower=XG_FLOOR).values)

    age = (as_of - hist.date).dt.days.values
    w = 0.5 ** (age / HALF_LIFE)

    m = Ridge(alpha=ALPHA, fit_intercept=True)
    m.fit(X, y, sample_weight=w)

    att = pd.Series(m.coef_[1:1 + k], index=teams)
    dfc = pd.Series(m.coef_[1 + k:], index=teams)
    att -= att.mean()                      # identifiability
    dfc -= dfc.mean()

    played = hist.groupby("team").size().reindex(teams).fillna(0)
    meta = {"home": m.coef_[0], "mu": m.intercept_, "played": played}
    return att, dfc, meta


def table(att, dfc, meta):
    """Positive attack = creates more than average. Negative defence = concedes less."""
    t = pd.DataFrame({"P": meta["played"].astype(int),
                      "attack": att, "defence": dfc})
    t["rating"] = t.attack - t.defence      # net, higher is better
    t = t[t.P >= MIN_GAMES]
    return t.sort_values("rating", ascending=False).round(3)


def main():
    df = load()
    season = df.season.iloc[-1]
    cur = df[df.season == season]
    long = to_long(cur)

    as_of = cur.date.max() + pd.Timedelta(days=1)
    att, dfc, meta = fit_ratings(long, as_of)
    if att is None:
        print("no matches yet this season")
        return

    t = table(att, dfc, meta)
    print(f"\nNational League North {season} — opponent-adjusted xG ratings")
    print(f"(as of {as_of.date()}, half-life {HALF_LIFE}d, "
          f"home advantage {np.exp(meta['home']):.3f}x)\n")
    if t.empty:
        print(f"no team has reached {MIN_GAMES} matches yet")
    else:
        print(t.to_string())

    # raw xGD for comparison — this is what the unadjusted table would say
    raw = (long.groupby("team")
               .apply(lambda g: g.xg.mean(), include_groups=False)
               .rename("xgf"))
    ag = (long.groupby("opp").xg.mean().rename("xga"))
    cmp = pd.concat([raw, ag], axis=1)
    cmp["raw_xgd"] = cmp.xgf - cmp.xga
    cmp = cmp.join(t["rating"]).dropna()
    cmp["raw_rank"] = cmp.raw_xgd.rank(ascending=False).astype(int)
    cmp["adj_rank"] = cmp.rating.rank(ascending=False).astype(int)
    cmp["move"] = cmp.raw_rank - cmp.adj_rank

    print("\nbiggest movers (raw xGD rank vs opponent-adjusted rank):")
    print(cmp.reindex(cmp.move.abs().sort_values(ascending=False).index)
             [["raw_xgd", "rating", "raw_rank", "adj_rank", "move"]]
             .head(8).round(3).to_string())
    print("\npositive 'move' = adjusted rating flatters them vs raw xGD,")
    print("i.e. their numbers came against harder opposition.")

    if "--history" in sys.argv:
        rows = []
        for s, d in df.groupby("season"):
            lg = to_long(d)
            for date in sorted(d.date.unique()):
                a, f, mt = fit_ratings(lg, pd.Timestamp(date))
                if a is None:
                    continue
                ok = mt["played"][mt["played"] >= MIN_GAMES].index
                for tm in ok:
                    rows.append({"season": s, "date": date, "team": tm,
                                 "attack": a[tm], "defence": f[tm],
                                 "rating": a[tm] - f[tm]})
        pd.DataFrame(rows).to_csv("nln_ratings_history.csv", index=False)
        print(f"\nwrote nln_ratings_history.csv ({len(rows)} team-dates)")


if __name__ == "__main__":
    main()
