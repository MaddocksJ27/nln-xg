"""
NLN xG form tables — current season, refreshed every run.

Prints last-5 and last-10 xG for / against / differential per team, plus the
next round of fixtures with both sides' numbers side by side.

    python xg_table.py            # current season
    python xg_table.py --csv      # also write xg_table.csv

Finished-match stats are cached permanently (they never change). The fixture
list is re-fetched every run so new results appear.
"""

import sys

import pandas as pd

from sofascore_nln import get, seasons, team_stats, TOURNAMENT_ID

pd.set_option("display.width", 200)


def current_season():
    sid, year = seasons()[0]
    return sid, year


def fetch(sid, kind):
    """kind: 'last' for finished, 'next' for upcoming. Always fresh."""
    page, out = 0, []
    while True:
        path = f"unique-tournament/{TOURNAMENT_ID}/season/{sid}/events/{kind}/{page}"
        d = get(path, ttl_days=0)          # ttl_days=0 -> bypass cache
        if not d or not d.get("events"):
            break
        out += d["events"]
        if not d.get("hasNextPage"):
            break
        page += 1
    return out


def build(sid):
    rows = []
    for e in fetch(sid, "last"):
        if e.get("status", {}).get("type") != "finished":
            continue
        st = team_stats(e["id"])
        xg = st.get("expectedGoals")
        if not xg or xg[0] is None:
            continue
        for us, them, i in ((e["homeTeam"], e["awayTeam"], 0),
                            (e["awayTeam"], e["homeTeam"], 1)):
            rows.append({
                "kickoff": e["startTimestamp"],
                "team": us["name"],
                "opp": them["name"],
                "home": i == 0,
                "xg_for": xg[i],
                "xg_ag": xg[1 - i],
                "gf": e["homeScore" if i == 0 else "awayScore"]["current"],
                "ga": e["awayScore" if i == 0 else "homeScore"]["current"],
            })
    t = pd.DataFrame(rows).sort_values("kickoff")
    t["xgd"] = t.xg_for - t.xg_ag
    return t


def table(t, n):
    """Last n matches per team."""
    recent = t.groupby("team").tail(n)
    agg = recent.groupby("team").agg(
        P=("xgd", "size"),
        xGF=("xg_for", "mean"),
        xGA=("xg_ag", "mean"),
        xGD=("xgd", "mean"),
        GF=("gf", "mean"),
        GA=("ga", "mean"),
    ).round(2)
    agg["over"] = (agg.GF - agg.GA - agg.xGD).round(2)   # results minus underlying
    return agg.sort_values("xGD", ascending=False)


sid, year = current_season()
t = build(sid)
print(f"National League North {year} — {t.team.nunique()} teams, "
      f"{len(t)//2} matches with xG\n")

for n in (5, 10):
    tab = table(t, n)
    short = (tab.P < n).sum()
    print(f"=== last {n} ===" + (f"  ({short} teams have played fewer)" if short else ""))
    print(tab.to_string())
    print()

# ---- next fixtures with both sides' form -------------------------------
f5, f10 = table(t, 5), table(t, 10)
nxt = fetch(sid, "next")[:12]
if nxt:
    print("=== next fixtures ===")
    out = []
    for e in nxt:
        h, a = e["homeTeam"]["name"], e["awayTeam"]["name"]
        out.append({
            "date": pd.to_datetime(e["startTimestamp"], unit="s").strftime("%a %d %b"),
            "home": h, "away": a,
            "H_xGD5": f5.xGD.get(h), "A_xGD5": f5.xGD.get(a),
            "H_xGD10": f10.xGD.get(h), "A_xGD10": f10.xGD.get(a),
        })
    nf = pd.DataFrame(out)
    nf["gap5"] = (nf.H_xGD5 - nf.A_xGD5).round(2)
    nf["gap10"] = (nf.H_xGD10 - nf.A_xGD10).round(2)
    print(nf.to_string(index=False))

if "--csv" in sys.argv:
    table(t, 5).to_csv("xg_table_5.csv")
    table(t, 10).to_csv("xg_table_10.csv")
    print("\nwrote xg_table_5.csv, xg_table_10.csv")
