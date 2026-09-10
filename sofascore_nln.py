"""
Sofascore -> National League North match xG (and optional shot-level xG).

Tournament id 176 = National League North (Sofascore files it under
"england-amateur": /football/tournament/england-amateur/national-league-north/176).

Run `python sofascore_nln.py check` FIRST. It samples ~15 finished matches and
reports what fraction actually carry an `expectedGoals` stat and a populated
shotmap. If that comes back near zero, stop — Sofascore does not collect
shot-location data at this level and there is nothing to backfill.

Then `python sofascore_nln.py backfill` to pull all seasons.

Everything is cached to ./cache as raw JSON, so re-runs cost nothing and a
rate-limit hit loses at most one request.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from curl_cffi.requests import Session as CurlSession


TOURNAMENT_ID = 176          # National League North
BASE = "https://www.sofascore.com/api/v1"   # api.sofascore.com now 403s
CACHE = Path("cache")
OUT = Path("nln_xg.csv")

SESSION = CurlSession(impersonate="chrome")
SESSION.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json",
})


def get(path, ttl_days=None):
    """GET with on-disk cache, jittered rate limit, and 429 backoff."""
    key = CACHE / (path.strip("/").replace("/", "_") + ".json")
    key.parent.mkdir(parents=True, exist_ok=True)
    if key.exists():
        if ttl_days is None or time.time() - key.stat().st_mtime < ttl_days * 86400:
            return json.loads(key.read_text())

    wait = 30
    for attempt in range(5):
        time.sleep(1.5 + random.uniform(-0.4, 0.6))
        r = SESSION.get(f"{BASE}/{path}", timeout=20)
        if r.status_code == 404:
            key.write_text("null")
            return None
        if r.status_code in (429, 403):
            print(f"  {r.status_code} — backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            wait *= 2
            continue
        r.raise_for_status()
        key.write_text(r.text)
        return r.json()
    raise RuntimeError(f"gave up on {path}")


def seasons():
    d = get(f"unique-tournament/{TOURNAMENT_ID}/seasons", ttl_days=7)
    return [(s["id"], s["year"]) for s in d["seasons"]]


def finished_events(season_id):
    """Page backwards through completed fixtures. 30 per page."""
    page, out = 0, []
    while True:
        d = get(f"unique-tournament/{TOURNAMENT_ID}/season/{season_id}/events/last/{page}")
        if not d or not d.get("events"):
            break
        for e in d["events"]:
            if e.get("status", {}).get("type") == "finished":
                out.append(e)
        if not d.get("hasNextPage"):
            break
        page += 1
    return out


def team_stats(event_id):
    """Return {'expectedGoals': (home, away), ...} for the full match, or {}."""
    d = get(f"event/{event_id}/statistics")
    if not d:
        return {}
    for block in d.get("statistics", []):
        if block.get("period") != "ALL":
            continue
        out = {}
        for group in block.get("groups", []):
            for item in group.get("statisticsItems", []):
                out[item["key"]] = (item.get("homeValue"), item.get("awayValue"))
        return out
    return {}


def shotmap(event_id):
    d = get(f"event/{event_id}/shotmap")
    return (d or {}).get("shotmap", []) or []


def cmd_check():
    sid, year = seasons()[0]
    print(f"Sampling most recent season: {year} (id {sid})")
    evs = finished_events(sid)[:15]
    print(f"{len(evs)} finished matches sampled\n")

    have_xg = have_shots = 0
    for e in evs:
        st = team_stats(e["id"])
        sm = shotmap(e["id"])
        xg = st.get("expectedGoals")
        have_xg += xg is not None
        have_shots += len(sm) > 0
        label = f'{e["homeTeam"]["name"]} v {e["awayTeam"]["name"]}'
        print(f'{label:<45} xG={xg}  shots={len(sm)}  keys={len(st)}')

    print(f"\nexpectedGoals present: {have_xg}/{len(evs)}")
    print(f"non-empty shotmap:     {have_shots}/{len(evs)}")
    if have_xg == 0:
        print("\n-> No Sofascore xG at this level. Do not backfill.")
    elif have_shots > 0:
        print("\n-> Shot coordinates available. You can fit your own xG model,")
        print("   which is the version actually worth testing.")


def cmd_backfill():
    rows = []
    for sid, year in seasons()[:3]:
        evs = finished_events(sid)
        print(f"{year}: {len(evs)} matches", file=sys.stderr)
        for i, e in enumerate(evs, 1):
            st = team_stats(e["id"])
            xg = st.get("expectedGoals", (None, None))
            rows.append({
                "event_id": e["id"],
                "season": year,
                "round": e.get("roundInfo", {}).get("round"),
                "kickoff": e["startTimestamp"],
                "home": e["homeTeam"]["name"],
                "away": e["awayTeam"]["name"],
                "home_goals": e["homeScore"]["current"],
                "away_goals": e["awayScore"]["current"],
                "home_xg": xg[0],
                "away_xg": xg[1],
                "home_shots": st.get("totalShotsOnGoal", (None, None))[0],
                "away_shots": st.get("totalShotsOnGoal", (None, None))[1],
                "home_sot": st.get("shotsOnGoal", (None, None))[0],
                "away_sot": st.get("shotsOnGoal", (None, None))[1],
                "home_corners": st.get("cornerKicks", (None, None))[0],
                "away_corners": st.get("cornerKicks", (None, None))[1],
                "home_poss": st.get("ballPossession", (None, None))[0],
                "away_poss": st.get("ballPossession", (None, None))[1],
            })
            if i % 25 == 0:
                print(f"  {i}/{len(evs)}", file=sys.stderr)

    import pandas as pd
    df = pd.DataFrame(rows)
    df["kickoff"] = pd.to_datetime(df["kickoff"], unit="s")
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} — {len(df)} matches, "
          f"{df['home_xg'].notna().mean():.1%} with xG")


def cmd_shots():
    """Dump shot-level rows (xg, xgot, coordinates) for all cached events."""
    rows = []
    for sid, _ in seasons():
        for e in finished_events(sid):
            for s in shotmap(e["id"]):
                pc = s.get("playerCoordinates", {})
                rows.append({
                    "event_id": e["id"],
                    "shot_id": s.get("id"),
                    "is_home": s.get("isHome"),
                    "minute": s.get("time"),
                    "shot_type": s.get("shotType"),
                    "situation": s.get("situation"),
                    "body_part": s.get("bodyPart"),
                    "x": pc.get("x"), "y": pc.get("y"),
                    "xg": s.get("xg"), "xgot": s.get("xgot"),
                })
    import pandas as pd
    pd.DataFrame(rows).to_csv("nln_shots.csv", index=False)
    print(f"wrote nln_shots.csv — {len(rows)} shots")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": cmd_check, "backfill": cmd_backfill, "shots": cmd_shots}[cmd]()
