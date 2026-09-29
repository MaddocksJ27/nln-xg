"""
Sofascore -> National League North match xG and shot-level data.

Tournament id 176 = National League North (filed under "england-amateur").

IMPERSONATION NOTE (Sep 2026): Sofascore cross-checks the User-Agent against
the TLS handshake. curl_cffi 0.13.0's newest Chrome fingerprint is chrome136,
which is now too old and gets 403 on every request. Safari targets still pass.
Do NOT set a User-Agent header by hand — curl_cffi supplies one that matches
its handshake, and a mismatch is an obvious bot signature (a Chrome UA on a
Safari handshake gets 403 even though the same request without it succeeds).

If safari260 starts failing, safari184, safari180 and safari_ios also worked.

    python sofascore_nln.py check [season_index]   # coverage check
    python sofascore_nln.py backfill               # match xG + team stats
    python sofascore_nln.py shots                  # shot-level data

Everything is cached to ./cache as raw JSON, so re-runs cost nothing and an
interrupted run loses at most one request.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

from curl_cffi.requests import Session as CurlSession

TOURNAMENT_ID = 176          # National League North
BASE = "https://www.sofascore.com/api/v1"   # api.sofascore.com 403s
CACHE = Path("cache")
OUT = Path("nln_xg.csv")
SEASON_LIMIT = 3             # xG coverage only exists for the newest 3 seasons

SESSION = CurlSession(impersonate="safari260")
SESSION.headers.update({
    "Accept": "*/*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
})


def get(path, ttl_days=None):
    """GET with on-disk cache, jittered rate limit, and 429/403 backoff.

    ttl_days=None  -> cache forever (finished-match data never changes)
    ttl_days=0     -> always refetch (fixture lists, which do change)
    """
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
            if attempt == 0 and r.status_code == 403:
                print("  (403 on the first try usually means the impersonation "
                      "target is stale — see the note at the top of this file)",
                      file=sys.stderr)
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
        d = get(f"unique-tournament/{TOURNAMENT_ID}/season/{season_id}"
                f"/events/last/{page}")
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
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sid, year = seasons()[idx]
    print(f"Sampling season: {year} (id {sid})")
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
        print("\n-> No Sofascore xG for this season. Do not backfill it.")


def cmd_backfill():
    rows = []
    for sid, year in seasons()[:SEASON_LIMIT]:
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
    """Shot-level rows: xg, xgot, coordinates, situation, body part."""
    rows = []
    for sid, year in seasons()[:SEASON_LIMIT]:
        evs = finished_events(sid)
        print(f"{year}: {len(evs)} matches", file=sys.stderr)
        for i, e in enumerate(evs, 1):
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
            if i % 50 == 0:
                print(f"  {i}/{len(evs)}  ({len(rows)} shots)", file=sys.stderr)

    import pandas as pd
    pd.DataFrame(rows).to_csv("nln_shots.csv", index=False)
    print(f"\nwrote nln_shots.csv — {len(rows)} shots")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": cmd_check, "backfill": cmd_backfill, "shots": cmd_shots}[cmd]()
