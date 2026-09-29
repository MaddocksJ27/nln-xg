"""
Find a working Sofascore configuration.

curl_cffi 0.13.0 only ships Chrome fingerprints up to chrome136, which is
now badly stale — the bare "chrome" alias points there. Its Safari targets
(safari260, safari184) correspond to much more recent browsers and may pass
where chrome136 does not.

This tries each target, with and without full browser headers, against the
real endpoint. One request per combination, 3s apart.

    python probe.py
"""

import time

from curl_cffi.requests import Session

URL = "https://www.sofascore.com/api/v1/unique-tournament/176/seasons"

TARGETS = ["chrome", "chrome136", "chrome133a", "chrome131",
           "safari", "safari260", "safari184", "safari180",
           "edge", "chrome_android", "safari_ios"]

MINIMAL = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/136.0.0.0 Safari/537.36"),
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json",
}

# what a real browser actually sends to this endpoint
FULL = {
    "Accept": "*/*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

winners = []

for target in TARGETS:
    for label, hdrs in (("full", FULL), ("minimal", MINIMAL), ("none", {})):
        try:
            s = Session(impersonate=target)
            if hdrs:
                s.headers.update(hdrs)
            r = s.get(URL, timeout=15)
            code = r.status_code
            ok = code == 200 and "seasons" in r.text[:400]
            mark = "  <-- WORKS" if ok else ""
            print(f"{target:<16} headers={label:<8} {code}{mark}")
            if ok:
                winners.append((target, label))
        except Exception as e:
            print(f"{target:<16} headers={label:<8} ERROR {type(e).__name__}")
        time.sleep(3)

print()
if winners:
    t, h = winners[0]
    print(f"USE: impersonate={t!r}, headers={h}")
    print(f"all working: {winners}")
else:
    print("Nothing worked. Either the IP is blocked despite the browser")
    print("loading (different path to the API than to the page), or every")
    print("fingerprint this curl_cffi version has is too old.")
    print("Next step: upgrade curl_cffi for newer targets —")
    print("  pip install 'curl_cffi==0.15.0'  (then 0.14, 0.16 if it fails)")
    print("and re-run. Check `import curl_cffi` works after each install;")
    print("0.16.3 had a macOS CoreFoundation linking bug.")
