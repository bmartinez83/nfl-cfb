"""Pull college-football data from CollegeFootballData (CFBD) into data/raw/cfb/.

Run on YOUR machine (the key is read from the environment and never printed
or written to disk):

    export CFBD_API_KEY='...'
    python src/cfb_pull.py                 # seasons 2014-2026
    python src/cfb_pull.py --start 2019    # fewer seasons = fewer API calls
    python src/cfb_pull.py --dry-run       # list the calls, make none

Endpoints (one call each per season, so roughly 7 calls/season):
  games              /games                 results + teams + dates
  lines              /lines                 one row per game x sportsbook
  adv_game_stats     /stats/game/advanced   per-game team advanced stats (PPA, success rate...)
  season_stats       /stats/season          team season totals
  sp / elo / fpi     /ratings/sp|elo|fpi    public ratings (use PRIOR season only as priors;
                                            same-season values are end-of-season = leakage)
  fbs_teams          /teams/fbs             FBS membership per season

The CFBD terms page does not state the free-tier call quota, so this script is
polite (small sleep, stops on 429) and can resume: existing per-season files
are skipped unless --refresh is passed.

IMPORTANT first deliverable: data/raw/cfb/_schema.json lists the columns each
endpoint actually returned. In particular, check the lines columns for any
opening-line fields — that decides whether we can measure CLV on CFB or only a
single-snapshot residual test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://api.collegefootballdata.com"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cfb"
SLEEP = 0.4

# name -> (path, params-builder(year))
ENDPOINTS = {
    "games": ("/games", lambda y: {"year": y, "seasonType": "regular"}),
    "lines": ("/lines", lambda y: {"year": y, "seasonType": "regular"}),
    "adv_game_stats": ("/stats/game/advanced", lambda y: {"year": y, "seasonType": "regular"}),
    "season_stats": ("/stats/season", lambda y: {"year": y}),
    "sp": ("/ratings/sp", lambda y: {"year": y}),
    "elo": ("/ratings/elo", lambda y: {"year": y}),
    "fpi": ("/ratings/fpi", lambda y: {"year": y}),
    "fbs_teams": ("/teams/fbs", lambda y: {"year": y}),
}


class QuotaOrAuth(Exception):
    pass


def fetch(path: str, params: dict, key: str):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
        "User-Agent": "cfb-model-pull"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise QuotaOrAuth(f"HTTP {e.code} on {path}: key rejected or not permitted") from e
        if e.code == 429:
            raise QuotaOrAuth(f"HTTP 429 on {path}: quota/rate limit hit - stop and retry later") from e
        raise


def flatten_lines(games: list[dict]) -> pd.DataFrame:
    """One row per (game, sportsbook). Keeps every field the API returns for a
    line, so opening-line columns (if any) survive without code changes."""
    rows = []
    for g in games:
        head = {k: v for k, v in g.items() if k != "lines" and not isinstance(v, (list, dict))}
        for ln in g.get("lines") or []:
            rows.append({**head, **{f"line_{k}": v for k, v in ln.items()
                                    if not isinstance(v, (list, dict))}})
    return pd.DataFrame(rows)


def to_frame(name: str, data) -> pd.DataFrame:
    if name == "lines":
        return flatten_lines(data)
    if isinstance(data, dict):
        data = data.get("items") or [data]
    # json_normalize flattens nested stat dicts (e.g. offense.ppa -> offense_ppa)
    return pd.json_normalize(data, sep="_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true", help="re-pull files that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key and not args.dry_run:
        print("Set CFBD_API_KEY in your environment first (export CFBD_API_KEY='...').")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    schema_path = OUT / "_schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    calls = 0
    try:
        for year in range(args.start, args.end + 1):
            for name, (path, params_fn) in ENDPOINTS.items():
                dest = OUT / f"{name}_{year}.csv"
                if dest.exists() and not args.refresh:
                    continue
                if args.dry_run:
                    print(f"would call {path} {params_fn(year)} -> {dest.name}")
                    continue
                try:
                    data = fetch(path, params_fn(year), key)
                except urllib.error.HTTPError as e:
                    print(f"{name} {year}: HTTP {e.code} (skipped)")
                    continue
                except urllib.error.URLError as e:
                    print(f"{name} {year}: network error {e.reason} (skipped)")
                    continue
                calls += 1
                df = to_frame(name, data)
                df.to_csv(dest, index=False)
                schema[name] = sorted(set(schema.get(name, [])) | set(df.columns))
                print(f"{name} {year}: {len(df)} rows -> {dest.name}")
                time.sleep(SLEEP)
        # venue list (stadium coordinates + dome flag) for the weather flags: one call, not per season
        vdest = OUT / "venues.csv"
        if args.dry_run:
            print(f"would call /venues -> {vdest.name}")
        elif args.refresh or not vdest.exists():
            try:
                vdata = fetch("/venues", {}, key)
                vdf = to_frame("venues", vdata)
                vdf.to_csv(vdest, index=False)
                schema["venues"] = sorted(set(schema.get("venues", [])) | set(vdf.columns))
                calls += 1
                print(f"venues: {len(vdf)} rows -> {vdest.name}; columns: {list(vdf.columns)[:14]}")
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"venues: skipped ({e.__class__.__name__}); weather flags will say 'venue unknown'")
    except QuotaOrAuth as e:
        print(f"\nSTOPPED: {e}")
        print("Files already written are kept; re-run later to resume.")
    finally:
        schema_path.write_text(json.dumps(schema, indent=1))
    print(f"\n{calls} API calls made. Column lists: {schema_path}")
    print("Next: send back _schema.json (small) and, if you're comfortable, the CSVs "
          "or a zip of data/raw/cfb/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
