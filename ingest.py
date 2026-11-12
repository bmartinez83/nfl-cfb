"""Ingest NFL data from nflverse into data/raw/.

Primary source: nflverse-data GitHub release files (the same files
nfl_data_py wraps). nfl_data_py itself was tried first (2026-10-29):
`pip install nfl_data_py` fails in this environment because it pins an
old pandas that must build from source, so this module downloads the
release CSVs directly:

  * schedules/games.csv          -> data/raw/games.csv
      One row per game, 1999-present. Carries results (scores) AND the
      market: closing spread_line / total_line / moneylines, plus QB
      starter names, rest days, roof/surface. This is the backtest market.
  * stats_team/stats_team_week_<season>.csv -> data/raw/team_week_<season>.csv
      Team-level weekly stats (yards, plays, turnovers, first downs...).

Seasons: 2019 through the current season (2026), as available.
Play-by-play is deliberately NOT pulled (multi-GB parquet; points/yards
features suffice for v1 — noted in the project README).

Raw files are saved untouched. Run:  python src/ingest.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_SEASON = 2019
LAST_SEASON = 2026  # current season; files for it update weekly

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model-ingest"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True

    # --- schedules / games -------------------------------------------------
    url = f"{BASE}/schedules/games.csv"
    try:
        data = fetch(url)
        (RAW / "games.csv").write_bytes(data)
        games = pd.read_csv(io.BytesIO(data))
        reg = games[games["game_type"] == "REG"]
        print(f"games.csv: {len(games)} rows total, {len(reg)} regular-season")
        for season, g in reg.groupby("season"):
            if season >= FIRST_SEASON:
                played = g["home_score"].notna().sum()
                print(f"  season {season}: {len(g)} games, {played} with final scores")
    except Exception as exc:  # pragma: no cover - network failure path
        print(f"FAILED to download schedules: {exc}")
        ok = False

    # --- team weekly stats --------------------------------------------------
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        url = f"{BASE}/stats_team/stats_team_week_{season}.csv"
        try:
            data = fetch(url)
            (RAW / f"team_week_{season}.csv").write_bytes(data)
            df = pd.read_csv(io.BytesIO(data))
            weeks = sorted(df["week"].dropna().unique().tolist()) if "week" in df else []
            print(f"team_week_{season}.csv: {len(df)} rows, weeks {weeks[:3]}...{weeks[-2:] if weeks else []}")
        except Exception as exc:
            print(f"team_week_{season}.csv: not available ({exc.__class__.__name__})")

    if not ok:
        print("INGEST INCOMPLETE: schedules file is required. Stopping.")
        return 1
    print(f"Raw files in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
