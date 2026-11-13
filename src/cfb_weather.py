"""Game-time weather flags for upcoming college-football games. FLAG ONLY - does not change model numbers.

Uses the venue coordinates CFBD publishes (/venues, one API call, saved to data/raw/cfb/venues.csv) and the same
free Open-Meteo forecast as the NFL scan (weather.py). Neutral-site games work because the VENUE is used, not
the home team. Dome venues are marked "no weather effect".

NOTE: the exact field names of CFBD's /venues response could not be checked from the build environment, so the
coordinate/dome lookup accepts the common shapes (latitude/longitude, or location.x = lon / location.y = lat).
If a venue has no usable coordinates the note says so and nothing else is affected.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import weather as W

LAT_COLS = ("latitude", "lat", "location_y")
LON_COLS = ("longitude", "lon", "lng", "location_x")


def load_venues(data_dir: Path, key: str | None = None, fetcher=None) -> pd.DataFrame | None:
    path = Path(data_dir) / "venues.csv"
    if path.exists():
        return pd.read_csv(path)
    if not key and fetcher is None:
        return None
    try:
        if fetcher is None:
            import cfb_pull
            fetcher = lambda: cfb_pull.fetch("/venues", {}, key)  # noqa: E731
        df = pd.json_normalize(fetcher(), sep="_")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load venues ({e.__class__.__name__}); CFB weather skipped")
        return None


def _first(row, cols):
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except (TypeError, ValueError):
                continue
    return None


def venue_info(venues: pd.DataFrame, venue_id) -> dict | None:
    if venues is None or pd.isna(venue_id) or "id" not in venues.columns:
        return None
    m = venues[venues["id"] == int(venue_id)]
    if m.empty:
        return None
    r = m.iloc[0]
    dome = bool(r["dome"]) if "dome" in r.index and pd.notna(r["dome"]) else False
    return {"lat": _first(r, LAT_COLS), "lon": _first(r, LON_COLS), "dome": dome}


def note_for(info, kick: datetime, fetcher=W.fetch) -> tuple[str, bool | None]:
    if info is None:
        return "venue unknown - weather not checked", None
    if info["dome"]:
        return "dome - no weather effect", False
    if info["lat"] is None or info["lon"] is None:
        return "venue location unavailable", None
    if kick - datetime.now(timezone.utc) > timedelta(days=15):
        return "forecast not available yet", None
    try:
        w = W.summarize(fetcher(info["lat"], info["lon"]), kick)
    except Exception as e:  # noqa: BLE001 - never break the scan over weather
        return f"weather unavailable ({e.__class__.__name__})", None
    if w is None:
        return "forecast not available yet", None
    return W.describe(w)


def flag_games(res: pd.DataFrame, data_dir: Path, fetcher=W.fetch, venues=None) -> pd.DataFrame:
    """Adds wx_note / wx_flag to a frame with venueId and start (UTC timestamps)."""
    res = res.copy()
    if venues is None:
        venues = load_venues(data_dir, os.environ.get("CFBD_API_KEY", "").strip() or None)
    notes, flags, cache = [], [], {}
    for r in res.itertuples():
        key = (r.venueId, pd.Timestamp(r.start).floor("h"))
        if key not in cache:
            kick = pd.Timestamp(r.start).to_pydatetime()
            cache[key] = note_for(venue_info(venues, r.venueId), kick, fetcher)
        n, f = cache[key]
        notes.append(n)
        flags.append(f)
    res["wx_note"], res["wx_flag"] = notes, flags
    return res
