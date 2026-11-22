"""Game-time weather forecast for upcoming NFL games (free Open-Meteo API, no key). FLAG ONLY - does not change model numbers.

Why a flag and not a model input: on 2,870 historical outdoor games, wind of 10+ mph went under 54-56% of the
time, but a model fit on prior seasons explained none of the total's error out of sample (R2 about 0.000) and the
market already prices most of it. It is still worth seeing before you trust an OVER lean.

wx_note examples:  "wind 18 mph (gusts 27), 41F, rain 60%"   |   "dome/roof - no weather effect"
wx_flag is True when wind >= 15 mph, gusts >= 25 mph, rain chance >= 60%, or temp <= 25F.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

# Home stadium (lat, lon) by nflverse team abbreviation.
COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2575), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SF": (37.4030, -121.9697),
    "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}
# Stadiums with a fixed or retractable roof: treated as no-weather unless the schedule says the roof is open/outdoors.
ROOFED = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "MIN", "NO"}

FLAG_WIND, FLAG_GUST, FLAG_RAIN, FLAG_COLD = 15.0, 25.0, 60.0, 25.0


def kickoff_utc(gameday, gametime) -> datetime | None:
    try:
        h, m = str(gametime).split(":")[:2]
        d = datetime.fromisoformat(str(gameday)).replace(hour=int(h), minute=int(m), tzinfo=ET)
        return d.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def summarize(api: dict, kick: datetime, hours: int = 3) -> dict | None:
    """Average the forecast over the first `hours` hours of the game. api = Open-Meteo JSON (UTC hours)."""
    h = api.get("hourly") or {}
    times = h.get("time") or []
    want = {(kick + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(hours + 1)}
    idx = [i for i, t in enumerate(times) if t in want]
    if not idx:
        return None

    def col(name):
        v = [h[name][i] for i in idx if h.get(name) and h[name][i] is not None]
        return v

    wind, gust, temp, rain = col("wind_speed_10m"), col("wind_gusts_10m"), col("temperature_2m"), col("precipitation_probability")
    if not wind or not temp:
        return None
    return {"wind": sum(wind) / len(wind), "gust": max(gust) if gust else None,
            "temp": sum(temp) / len(temp), "rain": max(rain) if rain else None}


def describe(w: dict) -> tuple[str, bool]:
    bits = [f"wind {w['wind']:.0f} mph" + (f" (gusts {w['gust']:.0f})" if w.get("gust") and w["gust"] >= w["wind"] + 5 else ""),
            f"{w['temp']:.0f}F"]
    if w.get("rain") is not None and w["rain"] >= 30:
        bits.append(f"rain {w['rain']:.0f}%")
    flag = (w["wind"] >= FLAG_WIND or (w.get("gust") or 0) >= FLAG_GUST or (w.get("rain") or 0) >= FLAG_RAIN
            or w["temp"] <= FLAG_COLD)
    return ", ".join(bits), bool(flag)


_CACHE: dict = {}
_LAST_CALL = [0.0]
_LAST_ERR: list = [None]
_FAILS = [0]            # consecutive stadiums that failed after all retries; at 3 we stop hammering the API this run
MIN_GAP = 0.3          # seconds between real requests: the runner shares an IP with other users of the free API


def err_label(e: Exception) -> str:
    """Short reason for the wx_note, with the HTTP code when there is one (e.g. 'HTTP 429')."""
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    return e.__class__.__name__


def fetch(lat: float, lon: float) -> dict:
    """One Open-Meteo request per stadium (cached), spaced out, retried with backoff on 429/5xx/network errors."""
    key = (round(lat, 3), round(lon, 3))
    if key in _CACHE:
        return _CACHE[key]
    if _FAILS[0] >= 3:
        raise _LAST_ERR[0]
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "UTC", "forecast_days": 16,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability"})
    req = urllib.request.Request(f"https://api.open-meteo.com/v1/forecast?{q}", headers={"User-Agent": "nfl-model-weather"})
    last: Exception | None = None
    for attempt in range(5):
        wait = MIN_GAP - (time.time() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.time()
        delay = 3 * (attempt + 1)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                _CACHE[key] = json.loads(r.read().decode())
                _FAILS[0] = 0
                return _CACHE[key]
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise                                   # a real client error: retrying will not help
            last = e
            try:
                delay = min(30, max(delay, int(e.headers.get("Retry-After", 0))))
            except (TypeError, ValueError, AttributeError):
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < 4:
            time.sleep(delay)
    _FAILS[0] += 1
    _LAST_ERR[0] = last
    raise last


def note_for(home: str, roof, gameday, gametime, location=None, fetcher=fetch) -> tuple[str, bool | None]:
    """(note, flag). flag None = unknown/not applicable."""
    roof = roof if isinstance(roof, str) else ""
    if roof in ("dome", "closed") or (not roof and home in ROOFED):
        return "dome/roof - no weather effect", False
    if location == "Neutral" or home not in COORDS:
        return "neutral site - weather not checked", None
    kick = kickoff_utc(gameday, gametime)
    if kick is None:
        return "", None
    if kick - datetime.now(timezone.utc) > timedelta(days=15):
        return "forecast not available yet", None
    try:
        w = summarize(fetcher(*COORDS[home]), kick)
    except Exception as e:  # noqa: BLE001 - never break the scan over weather
        return f"weather unavailable ({err_label(e)})", None
    if w is None:
        return "forecast not available yet", None
    text, flag = describe(w)
    if roof == "open":
        text += " (roof may be open)"
    return text, flag


def flag_games(pred: pd.DataFrame, games: pd.DataFrame | None = None, fetcher=fetch) -> pd.DataFrame:
    """Adds wx_note and wx_flag to a frame with home_team/gameday/gametime (and roof if present)."""
    pred = pred.copy()
    notes, flags = [], []
    cache: dict = {}
    for r in pred.itertuples():
        key = (r.home_team, str(r.gameday), str(getattr(r, "gametime", "")))
        if key not in cache:
            cache[key] = note_for(r.home_team, getattr(r, "roof", None), r.gameday, getattr(r, "gametime", None),
                                  getattr(r, "location", None), fetcher)
        n, f = cache[key]
        notes.append(n)
        flags.append(f)
    pred["wx_note"], pred["wx_flag"] = notes, flags
    return pred
