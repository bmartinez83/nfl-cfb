"""Snapshot Novig's NFL prices (unsigned public catalog - NO API key needed) into your own line history.

  python src/novig_snapshot.py [--hours 200] [--out DIR] [--data DIR]

Each run appends to outputs/:
  novig_nfl_quotes.csv  one row per outcome that has a bid (raw ladder: every spread/total strike)
  novig_nfl_games.csv   one row per game: Novig-implied home win prob, home spread and total
                        (where the 50/50 point of the strike ladder sits), plus nflverse's line.

Reading the numbers
  * An outcome's 'bid' is the best price someone will pay for it (0.55 = 55c = 55% implied).
    For a two-outcome market, mid(A) = (bid_A + 1 - bid_B) / 2; it is only computed when both sides bid.
  * Novig lists spreads/totals as a LADDER of strikes (e.g. WSH +10.5, +11.5, ...). The implied line is
    the strike where the home-cover (or over) probability crosses 50%, by linear interpolation.
  * home_spread_line follows nflverse: POSITIVE = home favored (home margin the market expects).
Novig exchange prices are only a benchmark: a gap to the nflverse line is a reason to look, not a bet.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odds import CIRCA_NAME_TO_ABBR  # noqa: E402

HOST = "https://api.novig.com"
ROOT = Path(__file__).resolve().parent.parent

# Novig team tokens that differ from nflverse abbreviations (extend if a game prints 'unmapped').
ALIASES = {"WAS": {"WSH"}, "LV": {"LVR"}, "LA": {"LAR"}, "JAX": {"JAC"}, "NO": {"NOR"}, "NE": {"NWE"},
           "SF": {"SFO"}, "TB": {"TAM"}, "GB": {"GNB"}, "KC": {"KAN"}}

QUOTE_COLS = ["ts_utc", "event_id", "event", "starts_ts", "market_type", "strike", "market_desc",
              "outcome", "bid_price", "bid_qty"]
GAME_COLS = ["ts_utc", "event_id", "event", "starts_ts", "away", "home", "ml_home_prob",
             "home_spread_line", "total_line", "n_spread_pts", "n_total_pts",
             "nflverse_spread_line", "nflverse_total_line", "gap_spread", "gap_total"]


def get(path: str, query: str = "", tries: int = 3):
    url = f"{HOST}{path}" + (f"?{query}" if query else "")
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-model-snapshot", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(2 * (i + 1))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + i)
    raise RuntimeError(f"GET {url} failed: {last}")


def team_abbr(full: str) -> str | None:
    nick = re.sub(r"[^A-Za-z0-9 ]", "", full).split()[-1].upper() if full.strip() else ""
    return CIRCA_NAME_TO_ABBR.get(nick)


def tokens(abbr: str) -> set[str]:
    return {abbr} | ALIASES.get(abbr, set())


def _bid(o) -> float | None:
    b = o.get("bid") or {}
    return float(b["price"]) if b.get("price") is not None else None


def mid(o, other) -> float | None:
    a, b = _bid(o), _bid(other)
    if a is None or b is None:
        return None
    return (a + 1.0 - b) / 2.0


def crossing(points: list[tuple[float, float]], increasing: bool) -> float | None:
    """x where p crosses 0.5, by linear interpolation. points = [(x, p)], needs a bracketing pair."""
    pts = sorted(points)
    for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p0 != p1:
            return x0 + (0.5 - p0) * (x1 - x0) / (p1 - p0)
    return None


def summarize(markets: list[dict], home_abbr: str) -> dict:
    """Novig-implied home win prob / home spread line (nflverse sign) / total from the market ladders."""
    home_tok = tokens(home_abbr)
    out = {"ml_home_prob": None, "home_spread_line": None, "total_line": None,
           "n_spread_pts": 0, "n_total_pts": 0}
    sp_pts, tot_pts = [], []
    for m in markets:
        outs = m.get("outcomes", [])
        if len(outs) != 2 or m.get("status") not in (None, "OPEN"):
            continue
        t = m.get("marketType")
        if t == "MONEY":
            for o, other in (outs, outs[::-1]):
                if o["name"].split()[0].upper() in home_tok:
                    out["ml_home_prob"] = mid(o, other)
        elif t == "SPREAD":
            for o, other in (outs, outs[::-1]):
                mt = re.match(r"^(\S+)\s+([+-]?\d+(?:\.\d+)?)$", o["name"].strip())
                if mt and mt.group(1).upper() in home_tok:
                    p = mid(o, other)
                    if p is not None:
                        # "WSH +11.5": home gets 11.5 -> home covers if margin > -11.5 -> line = -11.5
                        sp_pts.append((-float(mt.group(2)), p))
        elif t == "TOTAL":
            for o, other in (outs, outs[::-1]):
                mt = re.match(r"^Over\s+(\d+(?:\.\d+)?)$", o["name"].strip())
                if mt:
                    p = mid(o, other)
                    if p is not None:
                        tot_pts.append((float(mt.group(1)), p))
    out["n_spread_pts"], out["n_total_pts"] = len(sp_pts), len(tot_pts)
    out["home_spread_line"] = crossing(sp_pts, True)
    out["total_line"] = crossing(tot_pts, False)
    return out


def _append(path: Path, cols: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=200, help="look-ahead window for games")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--quotes", choices=["all", "none"], default="all",
                    help="'none' skips the big full-ladder file (use for frequent scheduled runs)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    nfl = {}
    gp = Path(args.data) / "games.csv"
    if gp.exists():
        import pandas as pd
        g = pd.read_csv(gp, usecols=["gameday", "away_team", "home_team", "spread_line", "total_line", "home_score"])
        g = g[g["home_score"].isna()]
        nfl = {(r.gameday, r.away_team, r.home_team): (r.spread_line, r.total_line) for r in g.itertuples()}

    ev = get("/v3/public/catalog/events",
             f"league=NFL&status=OPEN_PREGAME&startsAfter={now_ms}&startsBefore={now_ms + args.hours * 3600000}&limit=500")
    events = ev.get("items", [])
    quotes, games = [], []
    for e in events:
        m = get("/v3/public/catalog/markets", f"event={e['eventId']}&limit=2000").get("items", [])
        for mk in m:
            for o in mk.get("outcomes", []):
                p = _bid(o)
                if p is not None:
                    quotes.append({"ts_utc": ts, "event_id": e["eventId"], "event": e["description"],
                                   "starts_ts": e["startsTs"], "market_type": mk.get("marketType"),
                                   "strike": mk.get("strike"), "market_desc": mk.get("description"),
                                   "outcome": o["name"], "bid_price": p, "bid_qty": o["bid"].get("qty")})
        if "@" not in e["description"]:      # futures such as "To Make The Playoffs" - not a game
            continue
        parts = [s.strip() for s in e["description"].split("@")]
        away = team_abbr(parts[0]) if len(parts) == 2 else None
        home = team_abbr(parts[1]) if len(parts) == 2 else None
        if not away or not home:
            print(f"unmapped teams for '{e['description']}' - extend CIRCA_NAME_TO_ABBR/ALIASES")
            continue
        s = summarize(m, home)
        day = datetime.fromtimestamp(e["startsTs"] / 1000, timezone.utc).astimezone(
            __import__("zoneinfo").ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        ns, nt = nfl.get((day, away, home), (None, None))
        games.append({"ts_utc": ts, "event_id": e["eventId"], "event": e["description"],
                      "starts_ts": e["startsTs"], "away": away, "home": home, **s,
                      "nflverse_spread_line": ns, "nflverse_total_line": nt,
                      "gap_spread": (s["home_spread_line"] - ns) if s["home_spread_line"] is not None and ns is not None else None,
                      "gap_total": (s["total_line"] - nt) if s["total_line"] is not None and nt is not None else None})
    if args.quotes == "all":
        _append(out / "novig_nfl_quotes.csv", QUOTE_COLS, quotes)
    _append(out / "novig_nfl_games.csv", GAME_COLS, games)

    print(f"{len(events)} Novig NFL events (incl. futures), {len(quotes)} quotes"
          + (f" -> {out / 'novig_nfl_quotes.csv'}" if args.quotes == "all" else " (ladder file skipped)"))
    print(f"{len(games)} game rows -> {out / 'novig_nfl_games.csv'}\n")
    for r in sorted(games, key=lambda r: r["starts_ts"]):
        f = lambda v, d=1: "  n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"{r['away']:>3} @ {r['home']:<3}  ML home {f(r['ml_home_prob'] and r['ml_home_prob'] * 100, 0)}%  "
              f"Novig spread {f(r['home_spread_line'])} vs nflverse {f(r['nflverse_spread_line'])}  "
              f"Novig total {f(r['total_line'])} vs nflverse {f(r['nflverse_total_line'])}  "
              f"(pts {r['n_spread_pts']}/{r['n_total_pts']})")
    print("\nspread = HOME margin the market expects (positive = home favored), same sign as nflverse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
