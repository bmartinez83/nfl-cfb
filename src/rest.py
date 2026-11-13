"""Rest-days context for NFL and college-football games. FLAG ONLY - does not change model numbers.

Rest = calendar days since each team's previous game. Notable when a team is on a short week (<= 5 days, e.g. after
a Saturday game playing Thursday), when one side has 3+ more days than the other, or when a team is coming off a bye
(10+ days). Why a flag and not a model input: on 3,965 NFL games (2012+) and 8,310 college games with lines, rest
difference had ~0 correlation with the market error (0.01 / 0.02) and home-cover rates stayed 47-52%; the college
short-week samples are tiny (14-27 games). The market already prices it. It is still worth seeing before a lean.

rest_note examples:  "rest: MEM 4d (short week) vs USF 9d"   |   ""  (nothing notable)
"""
from __future__ import annotations

import pandas as pd

SHORT, GAP, BYE, OPENER = 5, 3, 10, 21


def _fmt(team: str, d) -> str:
    return f"{team} {int(d)}d"


def note_for(home: str, away: str, home_rest, away_rest) -> tuple[str, bool | None]:
    """(note, flag). Blank/None when rest is unknown or nothing is notable."""
    ok = lambda x: x is not None and pd.notna(x) and 0 < float(x) <= OPENER  # noqa: E731
    if not (ok(home_rest) and ok(away_rest)):
        return "", None
    h, a = float(home_rest), float(away_rest)
    tags = []
    for t, d in ((away, a), (home, h)):
        if d <= SHORT:
            tags.append(f"{t} short week")
        elif d >= BYE:
            tags.append(f"{t} off bye")
    if abs(h - a) >= GAP and not tags:
        tags.append(f"{home if h > a else away} +{int(abs(h - a))}d rest")
    if not tags:
        return "", False
    return f"rest: {_fmt(away, a)} @ {_fmt(home, h)} ({', '.join(tags)})", True


def nfl_flag_games(pred: pd.DataFrame) -> pd.DataFrame:
    """Needs home_team/away_team and home_rest/away_rest (nflverse games.csv columns)."""
    pred = pred.copy()
    res = [note_for(r.home_team, r.away_team, getattr(r, "home_rest", None), getattr(r, "away_rest", None))
           for r in pred.itertuples()]
    pred["rest_note"] = [n for n, _ in res]
    pred["rest_flag"] = [f for _, f in res]
    return pred


def cfb_flag_games(res: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """res needs id, homeTeam, awayTeam, start (UTC). games = all CFBD games (any season) with id/startDate/teams."""
    res = res.copy()
    g = games.copy()
    g["_t"] = pd.to_datetime(g["startDate"], utc=True)
    long = pd.concat([g[["id", "_t", "homeTeam"]].rename(columns={"homeTeam": "team"}),
                      g[["id", "_t", "awayTeam"]].rename(columns={"awayTeam": "team"})]).dropna()
    by_team = {t: s["_t"].sort_values().tolist() for t, s in long.groupby("team")}

    def days_since(team, start):
        prev = [x for x in by_team.get(team, []) if x < start - pd.Timedelta(hours=12)]
        if not prev:
            return None
        return (start.tz_convert("America/Los_Angeles").normalize().tz_localize(None)
                - prev[-1].tz_convert("America/Los_Angeles").normalize().tz_localize(None)).days

    notes, flags = [], []
    for r in res.itertuples():
        st = pd.Timestamp(r.start)
        n, f = note_for(r.homeTeam, r.awayTeam, days_since(r.homeTeam, st), days_since(r.awayTeam, st))
        notes.append(n)
        flags.append(f)
    res["rest_note"], res["rest_flag"] = notes, flags
    return res
