"""Down-and-distance and pass-rush matchups as a small, capped margin adjustment (play-by-play from nflverse).

For each upcoming game, from only PRIOR games (this season + a few prior-season games, shrunk to the league mean):
  d4go  home minus away 4th-down go-for-it rate
  d4c   home minus away (own 4th-down conversion + opponent's 4th-down conversion allowed)
  dprs  home minus away (own sacks+QB hits allowed per dropback + opponent's sacks+QB hits generated)
dsack = pocket stress: (QB time to throw above league avg) x (opposing pass rush above league avg), home minus away.
Its backtest fit was ~0, so its weight is a small hand-set prior (a 1-sd gap is ~0.3 pt) chosen by the user, not a measured edge.
Coefficients are sign-constrained fits to (actual margin - closing spread) over 2022-2025 (tools/situational_backtest.py).
3rd-down matchup and time-to-throw fit to zero; totals showed no relationship, so only the margin is adjusted.
Out of sample the signal was weak (corr ~0.03-0.04), so the adjustment is capped at +/-CAP points. Never breaks a scan."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
COEF = {"d4go": 1.77, "d4c": 3.11, "dprs": -4.80, "dsack": -40.0}      # points per unit of the home-minus-away difference
CAP = 1.0
K, PRIOR_W = 8, 4
PBP_COLS = ["game_id", "season_type", "week", "home_team", "away_team", "posteam", "defteam", "play_type", "down",
            "first_down", "qb_kneel", "qb_spike", "pass_attempt", "sack", "qb_hit", "play_id"]


def _reduce(p: pd.DataFrame, season: int) -> pd.DataFrame:
    p = p[p["season_type"] == "REG"]
    pl = p[p.play_type.isin(["pass", "run"]) & (p.qb_kneel != 1) & (p.qb_spike != 1)]
    f4 = p[(p.down == 4) & p.play_type.isin(["pass", "run", "punt", "field_goal"])].copy()
    f4["go"] = f4.play_type.isin(["pass", "run"]).astype(int)
    k = ["game_id", "posteam"]
    a4 = f4.groupby(k).agg(f4_n=("go", "size"), f4_go=("go", "sum")).reset_index()
    c4 = f4[f4.go == 1].groupby(k).agg(f4_c=("first_down", "sum")).reset_index()
    db = p[(p.pass_attempt == 1) | (p.sack == 1)]
    pr = db.groupby(k).agg(db_n=("play_id", "size"), sh=("sack", "sum"), hits=("qb_hit", "sum")).reset_index()
    m = a4.merge(c4, on=k, how="outer").merge(pr, on=k, how="outer").fillna(0)
    m["sh"] = m["sh"] + m["hits"]
    g = p.groupby("game_id").agg(week=("week", "first"), home=("home_team", "first"), away=("away_team", "first")).reset_index()
    m = m.merge(g, on="game_id").rename(columns={"posteam": "team"})
    m["opp"] = np.where(m.team == m.home, m.away, m.home)
    m["season"] = season
    return m[["game_id", "season", "week", "team", "opp", "f4_n", "f4_go", "f4_c", "db_n", "sh"]]


def team_games(seasons, cache_dir: Path) -> pd.DataFrame:
    import ingest
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in seasons:
        cf = cache_dir / f"situational_{s}.csv"
        try:
            raw = ingest.fetch(f"{BASE}/pbp/play_by_play_{s}.csv.gz")
            p = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in PBP_COLS, low_memory=False)
            t = _reduce(p, s)
            t.to_csv(cf, index=False)
        except Exception as e:  # noqa: BLE001
            if not cf.exists():
                print(f"WARNING: play-by-play {s} unavailable ({e.__class__.__name__})")
                continue
            print(f"WARNING: using cached situational data for {s}")
            t = pd.read_csv(cf)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ttt_table(seasons) -> pd.DataFrame:
    """NGS average time to throw per team-week (attempt-weighted over that team's QBs)."""
    import ingest
    raw = ingest.fetch(f"{BASE}/nextgen_stats/ngs_passing.csv.gz")
    n = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False)
    n = n[(n.week > 0) & n.season.isin(list(seasons))]
    n = n.assign(w=n.avg_time_to_throw * n.attempts)
    g = n.groupby(["season", "week", "team_abbr"]).agg(w=("w", "sum"), a=("attempts", "sum")).reset_index()
    g["ttt"] = g.w / g.a
    return g.rename(columns={"team_abbr": "team"})[["season", "week", "team", "ttt"]]


def _rates(tg: pd.DataFrame, num: str, den: str, key: str, season: int, week: int, team: str, lg: float, avg_den: float):
    d = tg[tg[key] == team]
    cur, prev = d[(d.season == season) & (d.week < week)], d[d.season == season - 1]
    sc = min(PRIOR_W, len(prev)) / max(len(prev), 1)
    n = cur[num].sum() + prev[num].sum() * sc
    t = cur[den].sum() + prev[den].sum() * sc
    return (n + lg * K * avg_den) / (t + K * avg_den)


def adjust(pred: pd.DataFrame, tg: pd.DataFrame, ttt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds sit_adj (home-minus-away points) and sit_note. pred needs season, week, home_team, away_team."""
    pred = pred.copy()
    pred["sit_adj"], pred["sit_note"] = 0.0, ""
    if tg.empty:
        pred["sit_note"] = "situational data unavailable - no adjustment"
        return pred
    L = {n: tg[n].sum() / max(tg[d].sum(), 1) for n, d in (("f4_go", "f4_n"), ("f4_c", "f4_go"), ("sh", "db_n"))}
    A = {d: tg[d].mean() for d in ("f4_n", "f4_go", "db_n")}
    adj, notes = [], []
    lg_ttt = float(ttt.ttt.mean()) if ttt is not None and len(ttt) else np.nan

    def T(season, week, team):
        if ttt is None or not len(ttt):
            return np.nan
        d = ttt[(ttt.team == team) & (((ttt.season == season) & (ttt.week < week)) | (ttt.season == season - 1))]
        return float(d.ttt.mean()) if len(d) else np.nan
    for r in pred.itertuples():
        s, w, h, a = int(r.season), int(r.week), r.home_team, r.away_team
        R = lambda num, den, key, t: _rates(tg, num, den, key, s, w, t, L[num], A[den])  # noqa: E731
        go = {t: R("f4_go", "f4_n", "team", t) for t in (h, a)}
        c4 = {t: R("f4_c", "f4_go", "team", t) + R("f4_c", "f4_go", "opp", o) for t, o in ((h, a), (a, h))}
        pr = {t: R("sh", "db_n", "team", t) + R("sh", "db_n", "opp", o) for t, o in ((h, a), (a, h))}
        lgp = 2 * L["sh"]
        risk = {t: (T(s, w, t) - lg_ttt) * (pr[t] - lgp) for t in (h, a)}
        dsack = risk[h] - risk[a]
        diff = {"d4go": go[h] - go[a], "d4c": c4[h] - c4[a], "dprs": pr[h] - pr[a], "dsack": 0.0 if np.isnan(dsack) else dsack}
        raw = sum(COEF[k] * v for k, v in diff.items())
        x = float(np.clip(raw, -CAP, CAP))
        adj.append(x)
        hot = [f"{t} QB {T(s, w, t):.2f}s to throw vs strong rush" for t in (h, a) if risk[t] > 0.003]
        notes.append((f"situational: {'home' if x > 0 else 'away'} {abs(x):.1f} pts (4th-down, pass rush, pocket stress)"
                      + (" | " + "; ".join(hot) if hot else "")) if abs(x) >= 0.05 or hot else "")
    pred["sit_adj"], pred["sit_note"] = adj, notes
    return pred
