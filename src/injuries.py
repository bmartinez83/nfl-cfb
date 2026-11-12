"""Injury / depth-chart check for upcoming NFL games (flags only - does NOT change model numbers).

Sources (nflverse release files, free, GitHub-hosted):
  injuries_<season>.csv       weekly team injury reports (report_status Out / Doubtful / Questionable)
  depth_charts_<season>.csv   depth charts, many snapshots; the LATEST snapshot is used

A "starter" = depth-chart rank 1 at an offense/defense slot (kickers, punters, returners, long snappers,
holders excluded). For each team we report the QB1 status, the number of starters Out/Doubtful, and a
readable list. `hold` = True when the QB1 is Out/Doubtful/Questionable, or 3+ starters are Out/Doubtful -
meaning the model's team ratings (built from past games) may be stale and the lean should be checked by hand.

Limits: nflverse refreshes these files on its own schedule (not minute-by-minute), so breaking news can
lag. Treat this as a safety flag, not a news feed.
"""
from __future__ import annotations

import io
import urllib.request
from functools import lru_cache

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SPECIAL = {"PK", "K", "P", "H", "PR", "KR", "LS"}
BAD = {"Out": 2, "Doubtful": 2, "Questionable": 1}


def _fetch_csv(url: str, usecols=None) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model-injuries"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return pd.read_csv(io.BytesIO(r.read()), usecols=usecols, low_memory=False)


@lru_cache(maxsize=4)
def load(season: int):
    inj = _fetch_csv(f"{BASE}/injuries/injuries_{season}.csv",
                     usecols=["team", "week", "gsis_id", "position", "full_name", "report_status",
                              "report_primary_injury", "season_type"])
    dc = _fetch_csv(f"{BASE}/depth_charts/depth_charts_{season}.csv",
                    usecols=["dt", "team", "player_name", "gsis_id", "pos_abb", "pos_rank"])
    return inj, dc


def summarize(inj: pd.DataFrame, dc: pd.DataFrame, week: int, teams) -> dict:
    """Per-team dict: qb, qb_status, n_out, hold, text. Pure function (no network) so it can be tested."""
    last = dc[dc["dt"] == dc["dt"].max()]
    starters = last[(last["pos_rank"] == 1) & (~last["pos_abb"].isin(SPECIAL))]
    w = inj[(inj["week"] == week) & inj["report_status"].isin(BAD)]
    out = {}
    for t in teams:
        st = starters[starters["team"] == t]
        ww = w[w["team"] == t]
        hit = st.merge(ww[["gsis_id", "report_status", "report_primary_injury"]], on="gsis_id", how="inner")
        qb = st[st["pos_abb"] == "QB"]
        qb_name = qb["player_name"].iloc[0] if len(qb) else None
        qb_status = None
        if qb_name is not None:
            m = hit[hit["pos_abb"] == "QB"]
            qb_status = m["report_status"].iloc[0] if len(m) else None
        out_like = hit[hit["report_status"].isin(["Out", "Doubtful"])]
        parts = [f"{r.pos_abb} {r.player_name} ({r.report_status})"
                 for r in hit.sort_values("report_status", key=lambda s: s.map(BAD), ascending=False).itertuples()]
        out[t] = {"qb": qb_name, "qb_status": qb_status, "n_out": len(out_like),
                  "hold": bool(qb_status in BAD or len(out_like) >= 3),
                  "text": "; ".join(parts) if parts else ""}
    return out


def flag_games(pred: pd.DataFrame, season: int) -> pd.DataFrame:
    """Adds inj_hold (bool) and inj_note (text) to a frame with away_team/home_team/week columns."""
    pred = pred.copy()
    try:
        inj, dc = load(season)
    except Exception as e:  # noqa: BLE001 - never break the scan over this
        print(f"WARNING: injury data unavailable ({e.__class__.__name__}); inj_hold left blank")
        pred["inj_hold"], pred["inj_note"] = None, "injury data unavailable"
        return pred
    holds, notes = [], []
    for r in pred.itertuples():
        s = summarize(inj, dc, int(r.week), [r.away_team, r.home_team])
        a, h = s[r.away_team], s[r.home_team]
        holds.append(a["hold"] or h["hold"])
        bits = []
        for team, d in ((r.away_team, a), (r.home_team, h)):
            if d["text"]:
                bits.append(f"{team}: {d['text']}")
        notes.append(" | ".join(bits))
    pred["inj_hold"], pred["inj_note"] = holds, notes
    return pred
