"""QB check for upcoming NFL games: adjusts the model's margin for who is actually starting, and flags games
where the starter changed.

Why: the team-level model blends a team's whole season of stats, so it cannot see that the QB behind those
numbers is hurt or replaced. Backtest (tools/qb_backtest.py, 680 games): model error correlated +0.18 with
"QB surprise", and in big-surprise games its RMSE was 14.7 vs the market's 12.8. Adjusting by ~25 points of margin
per 1.0 EPA/dropback made it less wrong, but still NOT better than the market. So: adjust, and mark big changes
as qb_hold (treat as NO LEAN, and exclude from the scoreboard).

QB rating = career-to-date EPA per dropback (passing + rushing EPA / (attempts + sacks)), shrunk toward PRIOR
with K dropbacks. Needs stats_player_week files from nflverse (downloaded on the runner; if that fails the
scan carries on with no QB adjustment and says so).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

K, PRIOR = 150.0, -0.05
SLOPE = 25.0          # points of margin per 1.0 EPA/dropback (backtest: 24.5-32.2 across folds; kept conservative)
CLIP = 0.5            # cap on the home-minus-away surprise, in EPA/dropback (= 12.5 pts)
HOLD_AT = 0.15        # |home-away surprise| at/above this = QB change big enough to distrust the lean
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
COLS = ["player_id", "player_display_name", "position", "season", "week", "team", "attempts",
        "sacks_suffered", "passing_epa", "rushing_epa", "season_type"]


def fetch_qb_games(seasons, cache: Path | None = None) -> pd.DataFrame:
    """QB game rows from nflverse (retrying fetch from ingest). Falls back to the cache file if the download fails."""
    import ingest
    frames = []
    for s in seasons:
        try:
            raw = ingest.fetch(f"{BASE}/stats_player/stats_player_week_{s}.csv")
            df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in COLS, low_memory=False)
            frames.append(df[df["position"] == "QB"])
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: QB stats {s} unavailable ({e.__class__.__name__})")
    if frames:
        out = pd.concat(frames, ignore_index=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(cache, index=False)
        return out
    if cache is not None and cache.exists():
        print("WARNING: using cached QB stats")
        return pd.read_csv(cache, low_memory=False)
    return pd.DataFrame(columns=COLS)


def ratings(qb: pd.DataFrame) -> pd.DataFrame:
    """Current rating per QB (through every game in `qb`) plus this-season dropbacks per team."""
    q = qb[(qb.get("season_type", "REG") == "REG")].copy() if "season_type" in qb else qb.copy()
    q["db"] = q["attempts"].fillna(0) + q["sacks_suffered"].fillna(0)
    q["epa"] = q["passing_epa"].fillna(0) + q["rushing_epa"].fillna(0)
    q = q[q["db"] > 0]
    g = q.groupby("player_id").agg(epa=("epa", "sum"), db=("db", "sum"), name=("player_display_name", "last"))
    g["rating"] = (g["epa"] + K * PRIOR) / (g["db"] + K)
    return g


def embedded(qb: pd.DataFrame, rate: pd.DataFrame, season: int) -> dict:
    """Per team: dropback-weighted rating of the QBs who played this season = the QB quality its stats embed."""
    q = qb[(qb["season"] == season)].copy()
    if "season_type" in q:
        q = q[q["season_type"] == "REG"]
    q["db"] = q["attempts"].fillna(0) + q["sacks_suffered"].fillna(0)
    q = q[q["db"] > 0]
    q["r"] = q["player_id"].map(rate["rating"]).fillna(PRIOR)
    out = {}
    for t, d in q.groupby("team"):
        out[t] = float((d["r"] * d["db"]).sum() / d["db"].sum())
    return out


def pick_starter(dc: pd.DataFrame, inj: pd.DataFrame, team: str, week: int):
    """(gsis_id, name, status_of_QB1, replaced) using the latest depth chart and the week's injury report."""
    last = dc[dc["dt"] == dc["dt"].max()]
    qbs = last[(last["team"] == team) & (last["pos_abb"] == "QB")].sort_values("pos_rank")
    if qbs.empty:
        return None, None, None, False
    w = inj[(inj["week"] == week) & (inj["team"] == team)]
    status = dict(zip(w["gsis_id"], w["report_status"]))
    name_col = "player_name" if "player_name" in qbs else None
    first = qbs.iloc[0]
    s1 = status.get(first["gsis_id"])
    for r in qbs.itertuples():
        if status.get(r.gsis_id) not in ("Out", "Doubtful"):
            nm = getattr(r, "player_name", None) if name_col else None
            return r.gsis_id, nm, s1, r.gsis_id != first["gsis_id"]
    r = qbs.iloc[0]
    return r["gsis_id"], (r["player_name"] if name_col else None), s1, False


def adjust(pred: pd.DataFrame, qb: pd.DataFrame, dc: pd.DataFrame, inj: pd.DataFrame, season: int) -> pd.DataFrame:
    """Adds qb_adj (points added to model_margin), qb_hold (bool) and qb_note. Does not touch other columns."""
    pred = pred.copy()
    rate = ratings(qb)
    emb = embedded(qb, rate, season)
    adjs, holds, notes = [], [], []
    for r in pred.itertuples():
        info = {}
        for side, team in (("home", r.home_team), ("away", r.away_team)):
            gid, nm, s1, replaced = pick_starter(dc, inj, team, int(r.week))
            rt = float(rate["rating"].get(gid, PRIOR)) if gid is not None else PRIOR
            nm = nm or (rate["name"].get(gid) if gid is not None else None) or "unknown QB"
            e = emb.get(team, rt)
            info[side] = dict(team=team, name=nm, rating=rt, delta=rt - e, s1=s1, replaced=replaced)
        h, a = info["home"], info["away"]
        qbd = float(np.clip(h["delta"] - a["delta"], -CLIP, CLIP))
        replaced_any = h["replaced"] or a["replaced"]
        hold = bool(replaced_any or abs(h["delta"] - a["delta"]) >= HOLD_AT)
        adjs.append(round(SLOPE * qbd, 2))
        holds.append(hold)
        bits = []
        for d in (a, h):
            if d["replaced"]:
                bits.append(f"{d['team']} starter OUT -> {d['name']}")
            elif d["s1"] == "Questionable":
                bits.append(f"{d['team']} QB {d['name']} Questionable")
            elif abs(d["delta"]) >= HOLD_AT:
                bits.append(f"{d['team']} QB {d['name']} {'below' if d['delta'] < 0 else 'above'} the team's season QB level")
        if bits:
            notes.append("QB: " + "; ".join(bits) + f" | model margin adjusted {SLOPE * qbd:+.1f}")
        elif abs(qbd) >= 0.05:
            notes.append(f"QB: model margin adjusted {SLOPE * qbd:+.1f} for QB quality vs team's season level")
        else:
            notes.append("")
    pred["qb_adj"], pred["qb_hold"], pred["qb_note"] = adjs, holds, notes
    return pred
