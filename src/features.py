"""Build per-game model features into data/processed/game_features.csv.

One row per regular-season game, from the HOME team's perspective, with
differentials (home - away) for the margin/win models and sums for the
totals model.

ANTI-LEAKAGE (hard rule): every rate used for a game is computed ONLY from
games strictly before it. Within a season we use expanding cumulative sums
shifted by one game. Early-season thin samples are stabilized with a
prior (see below) — never with future games.

PRIOR / SHRINKAGE: a team's rate entering a game is
    (cum_stats_this_season_before_game + PRIOR_WEIGHT * prior_rate)
    / (cum_plays_before_game + PRIOR_WEIGHT * prior_plays_per_game_scale)
implemented on summed counting stats with PRIOR_WEIGHT = 4.0 "games worth"
of the team's PRIOR-SEASON rate, itself shrunk 50% toward the prior-season
league rate. For 2019 (first ingested season) the prior is the league rate.
Prior stats are full prior-season aggregates — known before the season
starts, so no leakage.

HOUSE STYLE (standing directives from the sim work):
  * NO venue-split features — no home/road split rates anywhere; the only
    home effect is the single home_field indicator (standard HFA term).
  * NO opponent-class re-basing — rates are raw full-season rates, never
    re-based by opponent class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

PRIOR_WEIGHT = 4.0      # games-worth of prior-season rate blended in
PRIOR_SHRINK = 0.5      # prior-season team rate shrunk 50% to league mean
FIRST_SEASON, LAST_SEASON = 2019, 2026


def load_team_games() -> pd.DataFrame:
    """Team-game long frame: one row per team per game with raw counting stats."""
    games = pd.read_csv(RAW / "games.csv")
    games = games[(games["game_type"] == "REG") & (games["season"] >= FIRST_SEASON)]
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        p = RAW / f"team_week_{season}.csv"
        if p.exists():
            tw = pd.read_csv(p)
            frames.append(tw[tw["season_type"] == "REG"] if "season_type" in tw else tw)
    tw = pd.concat(frames, ignore_index=True)

    # Plays = pass attempts + sacks suffered + carries (dropbacks + rushes).
    tw["plays"] = tw["attempts"] + tw["sacks_suffered"] + tw["carries"]
    tw["yards"] = tw["passing_yards"] + tw["rushing_yards"]
    tw["epa"] = tw["passing_epa"].fillna(0) + tw["rushing_epa"].fillna(0)
    tw["first_downs"] = tw["passing_first_downs"] + tw["rushing_first_downs"]
    tw["giveaways"] = tw["passing_interceptions"] + tw["fumbles_lost_total"]
    keep = ["game_id", "season", "week", "team", "opponent_team",
            "plays", "yards", "epa", "first_downs", "giveaways",
            "def_interceptions", "fumble_recovery_opp"]
    tw = tw[keep].rename(columns={
        "def_interceptions": "takeaway_int", "fumble_recovery_opp": "takeaway_fum"})

    # Scores + QB starters from the schedules file (authoritative).
    home = games[["game_id", "season", "week", "gameday", "home_team", "home_score",
                  "home_qb_id", "home_rest"]].rename(columns={
        "home_team": "team", "home_score": "points", "home_qb_id": "qb_id",
        "home_rest": "rest"})
    away = games[["game_id", "season", "week", "gameday", "away_team", "away_score",
                  "away_qb_id", "away_rest"]].rename(columns={
        "away_team": "team", "away_score": "points", "away_qb_id": "qb_id",
        "away_rest": "rest"})
    sched = pd.concat([home, away], ignore_index=True)
    tg = sched.merge(tw, on=["game_id", "season", "week", "team"], how="left")
    # Opponent points: join the other row of the same game.
    opp_pts = tg[["game_id", "team", "points"]].rename(
        columns={"team": "opponent_team", "points": "points_allowed"})
    tg = tg.merge(opp_pts, on=["game_id", "opponent_team"], how="left")
    # Opponent offensive counting stats (for defensive rates).
    opp_stats = tw[["game_id", "team", "plays", "yards", "epa", "first_downs",
                    "giveaways"]].rename(columns={
        "team": "opponent_team", "plays": "opp_plays", "yards": "opp_yards",
        "epa": "opp_epa", "first_downs": "opp_first_downs",
        "giveaways": "opp_giveaways"})
    tg = tg.merge(opp_stats, on=["game_id", "opponent_team"], how="left")
    tg["takeaways"] = tg["takeaway_int"].fillna(0) + tg["takeaway_fum"].fillna(0)
    tg = tg.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
    return tg


RATE_DEFS = {
    # feature name -> (numerator col, denominator col) on the team-game frame
    "pts_per_play": ("points", "plays"),
    "yds_per_play": ("yards", "plays"),
    "epa_per_play": ("epa", "plays"),
    "fd_per_play": ("first_downs", "plays"),
    "giveaways_per_play": ("giveaways", "plays"),
    "def_pts_per_play": ("points_allowed", "opp_plays"),
    "def_yds_per_play": ("opp_yards", "opp_plays"),
    "def_epa_per_play": ("opp_epa", "opp_plays"),
    "def_fd_per_play": ("opp_first_downs", "opp_plays"),
    "takeaways_per_play": ("takeaways", "opp_plays"),
    "plays_per_game": ("plays", "games"),  # pace proxy
}


def blended_rates(tg: pd.DataFrame) -> pd.DataFrame:
    """Per team-game, the team's rates ENTERING that game (prior-only info)."""
    tg = tg.copy()
    tg["games"] = 1.0
    played = tg[tg["points"].notna()].copy()

    # Prior-season aggregates per team (full season, shrunk to league).
    prior_rows = {}
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        prev = played[played["season"] == season - 1]
        cur_teams = played[played["season"] == season]["team"].unique()
        if len(prev) == 0:
            # league-rate prior from the current season's own aggregates is
            # only used for the first ingested season (2019); noted in README.
            base = played[played["season"] == season]
            league = {name: (base[num].sum() / base[den].sum())
                      for name, (num, den) in RATE_DEFS.items()}
            for t in cur_teams:
                prior_rows[(season, t)] = league
            continue
        league = {name: (prev[num].sum() / prev[den].sum())
                  for name, (num, den) in RATE_DEFS.items()}
        for t in cur_teams:
            sub = prev[prev["team"] == t]
            if len(sub) == 0:
                prior_rows[(season, t)] = league
            else:
                prior_rows[(season, t)] = {
                    name: PRIOR_SHRINK * league[name]
                    + (1 - PRIOR_SHRINK) * (sub[num].sum() / sub[den].sum())
                    for name, (num, den) in RATE_DEFS.items()}

    out = []
    # Groups run over ALL team-game rows (played + upcoming) so upcoming
    # games also get entering rates: cumulative sums advance only on played
    # games (upcoming rows contribute 0), and every row's rate is the
    # shifted (prior-games-only) blend.
    for (season, team), grp in tg.groupby(["season", "team"], sort=False):
        grp = grp.sort_values(["week", "gameday"])
        prior = prior_rows[(season, team)]
        # Prior expressed as pseudo-counts: PRIOR_WEIGHT games at the prior
        # rate and the team's prior-season per-game denominator.
        prev = played[(played["season"] == season - 1) & (played["team"] == team)]
        den_prior_game = {}
        for name, (num, den) in RATE_DEFS.items():
            if len(prev):
                den_prior_game[name] = prev[den].sum() / len(prev)
            else:
                den_prior_game[name] = 60.0  # ~plays/game scale fallback
        rec = {}
        for name, (num, den) in RATE_DEFS.items():
            cum_num = grp[num].fillna(0.0).cumsum().shift(1).fillna(0.0)
            cum_den = grp[den].fillna(0.0).cumsum().shift(1).fillna(0.0)
            prior_num = PRIOR_WEIGHT * den_prior_game[name] * prior[name]
            prior_den = PRIOR_WEIGHT * den_prior_game[name]
            rec[name] = (cum_num + prior_num) / (cum_den + prior_den)
        r = pd.DataFrame(rec, index=grp.index)
        r["n_prior_games"] = grp["points"].notna().cumsum().shift(1).fillna(0).values
        out.append(r)
    rates = pd.concat(out).sort_index()
    return tg.join(rates)


def build_features() -> pd.DataFrame:
    tg = blended_rates(load_team_games())
    games = pd.read_csv(RAW / "games.csv")
    games = games[(games["game_type"] == "REG") & (games["season"] >= FIRST_SEASON)]

    # QB-change flag: starter differs from the team's previous game (any season).
    seq = tg.sort_values(["team", "gameday"]).copy()
    seq["prev_qb"] = seq.groupby("team")["qb_id"].shift(1)
    seq["qb_change"] = (seq["qb_id"] != seq["prev_qb"]).astype(float)
    seq.loc[seq["prev_qb"].isna(), "qb_change"] = 0.0
    qbmap = seq.set_index(["game_id", "team"])["qb_change"]

    rate_names = list(RATE_DEFS)
    home = tg.set_index(["game_id", "team"])
    rows = []
    for g in games.itertuples():
        hk, ak = (g.game_id, g.home_team), (g.game_id, g.away_team)
        if hk not in home.index or ak not in home.index:
            continue
        h, a = home.loc[hk], home.loc[ak]
        if isinstance(h, pd.DataFrame):  # duplicate guard
            h, a = h.iloc[0], a.iloc[0]
        row = {
            "game_id": g.game_id, "season": g.season, "week": g.week,
            "gameday": g.gameday, "away_team": g.away_team, "home_team": g.home_team,
            "home_field": 1.0 if str(g.location).lower() == "home" else 0.0,
            "rest_diff": (g.home_rest or 0) - (g.away_rest or 0),
            "qb_change_home": qbmap.get(hk, 0.0), "qb_change_away": qbmap.get(ak, 0.0),
            "spread_line": g.spread_line, "total_line": g.total_line,
            "home_moneyline": g.home_moneyline, "away_moneyline": g.away_moneyline,
            "home_score": g.home_score, "away_score": g.away_score,
        }
        for name in rate_names:
            row[f"d_{name}"] = h[name] - a[name]        # home - away (margin model)
            row[f"s_{name}"] = h[name] + a[name]        # sum (totals model)
        row["d_off_rating"] = (h["epa_per_play"] - h["def_epa_per_play"]) - \
                              (a["epa_per_play"] - a["def_epa_per_play"])
        row["d_pts_rating"] = (h["pts_per_play"] - h["def_pts_per_play"]) - \
                              (a["pts_per_play"] - a["def_pts_per_play"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_pts"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(float)
    df.loc[df["margin"] == 0, "home_win"] = np.nan  # ties excluded from prob grading
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


MARGIN_FEATURES = [
    "home_field", "rest_diff", "qb_change_home", "qb_change_away",
    "d_pts_per_play", "d_yds_per_play", "d_epa_per_play", "d_fd_per_play",
    "d_giveaways_per_play", "d_def_pts_per_play", "d_def_yds_per_play",
    "d_def_epa_per_play", "d_def_fd_per_play", "d_takeaways_per_play",
    "d_off_rating", "d_pts_rating",
]
TOTAL_FEATURES = [
    "s_pts_per_play", "s_yds_per_play", "s_epa_per_play", "s_fd_per_play",
    "s_def_pts_per_play", "s_def_yds_per_play", "s_def_epa_per_play",
    "s_plays_per_game", "rest_diff",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_features()
    path = OUT / "game_features.csv"
    df.to_csv(path, index=False)
    played = df[df["home_score"].notna()]
    print(f"game_features.csv: {len(df)} games "
          f"({played['season'].min():.0f}-{played['season'].max():.0f} played: {len(played)}, "
          f"upcoming: {len(df) - len(played)})")
    print(f"features: {len(MARGIN_FEATURES)} margin/win, {len(TOTAL_FEATURES)} totals")
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
