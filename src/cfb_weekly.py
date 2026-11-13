"""College football weekly scorer + logger + grader (ONE self-contained file).

WHAT IT DOES
  python src/cfb_weekly.py            score upcoming FBS-vs-FBS games that have a line,
                                      write outputs/cfb_weekly_predictions.csv, and
                                      append a snapshot row per game to outputs/cfb_log.csv
  python src/cfb_weekly.py --grade    grade every logged game that has since finished

WHAT IT SCORES (and what it does NOT)
  Backtests (cfb_residual_test.py, 3,559 games, 2022-2026) found:
    * SPREADS : no signal (R2 -0.1%; ~50% on the biggest disagreements). SIDE leans are
                scored and logged anyway, labeled "no edge", so the result can be checked.
    * TOTALS  : a tiny, BORDERLINE signal (R2 about +0.2-0.4% of the line's error; the
                95% CI's lower end sits at about zero depending on the bootstrap draw;
                best of 3 settings; bucket ROI about 0 at -110). Scored as a lean.
    * LINE MOVES: the model's direction agreed with the open->current move 51.5-53.4%
                of the time (50% = coin flip). Scored as "which way should the line
                move from here".
  These are leans to LOG and grade, not bets. Nothing here is a betting recommendation.

HOW THE NUMBERS ARE BUILT (same pipeline as the tested backtest)
  Training set = every completed FBS-vs-FBS regular-season game with a line (2020+).
  Features known before kickoff only: pregame Elo, PRIOR-season SP+/FPI, in-season
  rolling advanced stats shrunk toward last season, neutral site, conference game,
  and the market line. Ridge regression (alpha 3000 for totals, 300 for moves)
  predicts the error of the market line.
    pred_total_resid  = predicted (game total - market total), CENTERED on the historical
                        average so only the game-specific part remains; + means lean OVER.
                        (Uncentered, ~90% of games lean over; overs hit only 50.4% in testing.)
    remaining_move    = predicted extra move of the home spread from the CURRENT line
                        (model residual vs the open line, minus the move already made);
                        + means the line should move toward the HOME team.
  Market line = median across real sportsbooks (consensus/numberfire/teamrankings
  rows excluded). CFBD's "current" line is the latest snapshot it holds, not a
  verified closing line.

REFRESHING THE DATA (needed before every run; the script warns if it is stale)
    export CFBD_API_KEY=...      (Colab: os.environ[...] = ...)
    python src/cfb_pull.py --start 2026 --end 2026 --refresh
  then run this script.

GRADING (--grade)
  Uses the FIRST logged snapshot of each game as the entry line and the latest line in
  the refreshed data as the proxy close:
    total : entry total vs final total; lean over/under wins/loses/pushes; ROI at -110.
    move  : did the home spread move from the entry line in the direction of the lean?
  Reports record + Wilson 95% CIs. Need a few hundred graded games before believing
  anything.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PT = ZoneInfo("America/Los_Angeles")

K_PRIOR = 4.0
ALPHA_TOTAL = 3000.0
ALPHA_MOVE = 300.0
ALPHA_SIDE = 30000.0   # best-tested setting for sides; predictions are tiny because there is no signal
NON_BOOKS = {"numberfire", "teamrankings", "consensus"}
STATS = ["offense_ppa", "defense_ppa", "offense_successRate", "defense_successRate",
         "offense_explosiveness", "defense_explosiveness", "offense_plays"]
TOTAL_F = ["s_off_ppa", "s_def_ppa", "s_off_sr", "s_def_sr", "s_plays", "s_sp_off", "s_sp_def",
           "neutral", "conf_game", "mkt_total"]
OPEN_F = ["d_elo", "d_sp", "d_fpi", "d_off_ppa", "d_def_ppa", "d_off_sr", "d_def_sr",
          "d_off_xp", "d_def_xp", "neutral", "conf_game", "sp_open"]
SIDE_F = ["d_elo", "d_sp", "d_fpi", "d_off_ppa", "d_def_ppa", "d_off_sr", "d_def_sr",
          "d_off_xp", "d_def_xp", "neutral", "conf_game", "mkt_spread", "line_abs"]
LOG_COLS = ["run_ts_utc", "season", "week", "game_id", "start_utc", "home", "away",
            "mkt_spread", "sp_open", "mkt_total", "ou_open", "n_books",
            "pred_total_resid", "total_lean", "pred_open_resid", "move_so_far",
            "remaining_move", "move_lean", "pred_side_resid", "side_lean"]


# ----------------------------------------------------------------------- data
def load(data_dir: Path, name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(data_dir / f"{name}_*.csv")))
    if not fs:
        raise SystemExit(f"No {name}_*.csv files in {data_dir}. Run cfb_pull.py first.")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def _base_for(season, team, prior, league):
    if (season - 1) in league.index:
        lg = league.loc[season - 1]
        pv = prior.loc[(season - 1, team)] if (season - 1, team) in prior.index else lg
        return 0.5 * lg + 0.5 * pv
    return pd.Series(np.nan, index=STATS)


def rolling_rates(adv: pd.DataFrame) -> pd.DataFrame:
    """Per team-game rates ENTERING that game (prior games + shrunk prior season)."""
    adv = adv[adv["seasonType"] == "regular"][["gameId", "season", "week", "team"] + STATS]
    adv = adv.sort_values(["season", "team", "week"]).reset_index(drop=True)
    prior = adv.groupby(["season", "team"])[STATS].mean()
    league = adv.groupby("season")[STATS].mean()
    out = []
    for (season, team), grp in adv.groupby(["season", "team"], sort=False):
        base = _base_for(season, team, prior, league)
        cum = grp[STATS].cumsum().shift(1).fillna(0.0)
        n = np.arange(len(grp)).reshape(-1, 1)
        rate = (cum.to_numpy() + K_PRIOR * base.to_numpy()) / (n + K_PRIOR)
        r = pd.DataFrame(rate, columns=STATS, index=grp.index)
        r["gameId"], r["team"] = grp["gameId"].values, team
        out.append(r)
    return pd.concat(out, ignore_index=True)


def current_rates(adv: pd.DataFrame) -> pd.DataFrame:
    """Each team's rates entering its NEXT game: every completed game this season
    plus the same shrunk prior-season base as rolling_rates (identical formula)."""
    adv = adv[adv["seasonType"] == "regular"][["gameId", "season", "week", "team"] + STATS]
    prior = adv.groupby(["season", "team"])[STATS].mean()
    league = adv.groupby("season")[STATS].mean()
    season = int(adv["season"].max())
    teams = set(adv.loc[adv["season"] == season, "team"]) | \
        set(adv.loc[adv["season"] == season - 1, "team"])
    rows = []
    for team in sorted(teams):
        cur = adv[(adv["season"] == season) & (adv["team"] == team)]
        base = _base_for(season, team, prior, league)
        rate = (cur[STATS].sum().to_numpy() + K_PRIOR * base.to_numpy()) / (len(cur) + K_PRIOR)
        rows.append({"season": season, "team": team, **dict(zip(STATS, rate))})
    return pd.DataFrame(rows)


def prior_ratings(data_dir: Path):
    sp = load(data_dir, "sp")[["year", "team", "rating", "offense_rating", "defense_rating"]].copy()
    sp = sp.rename(columns={"year": "season", "rating": "sp", "offense_rating": "sp_off",
                            "defense_rating": "sp_def"})
    sp["season"] += 1
    fp = load(data_dir, "fpi")[["year", "team", "fpi"]].rename(columns={"year": "season"})
    fp["season"] += 1
    return sp, fp


def build(data_dir: Path, upcoming: bool = False) -> pd.DataFrame:
    """One row per FBS-vs-FBS regular-season game that has a market line.
    upcoming=False -> completed games (training / grading);
    upcoming=True  -> games not yet marked completed (callers filter by kickoff)."""
    games, lines = load(data_dir, "games"), load(data_dir, "lines")
    fbs = (games.seasonType == "regular") & (games.homeClassification == "fbs") \
        & (games.awayClassification == "fbs")
    done = games["completed"].fillna(False).astype(bool) & games["homePoints"].notna()
    g = games[fbs & (~done if upcoming else done)].copy()
    g["margin"] = g.homePoints - g.awayPoints
    g["game_total"] = g.homePoints + g.awayPoints

    books = lines[~lines.line_provider.isin(NON_BOOKS)]
    mk = books.groupby("id").agg(
        mkt_spread=("line_spread", "median"), mkt_total=("line_overUnder", "median"),
        sp_open=("line_spreadOpen", "median"), ou_open=("line_overUnderOpen", "median"),
        n_books=("line_provider", "nunique")).reset_index()
    g = g.merge(mk, on="id", how="inner").dropna(subset=["mkt_spread", "mkt_total"])

    adv = load(data_dir, "adv_game_stats")
    if not upcoming:
        rr = rolling_rates(adv)
        for side, col in (("home", "homeTeam"), ("away", "awayTeam")):
            x = rr.rename(columns={s: f"{side}_{s}" for s in STATS})
            x = x.rename(columns={"gameId": "id", "team": col})
            g = g.merge(x, on=["id", col], how="left")
    else:
        cr = current_rates(adv)
        for side, col in (("home", "homeTeam"), ("away", "awayTeam")):
            x = cr.rename(columns={s: f"{side}_{s}" for s in STATS}).rename(columns={"team": col})
            g = g.merge(x, on=["season", col], how="left")
        # fall back to the current Elo rating if a pregame value is missing
        elo = load(data_dir, "elo")[["year", "team", "elo"]].rename(columns={"year": "season"})
        for side, col in (("home", "homeTeam"), ("away", "awayTeam")):
            e = elo.rename(columns={"team": col, "elo": f"_{side}_elo"})
            g = g.merge(e, on=["season", col], how="left")
            g[f"{side}PregameElo"] = g[f"{side}PregameElo"].fillna(g[f"_{side}_elo"])

    sp, fp = prior_ratings(data_dir)
    for side, col in (("home", "homeTeam"), ("away", "awayTeam")):
        g = g.merge(sp.rename(columns={"team": col, "sp": f"{side}_sp", "sp_off": f"{side}_sp_off",
                                       "sp_def": f"{side}_sp_def"}), on=["season", col], how="left")
        g = g.merge(fp.rename(columns={"team": col, "fpi": f"{side}_fpi"}), on=["season", col], how="left")
    for c in ("sp", "sp_off", "sp_def", "fpi"):
        src = sp if c != "fpi" else fp
        fill = src.groupby("season")[c].quantile(0.10)
        for side in ("home", "away"):
            g[f"{side}_{c}"] = g[f"{side}_{c}"].fillna(g["season"].map(fill))

    d = lambda c: g[f"home_{c}"] - g[f"away_{c}"]   # noqa: E731
    s = lambda c: g[f"home_{c}"] + g[f"away_{c}"]   # noqa: E731
    g["d_elo"] = g["homePregameElo"] - g["awayPregameElo"]
    g["d_sp"], g["d_fpi"] = d("sp"), d("fpi")
    g["d_off_ppa"], g["d_def_ppa"] = d("offense_ppa"), -d("defense_ppa")
    g["d_off_sr"], g["d_def_sr"] = d("offense_successRate"), -d("defense_successRate")
    g["d_off_xp"], g["d_def_xp"] = d("offense_explosiveness"), -d("defense_explosiveness")
    g["neutral"] = g["neutralSite"].astype(float)
    g["conf_game"] = g["conferenceGame"].astype(float)
    g["line_abs"] = g["mkt_spread"].abs()
    g["s_off_ppa"], g["s_def_ppa"] = s("offense_ppa"), s("defense_ppa")
    g["s_off_sr"], g["s_def_sr"] = s("offense_successRate"), s("defense_successRate")
    g["s_plays"] = s("offense_plays")
    g["s_sp_off"], g["s_sp_def"] = s("sp_off"), s("sp_def")
    g["res_sp"] = g["margin"] + g["mkt_spread"]
    g["res_tot"] = g["game_total"] - g["mkt_total"]
    g["res_sp_open"] = g["margin"] + g["sp_open"]
    g["move_home"] = g["sp_open"] - g["mkt_spread"]
    return g.sort_values(["season", "week", "id"]).reset_index(drop=True)


# --------------------------------------------------------------------- models
def _pipe(alpha: float) -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])


def _fit(train: pd.DataFrame, feats: list[str], target: str, alpha: float):
    d = train.dropna(subset=feats + [target])
    return _pipe(alpha).fit(d[feats], d[target]), len(d)


def _ev_pct(pred_pts: pd.Series, sd: float) -> pd.Series:
    """EV per 100 risked at -110 for betting the lean, IF the model's win chance were right."""
    from math import erf, sqrt
    p_pos = pred_pts.map(lambda x: np.nan if pd.isna(x) else 0.5 * (1 + erf(x / (sd * sqrt(2)))))
    p_lean = np.where(pred_pts > 0, p_pos, 1 - p_pos)
    return pd.Series(100 * (p_lean * (100 / 110) - (1 - p_lean)), index=pred_pts.index).round(1)


def score_upcoming(train: pd.DataFrame, up: pd.DataFrame) -> pd.DataFrame:
    mt, nt = _fit(train, TOTAL_F, "res_tot", ALPHA_TOTAL)
    mm, nm = _fit(train[train["sp_open"].notna()], OPEN_F, "res_sp_open", ALPHA_MOVE)
    ms, ns = _fit(train, SIDE_F, "res_sp", ALPHA_SIDE)
    s_mean = float(train.dropna(subset=SIDE_F + ["res_sp"])["res_sp"].mean())   # centered like totals
    # Totals are CENTERED: the ridge intercept is just the historical average of
    # (total - line) (about +0.4), which would make nearly every game "lean over" even
    # though overs hit only 50.4% in testing. Only the game-specific part is a lean.
    t_mean = float(train.dropna(subset=TOTAL_F + ["res_tot"])["res_tot"].mean())
    up = up.copy()
    up["pred_total_resid"] = np.nan
    up["pred_open_resid"] = np.nan
    ok_t = up[TOTAL_F].notna().all(axis=1)
    ok_m = up[OPEN_F].notna().all(axis=1)
    if ok_t.any():
        up.loc[ok_t, "pred_total_resid"] = mt.predict(up.loc[ok_t, TOTAL_F]) - t_mean
    if ok_m.any():
        up.loc[ok_m, "pred_open_resid"] = mm.predict(up.loc[ok_m, OPEN_F])
    up["pred_side_resid"] = np.nan
    ok_s = up[SIDE_F].notna().all(axis=1)
    if ok_s.any():
        up.loc[ok_s, "pred_side_resid"] = ms.predict(up.loc[ok_s, SIDE_F]) - s_mean
    # + = model thinks the HOME side covers the current spread; magnitude is in points.
    up["side_lean"] = np.where(up["pred_side_resid"].isna(), "",
                               np.where(up["pred_side_resid"] > 0, "home", "away"))
    up["side_fair_spread"] = up["mkt_spread"] - up["pred_side_resid"]
    up["fair_total"] = up["mkt_total"] + up["pred_total_resid"]
    up["total_lean"] = np.where(up["pred_total_resid"].isna(), "",
                                np.where(up["pred_total_resid"] > 0, "over", "under"))
    # UNCALIBRATED expected value if the model's probabilities were right (standard -110 both sides).
    # Win chance = normal CDF of (predicted points off the line / historical residual SD). The model
    # has NOT shown an edge, so treat this as 'what the model believes', not a real expectation.
    sd_t = float(train["res_tot"].std())
    sd_s = float(train["res_sp"].std())
    up["ev_total_pct"] = _ev_pct(up["pred_total_resid"], sd_t)
    up["ev_side_pct"] = _ev_pct(up["pred_side_resid"], sd_s)
    up["move_so_far"] = up["sp_open"] - up["mkt_spread"]        # + = already moved toward home
    up["remaining_move"] = up["pred_open_resid"] - up["move_so_far"]
    up["fair_home_spread"] = up["mkt_spread"] - up["remaining_move"]
    up["move_lean"] = np.where(up["remaining_move"].isna(), "",
                               np.where(up["remaining_move"] > 0, "home", "away"))
    # Big moves since the open usually mean news (injury / QB / weather) the model cannot see.
    up["check_news"] = np.where(((up["sp_open"] - up["mkt_spread"]).abs() >= 4) |
                                ((up["ou_open"] - up["mkt_total"]).abs() >= 4), "CHECK NEWS", "")
    print(f"models fit on {nt} games (totals) / {nm} games (line moves) / {ns} games (sides)")
    return up


# ------------------------------------------------------------------------ stats
def wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph, den = w / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fmt_tally(label: str, w: int, n: int, pushes: int = 0, roi: bool = False) -> str:
    if n == 0:
        return f"  {label}: no graded games yet"
    lo, hi = wilson(w, n)
    s = f"  {label}: {w}-{n - w}" + (f"-{pushes}" if pushes else "") + \
        f"  = {100 * w / n:.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f}%, n={n})"
    if roi:
        s += f"  ROI at -110: {100 * (w * 100 / 110 - (n - w)) / n:+.1f}%"
    return s


def plain_view(res: pd.DataFrame) -> pd.DataFrame:
    """One readable line per game (the full numeric table still goes to the CSV)."""
    def f(r):
        h, a = r.homeTeam, r.awayTeam
        line = f"{h} {r.mkt_spread:+.1f}" + (f" (open {r.sp_open:+.1f})" if pd.notna(r.sp_open) else "")
        side = ""
        if pd.notna(r.pred_side_resid) and abs(r.pred_side_resid) >= 0.5:
            side = f"{h if r.side_lean == 'home' else a} +{abs(r.pred_side_resid):.1f}"
        tot = ""
        if pd.notna(r.pred_total_resid) and abs(r.pred_total_resid) >= 0.5:
            tot = f"{r.total_lean.upper()} {r.mkt_total:.1f}->{r.fair_total:.1f}"
        mv = ""
        if pd.notna(r.remaining_move) and abs(r.remaining_move) >= 0.5:
            mv = f"toward {h if r.move_lean == 'home' else a} {abs(r.remaining_move):.1f}"
        evs = [(r.ev_side_pct, "side"), (r.ev_total_pct, "total")]
        evs = [e for e in evs if pd.notna(e[0])]
        ev = f"{max(evs)[0]:+.0f}% {max(evs)[1]}" if evs else ""
        wx = r.wx_note if getattr(r, "wx_flag", None) is True else ""
        return pd.Series({"Kickoff": r.kickoff_pt, "Game": f"{a} @ {h}", "Line": line, "SIDE": side,
                          "TOTAL": tot, "MOVE": mv, "EV": ev,
                          "NEWS?": "CHECK" if r.check_news else "", "WEATHER": wx})
    return res.apply(f, axis=1)


# ------------------------------------------------------------------------- run
def run_score(args) -> int:
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")

    train = build(data_dir, upcoming=False)
    up_all = build(data_dir, upcoming=True)
    up_all["start"] = pd.to_datetime(up_all["startDate"], utc=True)

    stale = up_all[up_all["start"] < now - pd.Timedelta(hours=4)]
    if len(stale):
        print(f"WARNING: {len(stale)} game(s) in your data kicked off more than 4 hours ago "
              f"but are not marked completed -> the data is STALE. Re-run:\n"
              f"  python src/cfb_pull.py --start {int(train['season'].max())} "
              f"--end {int(train['season'].max())} --refresh\n")
    last_done = pd.to_datetime(train.loc[train["season"] == train["season"].max(), "startDate"], utc=True).max()
    print(f"data as of: most recent completed game kicked off {last_done:%a %b %d %H:%M} UTC | now {now:%a %b %d %H:%M} UTC")

    up = up_all[(up_all["start"] > now) & (up_all["start"] <= now + pd.Timedelta(days=args.days))].copy()
    if up.empty:
        print(f"No upcoming FBS-vs-FBS games with a market line in the next {args.days} days "
              f"(lines usually post a few days before kickoff).")
        return 0
    res = score_upcoming(train, up).sort_values("start")
    try:
        import cfb_weather
        res = cfb_weather.flag_games(res, data_dir)
    except Exception as e:  # noqa: BLE001 - weather is optional context
        print(f"WARNING: weather skipped ({e.__class__.__name__}: {e})")
        res["wx_note"], res["wx_flag"] = "", None
    res["kickoff_pt"] = res["start"].dt.tz_convert(PT).dt.strftime("%a %m/%d %I:%M %p")
    res["game"] = res["awayTeam"] + " @ " + res["homeTeam"]

    show = res[["kickoff_pt", "game", "mkt_spread", "sp_open", "mkt_total", "ou_open", "n_books",
                "side_lean", "pred_side_resid", "side_fair_spread",
                "pred_total_resid", "fair_total", "total_lean", "remaining_move", "fair_home_spread",
                "move_lean", "ev_side_pct", "ev_total_pct", "check_news", "wx_note", "wx_flag"]].copy()
    for c in ("fair_total", "remaining_move", "fair_home_spread", "pred_total_resid"):
        show[c] = show[c].round(1)
    for c in ("pred_side_resid", "side_fair_spread"):
        show[c] = show[c].round(2)
    show.to_csv(out_dir / "cfb_weekly_predictions.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    if getattr(args, "wide", False):
        print(f"\n{len(show)} upcoming games with lines (spread is the HOME line: negative = home favored)\n")
        print(show.to_string(index=False))
    else:
        print(f"\n{len(show)} upcoming games with lines. Full columns are in {out_dir / 'cfb_weekly_predictions.csv'} "
              f"(or run with --wide).\n")
        print(plain_view(res).to_string(index=False))
        print("\nHOW TO READ: 'Line' = home team's current spread (open in brackets); negative = favored.\n"
              "SIDE = team the model says covers, and by how many points of edge (blank = under 0.5 pt, i.e. no lean). TOTAL = lean, market total -> model total (blank = under 0.5 pt).\n"
              "MOVE = which way the model expects the spread to still move. EV = best uncalibrated EV (side/total).\n"
              "WEATHER = shown only for bad conditions (wind 15+ mph, gusts 25+, rain 60%+, 25F or colder); context only.\n"
              "NEWS? = line or total moved 4+ points since the open (injury/suspension?) - skip those leans until checked.")
    print("\nev_side_pct / ev_total_pct = UNCALIBRATED expected value per 100 risked at -110 IF the model's win chance were\n"
          "right. Negative = the vig outweighs the model's edge. Not a bet signal: no edge has been proven.")
    print("\nSIDE leans (side_lean = which team covers the current spread; side_fair_spread = model's home line) are shown\n"
          "for the record only: NO edge was found on sides in testing (R2 -0.1%, ~50% on its biggest disagreements).\n"
          "Totals are borderline. All of these are leans to LOG and grade, not bets.")

    # ---- append to log (skip when the game's lines + predictions haven't changed)
    log_path = out_dir / "cfb_log.csv"
    old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=LOG_COLS)
    last = old.sort_values("run_ts_utc").groupby("game_id").tail(1).set_index("game_id") if len(old) else None
    rows = []
    for r in res.itertuples():
        same = (last is not None and r.id in last.index and "pred_side_resid" in last.columns and
                pd.notna(last.loc[r.id, "pred_side_resid"]) and
                last.loc[r.id, "mkt_spread"] == r.mkt_spread and last.loc[r.id, "mkt_total"] == r.mkt_total)
        if same:
            continue
        rows.append({"run_ts_utc": now.isoformat(), "season": r.season, "week": r.week, "game_id": r.id,
                     "start_utc": r.start.isoformat(), "home": r.homeTeam, "away": r.awayTeam,
                     "mkt_spread": r.mkt_spread, "sp_open": r.sp_open, "mkt_total": r.mkt_total,
                     "ou_open": r.ou_open, "n_books": r.n_books,
                     "pred_total_resid": r.pred_total_resid, "total_lean": r.total_lean,
                     "pred_open_resid": r.pred_open_resid, "move_so_far": r.move_so_far,
                     "remaining_move": r.remaining_move, "move_lean": r.move_lean,
                     "pred_side_resid": r.pred_side_resid, "side_lean": r.side_lean})
    if rows:
        pd.concat([old, pd.DataFrame(rows)[LOG_COLS]], ignore_index=True).to_csv(log_path, index=False)
    print(f"log: {len(rows)} new snapshot rows -> {log_path} (total rows {len(old) + len(rows)})")
    return 0


def _first(log: pd.DataFrame, col: str, valid) -> pd.DataFrame:
    """Earliest logged snapshot per game that actually carries `col` (older log rows
    predate some columns, e.g. the side leans)."""
    if col not in log.columns:
        return log.iloc[0:0]
    f = log[log[col].isin(valid)].sort_values("run_ts_utc")
    return f.groupby("game_id").head(1)


def run_grade(args) -> int:
    data_dir, out_dir = Path(args.data), Path(args.out)
    log_path = out_dir / "cfb_log.csv"
    if not log_path.exists():
        print(f"No log yet at {log_path}. Run the scorer first, then grade after games finish.")
        return 0
    log = pd.read_csv(log_path)
    done = build(data_dir, upcoming=False)[["id", "margin", "game_total", "mkt_spread", "mkt_total"]]
    done = done.rename(columns={"id": "game_id", "mkt_spread": "latest_spread", "mkt_total": "latest_total"})
    ids = set(log["game_id"].unique())
    n_done = len(ids & set(done["game_id"]))
    print(f"logged games: {len(ids)} | finished and gradeable: {n_done} | still pending: {len(ids) - n_done}")
    if n_done == 0:
        return 0

    # TOTALS
    t = _first(log, "total_lean", ["over", "under"]).merge(done, on="game_id", how="inner")
    t["diff"] = t["game_total"] - t["mkt_total"]
    t["push"] = t["diff"] == 0
    t["win"] = ((t["diff"] > 0) & (t["total_lean"] == "over")) | ((t["diff"] < 0) & (t["total_lean"] == "under"))
    live = t[~t["push"]]
    print("\nTOTALS (entry = first logged total; lean over/under; borderline signal in testing):")
    print(fmt_tally("all leans", int(live["win"].sum()), len(live), int(t["push"].sum()), roi=True))
    if len(live) >= 12:
        big = live[live["pred_total_resid"].abs() >= live["pred_total_resid"].abs().quantile(2 / 3)]
        print(fmt_tally("largest third by |pred|", int(big["win"].sum()), len(big), roi=True))

    # LINE MOVES
    v = _first(log, "move_lean", ["home", "away"]).merge(done, on="game_id", how="inner")
    v["moved_home"] = v["mkt_spread"] - v["latest_spread"]      # + = line moved toward home since entry
    mv = v[v["moved_home"] != 0].copy()
    mv["agree"] = (np.sign(mv["moved_home"]) == np.where(mv["move_lean"] == "home", 1, -1))
    print("\nLINE MOVES (entry spread -> latest spread in your data; direction of the lean):")
    print(fmt_tally(f"lean matched the move ({len(v) - len(mv)} games did not move)", int(mv["agree"].sum()), len(mv)))
    print("  (50% = coin flip. 'Latest' is the last line CFBD holds, a proxy for the close.)")

    # SIDES
    sd = _first(log, "side_lean", ["home", "away"]).merge(done, on="game_id", how="inner")
    sd["cover"] = sd["margin"] + sd["mkt_spread"]               # > 0: home covered the entry spread
    sd["push"] = sd["cover"] == 0
    sd["win"] = ((sd["cover"] > 0) & (sd["side_lean"] == "home")) | ((sd["cover"] < 0) & (sd["side_lean"] == "away"))
    sl = sd[~sd["push"]]
    print("\nSIDES (entry = first logged spread; NO edge found in testing, shown for the record):")
    print(fmt_tally("all side leans", int(sl["win"].sum()), len(sl), int(sd["push"].sum()), roi=True))
    if len(sl) >= 12:
        bigs = sl[sl["pred_side_resid"].abs() >= sl["pred_side_resid"].abs().quantile(2 / 3)]
        print(fmt_tally("largest third by |pred|", int(bigs["win"].sum()), len(bigs), roi=True))

    base = log.sort_values("run_ts_utc").groupby("game_id").head(1).merge(done, on="game_id", how="inner")
    base["total_win"] = base["game_id"].map(t.set_index("game_id")["win"])
    base["move_agree"] = base["game_id"].map(mv.set_index("game_id")["agree"])
    base["side_win"] = base["game_id"].map(sd.set_index("game_id")["win"])
    base.to_csv(out_dir / "cfb_graded.csv", index=False)
    print(f"\nper-game detail: {out_dir / 'cfb_graded.csv'}")
    print("Treat results as noise until n is in the hundreds.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "cfb"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--days", type=int, default=8, help="score games kicking off within N days")
    ap.add_argument("--grade", action="store_true", help="grade logged games that have finished")
    ap.add_argument("--now", default=None, help="override current time (ISO, UTC) - for testing")
    ap.add_argument("--wide", action="store_true", help="print every numeric column instead of the plain view")
    args = ap.parse_args()
    return run_grade(args) if args.grade else run_score(args)


if __name__ == "__main__":
    sys.exit(main())
