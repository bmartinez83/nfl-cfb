"""NFL weekly scorer + logger + grader (mirrors cfb_weekly.py).

  python src/nfl_weekly.py                 # score the next --days days, append to the log
  python src/nfl_weekly.py --refresh       # re-download nflverse data first (needs internet)
  python src/nfl_weekly.py --grade         # grade logged leans against final scores (use --refresh first)

What it does
  * Fits the existing NFL model (train.fit_bundle) on every played game, prices upcoming games,
    and compares to the CURRENT market line in the nflverse schedules file.
  * Every run appends a snapshot row per game ONLY when the market line/total/moneyline changed
    since the last snapshot, so the log doubles as your own line-history (open -> latest).
  * --grade scores each lean at its FIRST logged line (what you could have bet) and reports CLV
    versus the LAST logged line (a proxy for the close; only meaningful if you ran a snapshot
    shortly before kickoff).

Honest status: testing found NO edge from this model against the closing line. These are leans to
LOG and GRADE, not bets. Hundreds of graded games are needed before any record means anything.

Spread convention (nflverse): spread_line is POSITIVE when the HOME team is favored (home margin
the market expects). Home covers when actual margin > spread_line.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

LOG_COLS = ["run_ts_utc", "season", "week", "game_id", "gameday", "gametime", "away", "home",
            "spread_line", "total_line", "home_ml", "away_ml",
            "model_margin", "model_total", "model_prob_home",
            "gap_spread", "gap_total", "spread_lean", "total_lean", "inj_hold", "inj_note"]
LINE_COLS = ["spread_line", "total_line", "home_ml", "away_ml", "inj_note"]


def _now(args_now: str | None) -> datetime:
    if args_now:
        d = datetime.fromisoformat(args_now)
        return d.replace(tzinfo=ET) if d.tzinfo is None else d
    return datetime.now(timezone.utc)


def _load_log(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame(columns=LOG_COLS)


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):      # text columns (injury note): blank == missing
        return (a if isinstance(a, str) else "") == (b if isinstance(b, str) else "")
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-9


def run_refresh() -> None:
    import ingest
    rc = ingest.main()
    if rc != 0:
        raise SystemExit("data refresh failed (schedules file is required)")


def run_score(args) -> None:
    from features import build_features
    from train import fit_bundle, predict_frame

    now = _now(getattr(args, "now", None))
    today = now.astimezone(ET).date()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = build_features()
    if "gametime" not in df.columns:
        gt = pd.read_csv(Path(args.data) / "games.csv", usecols=["game_id", "gametime"])
        df = df.merge(gt, on="game_id", how="left")
    played = df[df["home_score"].notna()]
    upcoming = df[df["home_score"].isna()].copy()
    upcoming["gd"] = pd.to_datetime(upcoming["gameday"]).dt.date

    stale = upcoming[upcoming["gd"] < today - timedelta(days=1)]
    if len(stale):
        print(f"WARNING: STALE DATA - {len(stale)} game(s) dated before {today} have no final score yet "
              f"(e.g. {stale.iloc[0].away_team} @ {stale.iloc[0].home_team}). Run with --refresh.")
    upcoming = upcoming[(upcoming["gd"] >= today) & (upcoming["gd"] <= today + timedelta(days=args.days))]
    upcoming = upcoming[upcoming["spread_line"].notna() & upcoming["total_line"].notna()]
    if upcoming.empty:
        print(f"No upcoming games with lines in the next {args.days} days (today {today}).")
        return

    bundle = fit_bundle(played)
    pred = predict_frame(bundle, upcoming)
    print(f"data as of: last played game {played['gameday'].max()} | today (ET) {today}")
    print(f"model trained on {bundle['n_train']} games through {bundle['trained_through']}")

    pred["gap_spread"] = pred["model_margin"] - pred["spread_line"]
    pred["gap_total"] = pred["model_total"] - pred["total_line"]
    pred["spread_lean"] = np.where(pred["gap_spread"] > 0, "home", "away")
    pred["total_lean"] = np.where(pred["gap_total"] > 0, "over", "under")
    # UNCALIBRATED expected value if the model's win chance were right (-110 both sides). The model
    # overstates its confidence (its biggest disagreements won ~50% in testing), so this is what the
    # model BELIEVES, not a real expectation.
    from math import erf, sqrt

    def _ev(gap, sd):
        p_pos = gap.map(lambda x: 0.5 * (1 + erf(x / (sd * sqrt(2)))))
        p = np.where(gap > 0, p_pos, 1 - p_pos)
        return (100 * (p * (100 / 110) - (1 - p))).round(1)

    pred["ev_spread_pct"] = _ev(pred["gap_spread"], bundle["margin_sd"])
    pred["ev_total_pct"] = _ev(pred["gap_total"], bundle["total_sd"])
    import injuries
    pred = injuries.flag_games(pred, int(pred["season"].iloc[0]))
    pred = pred.sort_values(["gameday", "gametime", "game_id"])

    rows = pd.DataFrame({
        "run_ts_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "season": pred["season"].astype(int), "week": pred["week"].astype(int),
        "game_id": pred["game_id"], "gameday": pred["gameday"], "gametime": pred.get("gametime"),
        "away": pred["away_team"], "home": pred["home_team"],
        "spread_line": pred["spread_line"], "total_line": pred["total_line"],
        "home_ml": pred.get("home_moneyline"), "away_ml": pred.get("away_moneyline"),
        "model_margin": pred["model_margin"].round(2), "model_total": pred["model_total"].round(2),
        "model_prob_home": pred["model_prob_home"].round(4),
        "gap_spread": pred["gap_spread"].round(2), "gap_total": pred["gap_total"].round(2),
        "spread_lean": pred["spread_lean"], "total_lean": pred["total_lean"],
        "inj_hold": pred["inj_hold"], "inj_note": pred["inj_note"],
    })[LOG_COLS]

    log_path = out / "nfl_log.csv"
    old = _load_log(log_path)
    last = old.groupby("game_id").tail(1).set_index("game_id") if len(old) else pd.DataFrame()
    keep = []
    for r in rows.itertuples(index=False):
        if r.game_id in last.index:
            p = last.loc[r.game_id]
            if all(_same(getattr(r, c), p[c]) for c in LINE_COLS):
                continue
        keep.append(r._asdict())
    if keep:
        new = pd.DataFrame(keep)[LOG_COLS]
        pd.concat([old, new], ignore_index=True).to_csv(log_path, index=False)
    print(f"log: {len(keep)} new snapshot rows (lines changed or first seen) -> {log_path} "
          f"(total rows {len(old) + len(keep)})")

    view = pred.assign(
        kickoff=pred["gameday"].astype(str) + " " + pred["gametime"].fillna("").astype(str),
        game=pred["away_team"] + " @ " + pred["home_team"])
    cols = ["kickoff", "game", "spread_line", "model_margin", "gap_spread", "spread_lean",
            "ev_spread_pct", "total_line", "model_total", "gap_total", "total_lean", "ev_total_pct",
            "inj_hold", "inj_note"]
    pd.set_option("display.width", 250)
    print("\n" + view[cols].round(1).to_string(index=False))
    view[cols].round(2).to_csv(out / "nfl_weekly_predictions.csv", index=False)
    print("\nspread_line is the HOME margin the market expects (positive = home favored). "
          "gap = model minus market.\nev_*_pct = UNCALIBRATED expected value per 100 risked at -110 IF the "
          "model's win chance were right.\nThe model overstates its confidence, so large positive numbers are "
          "NOT real.\ninj_hold=True: a QB is Out/Doubtful/Questionable or 3+ starters are out - the model's ratings "
          "may be stale, so check that lean by hand.\nNO edge was found in testing: these are leans to LOG and grade, not bets.")


def _wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _tally(name: str, win: pd.Series, push: pd.Series) -> None:
    w, l, p = int(win.sum()), int((~win & ~push).sum()), int(push.sum())
    n = w + l
    if n == 0:
        print(f"{name}: no graded games yet")
        return
    lo, hi = _wilson(w, n)
    roi = (w * (100 / 110) - l) / n * 100
    print(f"{name}: {w}-{l}-{p}  win% {100 * w / n:.1f} (95% CI {100 * lo:.1f}-{100 * hi:.1f}, "
          f"breakeven 52.4)  ROI at -110 {roi:+.1f}%")


def run_grade(args) -> None:
    out = Path(args.out)
    log = _load_log(out / "nfl_log.csv")
    if log.empty:
        print("log is empty - nothing to grade")
        return
    games = pd.read_csv(Path(args.data) / "games.csv", low_memory=False)
    games = games[["game_id", "home_score", "away_score"]]
    log = log.sort_values(["game_id", "run_ts_utc"])
    first = log.groupby("game_id").head(1).set_index("game_id")
    lastr = log.groupby("game_id").tail(1).set_index("game_id")
    g = first.join(lastr[["spread_line", "total_line"]].rename(
        columns={"spread_line": "close_spread", "total_line": "close_total"}))
    g = g.reset_index().merge(games, on="game_id", how="left")
    fin = g[g["home_score"].notna()].copy()
    print(f"games logged: {len(g)} | finished and gradeable: {len(fin)} | pending: {len(g) - len(fin)}")
    if fin.empty:
        return
    fin["margin"] = fin["home_score"] - fin["away_score"]
    fin["pts"] = fin["home_score"] + fin["away_score"]

    cover = fin["margin"] - fin["spread_line"]
    push = cover == 0
    win = ((cover > 0) & (fin["spread_lean"] == "home")) | ((cover < 0) & (fin["spread_lean"] == "away"))
    print()
    _tally("SPREAD leans (ATS at first logged line)", win, push)
    diff = fin["pts"] - fin["total_line"]
    push_t = diff == 0
    win_t = ((diff > 0) & (fin["total_lean"] == "over")) | ((diff < 0) & (fin["total_lean"] == "under"))
    _tally("TOTAL leans (at first logged line)", win_t, push_t)

    sp_move = fin["close_spread"] - fin["spread_line"]
    clv_s = np.where(fin["spread_lean"] == "home", sp_move, -sp_move)
    tt_move = fin["close_total"] - fin["total_line"]
    clv_t = np.where(fin["total_lean"] == "over", tt_move, -tt_move)
    for name, clv in (("spread", clv_s), ("total", clv_t)):
        moved = clv != 0
        print(f"CLV {name}: avg {np.mean(clv):+.2f} pts vs last logged line | "
              f"line moved in your favor {int((clv > 0).sum())}, against {int((clv < 0).sum())}, "
              f"unchanged {int((~moved).sum())}")
    print("\n(CLV uses the LAST logged line as the close. Run a snapshot shortly before kickoff for it "
          "to mean anything.)")
    fin.drop(columns=["pts"]).to_csv(out / "nfl_graded.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--now", default=None, help="ISO time (ET if naive) for testing")
    args = ap.parse_args()
    if args.refresh:
        run_refresh()
    (run_grade if args.grade else run_score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
