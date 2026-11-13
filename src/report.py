"""Builds a shareable one-page HTML report from the CSVs in outputs/.

  python src/report.py [--out DIR] [--docs DIR] [--fragment FILE]

Writes docs/index.html (self-contained; works with GitHub Pages and prints cleanly to PDF from any browser).
Reads: nfl_weekly_predictions.csv, cfb_weekly_predictions.csv, novig_nfl_games.csv, grade_report.txt (each optional).
Everything on the page is a lean to LOG and GRADE. No edge has been proven.
"""
from __future__ import annotations

import argparse
import csv
import html
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
E = html.escape


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sgn(x, d=1):
    return "" if x is None else f"{x:+.{d}f}"


# ------------------------------------------------------------------ NFL
def nfl_rows(rows):
    out = []
    for r in rows:
        away, home = [s.strip() for s in r["game"].split("@")]
        sp, mm = num(r["spread_line"]), num(r["model_margin"])
        gap = num(r["gap_spread"])
        tl, mt, gt = num(r["total_line"]), num(r["model_total"]), num(r["gap_total"])
        side = ""
        if gap is not None and abs(gap) >= 0.5:
            side = f"{home if r['spread_lean'] == 'home' else away} +{abs(gap):.1f}"
        tot = ""
        if gt is not None and abs(gt) >= 0.5:
            tot = f"{r['total_lean'].upper()} {tl:.1f}→{mt:.1f}"
        hold = str(r.get("inj_hold", "")).lower() == "true"
        note = r.get("inj_note", "") or ""
        wx = r.get("wx_note", "") or ""
        wxflag = str(r.get("wx_flag", "")).lower() == "true"
        rs = r.get("rest_note", "") or ""
        out.append((r["kickoff"], f"{away} @ {home}", f"{home} {-sp:+.1f}" if sp is not None else "",
                    side, tot, max(x for x in (num(r["ev_spread_pct"]), num(r["ev_total_pct"]), -999) if x is not None),
                    hold, note, wx, wxflag, rs))
    return out


def nfl_section(rows):
    if not rows:
        return "<p class='muted'>No NFL games with lines right now.</p>"
    body = []
    for k, g, line, side, tot, ev, hold, note, wx, wxflag, rs in nfl_rows(rows):
        flag = ("<span class='pill warn'>CHECK INJURIES</span> " if hold else "") + \
               ("<span class='pill warn'>WEATHER</span>" if wxflag else "")
        n = (f"<div class='note'>{E(note)}</div>" if note else "") + \
            (f"<div class='note'>Weather: {E(wx)}</div>" if wx and not any(k in wx for k in ("no weather effect", "unavailable", "not available", "not checked")) else "") + \
            (f"<div class='note'>{E(rs)}</div>" if rs else "")
        body.append(f"<tr><td class='nowrap'>{E(k)}</td><td><b>{E(g)}</b>{n}</td><td class='nowrap'>{E(line)}</td>"
                    f"<td>{E(side) or '<span class=muted>—</span>'}</td>"
                    f"<td class='nowrap'>{E(tot) or '<span class=muted>—</span>'}</td>"
                    f"<td class='num'>{ev:+.0f}%</td><td>{flag}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Kickoff (ET)</th><th>Game</th><th>Home line</th>"
            "<th>Spread lean</th><th>Total lean</th><th class='num'>Best EV*</th><th></th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


# ------------------------------------------------------------------ CFB
def cfb_section(rows):
    if not rows:
        return "<p class='muted'>No CFB games with lines right now.</p>"
    body = []
    for r in rows:
        away, home = [s.strip() for s in r["game"].split("@")]
        sp, so = num(r["mkt_spread"]), num(r["sp_open"])
        line = f"{home} {sp:+.1f}" + (f" (open {so:+.1f})" if so is not None else "") if sp is not None else ""
        ps = num(r["pred_side_resid"])
        side = f"{home if r['side_lean'] == 'home' else away} +{abs(ps):.1f}" if ps is not None and abs(ps) >= 0.5 else ""
        pt, mt, ft = num(r["pred_total_resid"]), num(r["mkt_total"]), num(r["fair_total"])
        tot = f"{r['total_lean'].upper()} {mt:.1f}→{ft:.1f}" if pt is not None and abs(pt) >= 0.5 else ""
        rm = num(r["remaining_move"])
        mv = f"toward {home if r['move_lean'] == 'home' else away} {abs(rm):.1f}" if rm is not None and abs(rm) >= 0.5 else ""
        evs = [x for x in (num(r["ev_side_pct"]), num(r["ev_total_pct"])) if x is not None]
        ev = f"{max(evs):+.0f}%" if evs else ""
        news = "<span class='pill warn'>CHECK NEWS</span>" if r.get("check_news") else ""
        wxflag = str(r.get("wx_flag", "")).lower() == "true"
        if wxflag:
            news += " <span class='pill warn'>WEATHER</span>"
        wxn = f"<div class='note'>Weather: {E(r.get('wx_note', ''))}</div>" if wxflag else ""
        rs = r.get("rest_note", "") or ""
        if rs:
            wxn += f"<div class='note'>{E(rs)}</div>"
        dash = "<span class=muted>—</span>"
        body.append(f"<tr><td class='nowrap'>{E(r['kickoff_pt'])}</td><td><b>{E(away)} @ {E(home)}</b>{wxn}</td>"
                    f"<td class='nowrap'>{E(line)}</td><td>{E(side) or dash}</td><td class='nowrap'>{E(tot) or dash}</td>"
                    f"<td class='nowrap'>{E(mv) or dash}</td><td class='num'>{ev}</td><td>{news}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Kickoff (PT)</th><th>Game</th><th>Home line</th>"
            "<th>Side lean</th><th>Total lean</th><th>Line move</th><th class='num'>Best EV*</th><th></th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


# ------------------------------------------------------------------ Novig
def novig_section(rows):
    latest = {}
    for r in rows:                      # file is append-only, so the last row per event is the newest
        latest[r["event_id"]] = r
    if not latest:
        return ""
    body = []
    for r in sorted(latest.values(), key=lambda r: num(r["starts_ts"]) or 0):
        sp, ns, gs = num(r["home_spread_line"]), num(r["nflverse_spread_line"]), num(r["gap_spread"])
        tl, nt = num(r["total_line"]), num(r["nflverse_total_line"])
        ml = num(r["ml_home_prob"])
        dash = "<span class=muted>—</span>"
        body.append(f"<tr><td><b>{E(r['away'])} @ {E(r['home'])}</b></td>"
                    f"<td class='num'>{'' if ml is None else f'{ml * 100:.0f}%'}</td>"
                    f"<td class='num'>{sgn(sp) or dash}</td><td class='num'>{sgn(ns) or dash}</td>"
                    f"<td class='num'>{tl:.1f}</td><td class='num'>{'' if nt is None else f'{nt:.1f}'}</td></tr>"
                    if tl is not None else
                    f"<tr><td><b>{E(r['away'])} @ {E(r['home'])}</b></td><td class='num'>—</td><td></td><td></td><td></td><td></td></tr>")
    ts = max(r["ts_utc"] for r in latest.values())
    return ("<section><h2>Novig exchange prices</h2><p class='muted'>Latest snapshot "
            f"{E(ts)} UTC. Spread is the home margin the market expects (positive = home favored).</p>"
            "<div class='scroll'><table><thead><tr><th>Game</th><th class='num'>Home win</th><th class='num'>Novig spread</th>"
            "<th class='num'>Schedule spread</th><th class='num'>Novig total</th><th class='num'>Schedule total</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div></section>")


CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--fg:#14181f;--muted:#5d6675;--line:#dfe3ea;--head:#eef1f5;--accent:#0b5fd6;
--warn-bg:#fff1d6;--warn-fg:#7a4a00;--font:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1318;--panel:#171c23;--fg:#e8ecf2;--muted:#9aa4b4;
--line:#2a313c;--head:#1e252e;--accent:#6aa5ff;--warn-bg:#3d2e0a;--warn-fg:#ffd27a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1318;--panel:#171c23;--fg:#e8ecf2;--muted:#9aa4b4;--line:#2a313c;--head:#1e252e;
--accent:#6aa5ff;--warn-bg:#3d2e0a;--warn-fg:#ffd27a;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--font);margin:0;
padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:17px;margin:0 0 8px}
.muted{color:var(--muted)}
.banner{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;min-width:0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{background:var(--head);font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.num{text-align:right}.nowrap{white-space:nowrap}
.note{font-size:12px;color:var(--muted);margin-top:2px;max-width:46ch}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 7px;border-radius:999px;white-space:nowrap}
.warn{background:var(--warn-bg);color:var(--warn-fg)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;overflow-x:auto;margin:0;font-size:12.5px}
footer{font-size:12.5px;color:var(--muted);max-width:90ch}
@media print{@page{size:landscape;margin:12mm}body{background:#fff;color:#000;padding:0}
.scroll,.banner,pre{border-color:#bbb;background:#fff}th{background:#eee;color:#333}tr{break-inside:avoid}}
"""


def build(out: Path) -> str:
    nfl = read(out / "nfl_weekly_predictions.csv")
    cfb = read(out / "cfb_weekly_predictions.csv")
    nov = read(out / "novig_nfl_games.csv")
    gp = out / "grade_report.txt"
    grade = gp.read_text() if gp.exists() else ""
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%a %b %d, %I:%M %p PT")
    parts = [
        f"<header><h1>Weekly Lean Report</h1><p class='muted' style='margin:0'>Updated {E(now)}. Lines come from the latest scheduled scan.</p></header>",
        "<div class='banner'><b>These are leans to log and grade, not bets.</b> Testing found no proven edge. "
        "Check any game marked CHECK INJURIES, WEATHER or CHECK NEWS before reading anything into its lean. Weather is context only: it did not predict totals out of sample.</div>",
        f"<section><h2>NFL ({len(nfl)} games)</h2>{nfl_section(nfl)}</section>",
        f"<section><h2>College football ({len(cfb)} games)</h2>{cfb_section(cfb)}</section>",
        novig_section(nov),
    ]
    if grade.strip():
        parts.append(f"<section><h2>Latest grade report</h2><pre>{E(grade.strip())}</pre></section>")
    parts.append("<footer><p>How to read: a spread lean names the team the model likes and by how many points. "
                 "A total lean shows the market total → the model's total. Edges under 0.5 points show as a dash (no lean). "
                 "*Best EV is the model's own uncalibrated expected return per 100 risked at -110 if its confidence were right. "
                 "The model overstates its confidence, so use it only to rank leans.</p></footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta name='claude:viewport-fit-sides' content='cover'>"
            "<title>Weekly Lean Report</title>"
            "<link rel='preconnect' href='https://fonts.googleapis.com'>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap'>"
            f"<style>{CSS}</style></head><body><main>" + "".join(parts) + "</main></body></html>")


def fragment(doc: str) -> str:
    """Same page minus the html/head/body wrapper (the Artifact publisher adds its own)."""
    head_end = doc.index("</head>")
    head = doc[doc.index("<title>"):head_end]
    body = doc[doc.index("<body>") + 6:doc.index("</body>")]
    return "<meta name='claude:viewport-fit-sides' content='cover'>\n" + head + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--docs", default=str(ROOT / "docs"))
    ap.add_argument("--fragment", default=None, help="also write the wrapper-free version to this file")
    a = ap.parse_args()
    doc = build(Path(a.out))
    docs = Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.html").write_text(doc)
    if a.fragment:
        Path(a.fragment).write_text(fragment(doc))
    print(f"wrote {docs / 'index.html'} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
