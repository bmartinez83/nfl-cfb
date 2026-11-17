"""Builds a shareable one-page HTML report from the CSVs in outputs/.

  python src/report.py [--out DIR] [--docs DIR] [--fragment FILE]

Writes docs/index.html (self-contained; works with GitHub Pages and prints cleanly to PDF from any browser).
Reads: nfl_weekly_predictions.csv, cfb_weekly_predictions.csv, novig_nfl_games.csv, grade_report.txt (each optional).
Everything on the page is a lean to LOG and GRADE. No edge has been proven.
"""
from __future__ import annotations

import re
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

REC = re.compile(r"^\s*(?P<label>.+?):?\s+(?P<rec>\d+-\d+(?:-\d+)?)\s+(?:=\s*(?P<pct>[\d.]+)%|win%\s+(?P<pct2>[\d.]+))\s+"
                 r"\(95% CI (?P<lo>[\d.]+)-(?P<hi>[\d.]+)%?(?:,\s*n=(?P<n>\d+))?(?:,\s*breakeven (?P<be>[\d.]+))?\)"
                 r"\s*(?:ROI at -110:?\s*(?P<roi>[+-]?[\d.]+)%)?")
CLV = re.compile(r"^CLV (?P<k>\w+): avg (?P<avg>[+-][\d.]+) pts.*in your favor (?P<f>\d+), against (?P<a>\d+), unchanged (?P<u>\d+)")
HEAD = re.compile(r"logged games: (\d+) \| finished and gradeable: (\d+) \| (?:still )?pending: (\d+)")
SECTION = {"TOTALS": "Totals (over / under)", "SIDES": "Sides (point spread)",
           "LINE MOVES": "Did the line move the way we leaned?", "SPREAD": "Point-spread leans", "TOTAL": "Total (over / under) leans"}
BREAKEVEN = 52.4


def _plain(label: str) -> str:
    l = label.strip()
    if l.startswith("largest third"):
        return "Strongest third (biggest model-vs-market gap)"
    if l.startswith("lean matched the move"):
        return "Line later moved toward our lean"
    if l.startswith(("all leans", "all side leans")) or l.startswith(("SPREAD leans", "TOTAL leans")):
        return "All leans"
    return l[:1].upper() + l[1:]


def _read(n: int, lo: float, hi: float) -> str:
    if n < 100:
        return "<span class='pill warn'>Too few games to tell</span>"
    if lo > BREAKEVEN:
        return "<span class='pill'>Above break-even</span>"
    if hi < BREAKEVEN:
        return "<span class='pill warn'>Below break-even</span>"
    return "<span class='pill'>No clear edge</span>"


def grade_section(text: str) -> str:
    """Plain-English scoreboard built from grade_report.txt (falls back to the raw text if parsing finds nothing)."""
    blocks = re.split(r"^=== (\w+) ===\s*$", text, flags=re.M)
    out, found = [], False
    for i in range(1, len(blocks) - 1, 2):
        name = {"CFB": "College football", "NFL": "NFL"}.get(blocks[i], blocks[i])
        body = blocks[i + 1]
        h = HEAD.search(body)
        head = (f"<p class='muted' style='margin:0 0 8px'>{h[1]} leans logged · <b>{h[2]}</b> finished and graded · {h[3]} still to play</p>"
                if h else "")
        rows, sect = [], ""
        for line in body.splitlines():
            m0 = re.match(r"^([A-Z][A-Z ]+?)(?: leans)?\b", line)
            if m0 and m0[1].strip() in SECTION and not REC.match(line) or re.match(r"^(TOTALS|SIDES|LINE MOVES)\b", line):
                sect = SECTION.get(re.match(r"^[A-Z ]+?(?=\s*\(|$)", line)[0].strip(), sect) if re.match(r"^[A-Z ]+", line) else sect
            if line.startswith(("SPREAD leans", "TOTAL leans")):
                sect = SECTION["SPREAD" if line.startswith("SPREAD") else "TOTAL"]
            m = REC.match(line)
            if not m:
                continue
            found = True
            w = [int(x) for x in m["rec"].split("-")]
            n = int(m["n"]) if m["n"] else w[0] + w[1]
            pct = float(m["pct"] or m["pct2"])
            roi = f"{float(m['roi']):+.0f}" if m["roi"] else "–"
            rows.append(f"<tr><td>{E(sect)}</td><td>{E(_plain(m['label']))}</td><td class='num nowrap'>{E(m['rec'])}</td>"
                        f"<td class='num'>{pct:.0f}%</td><td class='num nowrap'>{float(m['lo']):.0f}–{float(m['hi']):.0f}%</td>"
                        f"<td class='num'>{n}</td><td class='num'>{roi if roi == '–' else roi[0] + '$' + roi[1:]}</td><td>{_read(n, float(m['lo']), float(m['hi']))}</td></tr>")
        clv = [m for m in (CLV.match(l) for l in body.splitlines()) if m]
        clv_html = ""
        if clv:
            cr = "".join(f"<tr><td>{'Spread' if m['k'] == 'spread' else 'Total'}</td><td class='num'>{float(m['avg']):+.2f} pts</td>"
                         f"<td class='num'>{m['f']}</td><td class='num'>{m['a']}</td><td class='num'>{m['u']}</td></tr>" for m in clv)
            clv_html = ("<h3>Did the line move our way after we logged the lean?</h3><div class='scroll'><table><thead><tr><th>Bet</th>"
                        "<th class='num'>Average move</th><th class='num'>Moved our way</th><th class='num'>Moved against us</th>"
                        "<th class='num'>Didn't move</th></tr></thead><tbody>" + cr + "</tbody></table></div>"
                        "<p class='note'>Average move: positive = the line moved in our favor. It compares to the last line we logged, "
                        "so it only means something if a snapshot ran shortly before kickoff.</p>")
        tbl = ("<div class='scroll'><table><thead><tr><th>Bet type</th><th>Group</th><th class='num'>Record</th><th class='num'>Win %</th>"
               "<th class='num'>Likely range*</th><th class='num'>Games</th><th class='num'>Profit per $100 bet†</th><th>Read</th></tr></thead><tbody>"
               + "".join(rows) + "</tbody></table></div>") if rows else ""
        out.append(f"<h3 style='margin:16px 0 6px'>{E(name)}</h3>{head}{tbl}{clv_html}")
    if not found:
        return f"<pre>{E(text.strip())}</pre>"
    return "".join(out) + ("<p class='note' style='max-width:none'>Record = wins-losses (-pushes). Break-even at standard -110 prices is "
                           "<b>52.4%</b>. *Likely range: the span the true win rate plausibly falls in (95%); a wide range means "
                           "luck can't be ruled out. †Profit per $100 bet: flat $100 on every lean at -110. "
                           "'Too few games to tell' shows until a group has 100+ graded games.</p>")


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
        parts.append(f"<section><h2>Results so far: how the leans did</h2>{grade_section(grade)}</section>")
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
