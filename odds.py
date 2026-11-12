"""Odds / probability helpers.

Conventions used across the project:
  * American odds: -110, +150 ...
  * Cents: a price in cents equals implied probability x 100 (50c = 50%).
  * Spread sign: model/market "fair spread" for a game is expressed as the
    HOME team's margin expectation (home - away). A home favorite of 3.5 is
    fair_spread = +3.5 in margin terms; book notation would write -3.5.
    In the nflverse schedules file, `spread_line` is POSITIVE when the home
    team is favored (points laid by home), i.e. spread_line == home margin
    the market prices in. This module keeps everything in home-margin terms
    and converts at the edges.

NFL margin SD: the standard deviation of (actual margin - expected margin)
is ~13.5 points historically; we use MARGIN_SD = 13.5 for spread->prob.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MARGIN_SD = 13.5
TOTAL_SD = 13.0  # residual SD scale for totals probability conversion


def american_to_prob(odds: float) -> float:
    """Implied probability (with vig) from American odds."""
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def prob_to_american(p: float) -> float:
    p = min(max(float(p), 0.001), 0.999)
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (100.0 / -odds if odds < 0 else odds / 100.0)


def prob_to_cents(p: float) -> float:
    return 100.0 * float(p)


def cents_to_prob(cents: float) -> float:
    return float(cents) / 100.0


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """No-vig probabilities for a two-way market from American odds."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_from_spread(home_margin_expectation: float, line: float,
                     sd: float = MARGIN_SD) -> float:
    """P(home covers) where `line` is points laid by home (nflverse
    spread_line convention, positive = home favored).

    Home covers when actual margin > line. With margin ~ N(mu, sd):
    P = 1 - Phi((line - mu)/sd). For a moneyline-style win prob, call with
    line = 0 (ties handled as half-win mass is ignored; NFL ties are rare).
    """
    return 1.0 - _norm_cdf((line - home_margin_expectation) / sd)


def win_prob_from_margin(mu: float, sd: float = MARGIN_SD) -> float:
    """P(home wins) from an expected home margin."""
    return _norm_cdf(mu / sd)


# Circa opener JSONs use full team names; map to nflverse abbreviations.
CIRCA_NAME_TO_ABBR = {
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "CHIEFS": "KC",
    "RAIDERS": "LV", "CHARGERS": "LAC", "RAMS": "LA", "DOLPHINS": "MIA",
    "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO", "GIANTS": "NYG",
    "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT", "49ERS": "SF",
    "SEAHAWKS": "SEA", "BUCS": "TB", "BUCCANEERS": "TB", "TITANS": "TEN",
    "COMMANDERS": "WAS", "WASHINGTON": "WAS",
}


def load_circa_lines(path: str | Path) -> list[dict]:
    """Load a Circa opener JSON (~/workspace/tools/nfl_week<N>_circa_openers.json).

    Returns one dict per game with nflverse team abbreviations, the market
    spread in home-margin terms (positive = home favored), the total, and
    no-vig market win probabilities from the moneylines.
    """
    raw = json.loads(Path(path).read_text())
    out = []
    for g in raw.get("games", []):
        home = CIRCA_NAME_TO_ABBR.get(str(g["home"]).upper())
        away = CIRCA_NAME_TO_ABBR.get(str(g["away"]).upper())
        if home is None or away is None:
            continue
        # Circa file: `spread` is the FAVORITE's line (negative, e.g. -3.5
        # when GB is favored at home). Convert to home-margin expectation.
        spread = float(g["spread"])
        fav_is_home = str(g.get("fav", "")).upper() == str(g["home"]).upper()
        home_margin = -spread if fav_is_home else spread
        rec = {
            "date": g.get("date"),
            "away_team": away,
            "home_team": home,
            "market_spread_home_margin": home_margin,
            "market_total": float(g["total"]) if g.get("total") is not None else None,
            "ml_away": g.get("ml_away"),
            "ml_home": g.get("ml_home"),
        }
        if g.get("ml_home") is not None and g.get("ml_away") is not None:
            ph, pa = devig_two_way(g["ml_home"], g["ml_away"])
            rec["market_prob_home"] = ph
            rec["market_prob_away"] = pa
        out.append(rec)
    return out
