"""Plain-English pick labels shared by the report page, the printed views and the CSVs.

A SPREAD pick is "Team +3.5" = bet that team against the point spread (not the moneyline).
A TOTAL pick is "Over 49.5" / "Under 49.5" = bet the combined score against that number.
"""
from __future__ import annotations


def _n(x) -> str:
    s = f"{x:+.2f}".rstrip("0").rstrip(".")
    return s


def spread_pick(home, away, lean, home_line, edge=None) -> str:
    """home_line is the HOME team's spread (negative = home favored), the way sportsbooks print it."""
    if lean not in ("home", "away") or home_line is None:
        return ""
    team, line = (home, home_line) if lean == "home" else (away, -home_line)
    tail = f" · model edge {abs(edge):.1f} pts" if edge is not None else ""
    return f"{team} {_n(line)} (spread){tail}"


def total_pick(lean, line, model_total=None) -> str:
    if lean not in ("over", "under") or line is None:
        return ""
    tail = f" · model {model_total:.1f}" if model_total is not None else ""
    return f"{lean.capitalize()} {line:g}{tail}"


def result_word(win, push=False) -> str:
    if push:
        return "Push"
    if win is True or win == 1:
        return "Win"
    if win is False or win == 0:
        return "Loss"
    return ""
