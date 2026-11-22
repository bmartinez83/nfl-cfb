# Situational backtest: 3rd/4th down + pocket time/pressure (Nov 22, 2026)

Script: tools/situational_backtest.py (local, not yet uploaded). Data: nflverse pbp 2022-2026 wk11, NGS time-to-throw, closing spread/total. 918 games (week 5+), features use only prior games (shrunk, prior season blended).

Question: do these matchups predict (actual margin - closing spread) or (actual total - closing total)?

Result: NO usable edge.
- Every feature alone has |t| < 1.2 vs the closing spread (3rd down, 4th-down go rate, 4th conversion, pressure, time to throw).
- All five together: no t above 1.5, and 3rd down comes out wrong-signed (noise).
- Totals: 3rd-down and pressure features have |t| < 0.6.
- Out of sample (train 2022-24, test 2025-26, n=276): correlation 0.05; picks at |edge| 0.5+/1.0+/1.5+ went 48.0% / 44.1% / 40.6% vs 52.4% breakeven.
- Takeaway: the market already prices these. Fine as game-preview context, not as a model input.
Caveats: 'pressure' = sacks + QB hits per dropback (pbp proxy, not true time-to-pressure, which is paid PFF data). Time to throw is NGS team average.
