# NFL / CFB tracker (runs on GitHub Actions)

Logs model leans, Novig prices and line moves on a schedule, and grades them. Nothing here has shown an edge yet:
these are leans to LOG and GRADE, not bets.

## Setup (about 10 minutes)
1. GitHub -> New repository -> name it e.g. `bet-tracker` -> **Private** -> Create.
2. Upload everything in this folder to the repo (Add file -> Upload files; drag the contents in, including the
   hidden `.github` folder. On Mac press Cmd+Shift+. in Finder to show hidden folders). If `.github` won't upload,
   use Add file -> Create new file, type `.github/workflows/novig-snapshot.yml` as the name, and paste each workflow.
3. Settings -> Secrets and variables -> Actions -> New repository secret: name `CFBD_API_KEY`, paste your NEW CFBD key ONCE
   (no line break). Novig needs no key.
4. Settings -> Actions -> General -> Workflow permissions -> choose **Read and write permissions** -> Save
   (the bot needs this to save results back to the repo).
5. Put your existing logs from Colab into the `outputs/` folder (Add file -> Upload files):
   `cfb_log.csv`, `nfl_log.csv`, `novig_nfl_games.csv`, `novig_nfl_quotes.csv`.
6. Test each workflow once: Actions tab -> pick the workflow -> Run workflow. Order: `novig-snapshot`, `nfl-scan`, `cfb-scan`.
   The first `cfb-scan` is slow (it downloads 2019-2026, about 70 CFBD calls); later runs only refresh 2026 (8 calls).

## What runs when (times are UTC; Pacific = UTC-8 now, UTC-7 in summer)
| Workflow | Schedule | Does |
|---|---|---|
| novig-snapshot | every 30 min Thu-Mon, plus Tue/Wed 18:00 | saves Novig implied spread/total/moneyline per game |
| novig-ladder | Thu 22:30, Sun 16:30 | saves the full price ladder (large file) |
| nfl-scan | Wed 18:00, Thu 21:00, Sun 13:00 and 16:30, Mon 21:00 | logs NFL leans + current line |
| cfb-scan | Wed 20:00, Fri 20:00, Sat 14:00 and 15:30 | logs CFB leans |
| grade | Tue 15:00 | grades everything -> `outputs/grade_report.txt` |

GitHub may start a scheduled run several minutes late; that is normal. Edit the `cron:` lines to change times.

## Results
Everything lands in `outputs/` (open the files on github.com, or Code -> Download ZIP). Read `grade_report.txt` on Tuesdays.

## Limits
- Free private repos get 2,000 Actions minutes a month. The 30-minute snapshot is about 1,000 of them. If you run short,
  change `*/30` to `0 *` (hourly) in `novig-snapshot.yml`.
- If Novig refuses GitHub's servers (the snapshot job fails with an HTTP error), tell Claude. The fix is to run it from your own computer instead.
- Keys: only `CFBD_API_KEY` lives in GitHub, as a protected secret. Never paste keys into files or chat.
