# NFL Edge

A weekly NFL betting model. For every bet on the board it gives you:

- **Hit %**: how often the bet should win.
- **Break-even %**: how often it *needs* to win at that price.
- **Edge / EV**: the gap between the two, and the expected profit per $1.
- **Stake**: a conservative bet size for your bankroll (quarter-Kelly, capped at 2%).

It covers spreads, moneylines, totals, 6-point teasers and player props (passing, rushing
and receiving yards, receptions, pass TDs, anytime TD).

## What the backtest says

Full numbers are in `output/backtest_report.txt`. Re-run it any time with `python backtest.py`.

1. **The stats model doesn't beat the closing line.** From 2018 to 2026 (2,275 games) the
   market missed the final margin by 9.8 points on average and the model by 10.3. When the
   model disagreed with the line by 4+ points, betting its side won 47.8%, and you need 52.4%
   to break even at -110. The tool therefore prices every bet off the market, not its own
   spread. It still shows the model's line on the game board, but for information only.
2. **The hit % numbers are accurate.** Games priced at 75% won 74.5%. "Wong" teaser legs
   priced at 74.4% won 74.4%.
3. **A high hit % doesn't make a bet good.** Moneyline favorites of 80%+ won 85.5% of the
   time and still lost 3.2 cents per dollar. The price already charges you for that
   certainty.
4. **Prop projections beat a season-average baseline**, e.g. 63.8 vs 67.5 yards missed on
   QB passing yards. Their probabilities are well calibrated. Real historical prop lines
   aren't available for free, though, so profit against real books can't be backtested.
   Track your results with `grade.py` (below).

**Where real edges come from:**

- **Line shopping.** One book is often out of line with the rest. With an Odds API key the
  tool compares every book against a sharp-book consensus and flags the stragglers.
- **Half-points around 3 and 7.** The probability engine accounts for NFL "key numbers", so
  it knows +3.5 at -120 can be better than +3 at -105.
- **Teasers through 3 and 7.** They're close to break-even at -120 and worse at -130.
- **Props**, where books are softer than on sides and totals.

## Setup (one time)

You need Python 3.10 or newer.

```bash
cd nfl_edge
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Next, get a free key at https://the-odds-api.com (500 credits a month) and set it:

```bash
export ODDS_API_KEY=your_key_here  # Windows: set ODDS_API_KEY=your_key_here
```

The first run downloads about 230 MB of free nflverse data, which takes about a minute.
After that, only the current season is refreshed.

## Your sportsbooks

Bets are only recommended at the books listed in `MY_BOOKS` in `edge/config.py`. It's set to
`["draftkings"]`. Every other book, including ones you can't use like Pinnacle, is still used
to work out the fair price. You can override it for one run with
`--books draftkings,fanduel`, or use `--books all` to see every book.

## Weekly use

```bash
python picks.py                          # game board, +EV bets, teaser legs
python picks.py --props                  # add player props (uses API credits)
python picks.py --props --props-games IND,PIT   # props for specific games only
python picks.py --min-hit 0.60           # list every bet that hits 60%+
python picks.py --bankroll 500 --log     # size stakes for $500 and save picks
python grade.py                          # after the games: record, ROI, CLV
```

**Credit budget:** each run asks for 10 named books, which the API bills as one region.
Game lines cost 3 credits per run. Props cost 6 credits per game, about 96 for a full week.
On the free 500 a month, that's one full props run per week plus several game-line runs.
Use `--props-games` to pull fewer games. The output prints your remaining credits.

**No API key?** The tool still runs, using the consensus lines in the nflverse data, but it
won't find much. A single price already includes the vig, so line shopping is where edges
show up. For props, type in lines from your sportsbook using
`prop_lines_TEMPLATE.csv` (its rows are examples; replace them) and run:

```bash
python picks.py --prop-lines my_lines.csv
```

Market names you can use: `pass_yds, pass_tds, completions, pass_att, rush_yds, rush_att,
rec_yds, receptions, anytime_td`.

## Reading the output

| Column | Meaning |
|---|---|
| hit_pct | Chance the bet wins (pushes excluded) |
| breakeven_pct | Win rate needed at that price (-110 is 52.4%) |
| edge_pct | hit_pct minus breakeven_pct |
| ev_per_$1 | Expected profit per $1 bet. +0.03 means 3 cents per dollar long-run |
| stake | Suggested bet: quarter-Kelly, never more than 2% of bankroll |
| model_proj / market_proj | Props: the model's projection vs what the books' prices imply |
| news_risk | Model and books disagree by 25%+, usually injury or role news. Skipped |

A bet is flagged when its edge is at least 3% **and** its EV is at least +2%. You can
change both with `--min-edge` and `--min-ev`, or edit the defaults in `edge/config.py`.

## How it works

- **Probability engine (`edge/game_model.py`).** Each book's no-vig price is converted into
  the expected margin and total the market is pricing in. Outcomes follow a distribution
  fitted to real NFL results, which gives extra weight to margins of 3, 7, 10 and so on.
  That's what lets it compare different spreads and price teaser legs.
- **Stats model (`edge/ratings.py`).** Opponent-adjusted, recency-weighted EPA ratings built
  from play-by-play data, with a ridge regression on top. It's allowed to move the market
  number only by the weight it earned out of sample, which is currently zero. That weight is
  re-learned every run, so if the model starts adding value it will be used automatically.
- **Props (`edge/props.py`).** Volume (targets, carries, attempts) comes from recent games.
  Efficiency (yards per target, TD rate) is pulled toward the position average. Both are
  adjusted for the opponent's defense and the team's implied points from the betting line.
  A fitted distribution then turns each projection into P(over). The final projection is
  40% model and 60% sportsbook consensus (`PROP_MODEL_WEIGHT` in `edge/config.py`).
- **Line shopping (`edge/evaluate.py`).** Fair value comes from sharp books (Pinnacle,
  Circa, LowVig) when they're available, otherwise from the median across books. Every
  book's price is then scored against it.

## Limits

- **The model can't see injuries, weather or news.** It lists anyone who played in the
  team's last two games, so check the injury report before betting a prop. The
  `news_risk` flag catches the big mismatches.
- **Lines move.** Re-run close to when you bet.
- **Prop profit is untested.** `PROP_MODEL_WEIGHT` is a judgment call. After 100+ graded
  props, compare your win rate against the model's expected rate in `grade.py`. If you're
  consistently below it, lower the weight.
- **Parlays.** None of this makes parlays a good idea; the vig compounds with every leg.
- **Variance.** Even a genuine 3% edge loses money over a 50-bet stretch fairly often.
  Closing line value (CLV) is the best early sign of whether you're actually beating the
  market.

Bet only what you can afford to lose. For help: 1-800-GAMBLER.

## Tests

```bash
python tests/test_core.py
```

The tests include a mocked multi-book Odds API response, so the live-odds path is checked
without a key.
