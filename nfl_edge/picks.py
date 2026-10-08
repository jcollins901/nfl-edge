#!/usr/bin/env python3
"""This week's NFL board: hit % for every bet, and the ones worth taking.

Examples
  python picks.py                         # next week's games, lines from nflverse (no key)
  python picks.py --props                 # + player props (needs ODDS_API_KEY)
  python picks.py --prop-lines my.csv     # + props you typed in from your sportsbook
  python picks.py --min-hit 0.60          # only show bets that hit 60%+ of the time
  python picks.py --bankroll 500 --log    # size stakes and save picks to bet_log.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

from edge import config, data, evaluate as E, pipeline
from edge.board import export_board
from edge.odds import fmt_american

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--week", type=int, help="NFL week (default: next unplayed week)")
    ap.add_argument("--season", type=int)
    ap.add_argument("--no-update", action="store_true", help="skip downloading fresh data")
    ap.add_argument("--props", action="store_true", help="pull player props from The Odds API")
    ap.add_argument("--props-games", help="comma list of teams to pull props for, e.g. IND,PIT"
                    " (saves API credits)")
    ap.add_argument("--prop-lines", help="CSV of prop lines you entered yourself")
    ap.add_argument("--min-hit", type=float, default=0.0,
                    help="only list bets with at least this hit probability (e.g. 0.6)")
    ap.add_argument("--min-edge", type=float, default=config.MIN_EDGE)
    ap.add_argument("--min-ev", type=float, default=config.MIN_EV)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--teaser-price", type=float, default=-120,
                    help="your book's 2-team 6-point teaser price")
    ap.add_argument("--books", default=",".join(config.MY_BOOKS),
                    help="books you can bet at, e.g. draftkings,fanduel (default: MY_BOOKS in "
                    "edge/config.py). 'all' = every book. Fair value always uses every book.")
    ap.add_argument("--log", action="store_true", help="append the recommended bets to bet_log.csv")
    a = ap.parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    my_books = (None if a.books.lower() == "all"
                else [b.strip().lower() for b in a.books.split(",") if b.strip()])

    if not a.no_update:
        data.update()
    games = data.load_games()
    upcoming, week = pipeline.next_week(games, a.season, a.week)
    season = int(upcoming.season.iloc[0])
    print(f"\n=== NFL {season} Week {week} ===")

    pbp = data.load_pbp()
    gm, feats, _ = pipeline.game_model(games, pbp)
    wk = gm.predict(feats[feats.game_id.isin(upcoming.game_id)])

    # ---------------- odds
    use_api = bool(__import__("os").environ.get("ODDS_API_KEY"))
    api = None
    if use_api:
        from edge.oddsapi import OddsAPI
        api = OddsAPI()
        odds = api.game_odds()
        print(f"(Odds API credits remaining: {api.remaining})")
        odds = odds[odds.apply(lambda r: ((upcoming.home_team == r.home_team) &
                                          (upcoming.away_team == r.away_team)).any(), axis=1)]
        src = (f"The Odds API ({odds.book.nunique()} books for fair value; recommending "
               f"{'all books' if my_books is None else ', '.join(my_books)} only)")
        if my_books and not odds.book.isin(my_books).any():
            print(f"WARNING: none of {my_books} found. Books in this pull: "
                  f"{', '.join(sorted(odds.book.unique()))}")
    else:
        odds = E.games_to_odds_rows(upcoming[~upcoming.played])
        my_books = None  # the consensus line isn't a real book
        src = "nflverse consensus lines (one price per bet; add ODDS_API_KEY to shop books)"
    print(f"Odds source: {src}")

    bets, mk = E.evaluate_games(odds, wk, gm, a.bankroll, my_books)

    # ---------------- game board
    board = mk.merge(upcoming[["game_id", "gameday"]], on="game_id", how="left")
    board["Game"] = board.away_team + " @ " + board.home_team
    board["Market line"] = board.mkt_line.map(lambda x: f"{x:+g}")
    board["Model line"] = (-board.model_margin).map(lambda x: f"{x:+.1f}")
    board["Market total"] = board.mkt_total_line.map("{:g}".format)
    board["Model total"] = board.model_total.map("{:.1f}".format)
    ml = bets[bets.type == "Moneyline"].copy()
    fav = ml.sort_values("hit_pct", ascending=False).drop_duplicates("game")
    board = board.merge(fav[["game", "side", "hit_pct"]].rename(
        columns={"game": "Game", "side": "Fav", "hit_pct": "Fav win %"}), on="Game", how="left")
    board["Fav win %"] = (board["Fav win %"] * 100).map("{:.1f}%".format)
    print("\nGAME BOARD  (lines are for the home team; model line is informational only)")
    print(board[["Game", "gameday", "Fav", "Fav win %", "Market line", "Model line",
                 "Market total", "Model total"]].to_string(index=False))

    # ---------------- flagged bets
    show = bets[(bets.hit_pct >= a.min_hit)]
    picks = show[(show.edge_pct >= a.min_edge) & (show["ev_per_$1"] >= a.min_ev)]
    cols = ["game", "type", "bet", "book", "odds", "hit_pct", "breakeven_pct", "edge_pct",
            "ev_per_$1", "stake"]
    print(f"\n+EV GAME BETS  (hit % beats the break-even % by {a.min_edge:.0%}+)")
    if picks.empty:
        print("  None. At a single book the price already includes the vig, so an honest model"
              "\n  rarely finds value here. Shopping several books (ODDS_API_KEY) is where edges show up.")
    else:
        print(E.fmt_table(picks, cols).to_string(index=False))

    if a.min_hit > 0:
        hp = show.sort_values("hit_pct", ascending=False)
        print(f"\nALL GAME BETS WITH HIT % >= {a.min_hit:.0%}  (high hit rate is not the same as"
              " good value: check EV)")
        print(E.fmt_table(hp, cols).to_string(index=False))

    # ---------------- teasers
    legs = E.teaser_legs(mk, gm)
    be = E.teaser_breakeven(a.teaser_price, 2)
    good = legs[legs.leg_hit_pct >= be]
    print(f"\n6-POINT TEASER LEGS  (2-team at {fmt_american(a.teaser_price)} needs each leg to hit"
          f" {be:.1%}; 'wong' = crosses both 3 and 7)")
    print(E.fmt_table(legs.head(10), ["game", "team", "line", "teased_line", "leg_hit_pct",
                                      "wong"]).to_string(index=False))
    if len(good) >= 2:
        p2 = good.leg_hit_pct.iloc[0] * good.leg_hit_pct.iloc[1]
        print(f"  Best 2 legs together hit {p2:.1%} (break-even {be**2:.1%}).")
    else:
        print("  Fewer than two legs clear break-even at this price this week.")

    # ---------------- props
    prop_picks = pd.DataFrame()
    pb = pd.DataFrame()
    if a.props or a.prop_lines:
        stats = data.load_player_stats()
        proj = pipeline.player_projections(games, stats, upcoming[~upcoming.played])
        disp = pipeline.load_dispersion(games, stats)
        frames = []
        if a.props:
            if api is None:
                sys.exit("--props needs ODDS_API_KEY. Or use --prop-lines with your own CSV.")
            evs = odds.drop_duplicates("event_id")
            if a.props_games:
                want = set(a.props_games.upper().split(","))
                evs = evs[evs.home_team.isin(want) | evs.away_team.isin(want)]
            frames.append(api.prop_odds(list(evs.event_id)))
            print(f"\n(Odds API credits remaining: {api.remaining})")
        if a.prop_lines:
            frames.append(E.manual_props_to_rows(a.prop_lines, upcoming))
        prop_odds = pd.concat(frames, ignore_index=True)
        pb = E.evaluate_props(prop_odds, proj, disp, a.bankroll, my_books)
        if pb.empty:
            print("\nNo props matched to projections.")
        else:
            pcols = ["player", "market", "bet", "book", "odds", "model_proj", "market_proj",
                     "hit_pct", "breakeven_pct", "edge_pct", "ev_per_$1", "stake"]
            # props with no two-sided market (e.g. anytime TD "Yes" only) can't be checked
            # against the books' fair price, so they are never recommended
            ok = pb[~pb.news_risk & (pb.basis != "model only") & (pb.hit_pct >= a.min_hit)]
            prop_picks = ok[(ok.edge_pct >= a.min_edge) & (ok["ev_per_$1"] >= a.min_ev)]
            print(f"\n+EV PLAYER PROPS")
            print(E.fmt_table(prop_picks.head(40), pcols).to_string(index=False)
                  if len(prop_picks) else "  None cleared the thresholds.")
            n_mo = pb[pb.basis == "model only"].drop_duplicates(["player", "market"]).shape[0]
            if n_mo:
                print(f"\n(Excluded {n_mo} one-sided props, mostly anytime TD: DraftKings posts no"
                      " 'No' side, so there is no fair price to check the model against.)")
            risky = pb[pb.news_risk].drop_duplicates(["player", "market"])
            if len(risky):
                print("\nSkipped (model and books disagree by 25%+, usually injury/role news):")
                print("  " + ", ".join(f"{r.player} {r.market}" for r in risky.itertuples()))
            pb.to_csv(config.OUTPUT_DIR / f"props_{season}_wk{week}.csv", index=False)

        proj_out = proj[[c for c in proj.columns if c.startswith("mu_") or c in
                         ("player_display_name", "team", "position", "opponent_team", "n_games")]]
        proj_out.to_csv(config.OUTPUT_DIR / f"projections_{season}_wk{week}.csv", index=False)

    # ---------------- save
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    bets.to_csv(config.OUTPUT_DIR / f"game_bets_{season}_wk{week}.csv", index=False)
    legs.to_csv(config.OUTPUT_DIR / f"teasers_{season}_wk{week}.csv", index=False)
    board_path = export_board(config.OUTPUT_DIR / "board.json", season=season, week=week,
                              source=src, books=my_books, upcoming=upcoming, mk=mk,
                              bets=bets, props=pb, legs=legs, bankroll=a.bankroll)
    print(f"\nFull tables saved in {config.OUTPUT_DIR}/")
    print(f"Bet Builder file: {board_path}  (upload it in the app)")

    if a.log:
        log_bets(picks, prop_picks, season, week)


def log_bets(games_picks, prop_picks, season, week):
    today = dt.date.today().isoformat()
    rows = []
    for r in games_picks.to_dict("records"):
        rows.append(dict(date=today, season=season, week=week, game_id=r["game_id"],
                         kind="game", market=r["market"], side=r["side"], player="",
                         line=r["line"], odds=r["odds"], book=r["book"],
                         hit_pct=round(r["hit_pct"], 4), ev=round(r["ev_per_$1"], 4),
                         stake=r["stake"], result="", profit=""))
    for r in prop_picks.to_dict("records"):
        rows.append(dict(date=today, season=season, week=week, game_id=r["game"], kind="prop",
                         market=r["market"], side=r["side"], player=r["player"],
                         line=r["line"], odds=r["odds"], book=r["book"],
                         hit_pct=round(r["hit_pct"], 4), ev=round(r["ev_per_$1"], 4),
                         stake=r["stake"], result="", profit=""))
    if not rows:
        print("Nothing to log.")
        return
    new = pd.DataFrame(rows)
    if config.BET_LOG.exists():
        new = pd.concat([pd.read_csv(config.BET_LOG), new], ignore_index=True)
    new.to_csv(config.BET_LOG, index=False)
    print(f"Logged {len(rows)} bets to {config.BET_LOG.name}. Run grade.py after the games.")


if __name__ == "__main__":
    main()
