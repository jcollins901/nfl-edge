#!/usr/bin/env python3
"""Grade the bets in bet_log.csv after games finish, and show how you're doing.

  python grade.py

Tracks win rate, profit, ROI, and closing line value (CLV): whether the line moved your
way after you bet. Over a few hundred bets, CLV tells you more about skill than W-L does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edge import config, data
from edge.evaluate import norm_name
from edge.odds import american_to_decimal, devig_two_way

STAT_COL = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "completions": "completions",
            "pass_att": "attempts", "rush_yds": "rushing_yards", "rush_att": "carries",
            "rec_yds": "receiving_yards", "receptions": "receptions"}


def grade_game(b, g):
    """Return (result, clv) where result is 'win'/'loss'/'push' or '' if not final."""
    if pd.isna(g.result):
        return "", np.nan
    home = b.side == g.home_team
    if b.market == "spreads":
        margin = g.result if home else -g.result
        x = margin + b.line
        close = -g.spread_line if home else g.spread_line
        clv = b.line - close                       # points gained vs the closing number
    elif b.market == "h2h":
        margin = g.result if home else -g.result
        x = margin
        ph, pa = devig_two_way(g.home_moneyline, g.away_moneyline) \
            if pd.notna(g.home_moneyline) else (np.nan, np.nan)
        close_p = ph if home else pa
        clv = close_p - 1 / american_to_decimal(b.odds)  # closing fair % minus what you paid
    else:  # totals
        x = (g.total - b.line) if b.side == "Over" else (b.line - g.total)
        # an Over at 44 when it closes 46 = +2 points of value
        clv = (g.total_line - b.line) if b.side == "Over" else (b.line - g.total_line)
    return ("win" if x > 0 else "loss" if x < 0 else "push"), clv


def grade_prop(b, stats):
    row = stats[(stats.season == b.season) & (stats.week == b.week) &
                (stats.key == norm_name(b.player))]
    if row.empty:
        return ""  # not played yet, or player inactive (books usually void these)
    r = row.iloc[0]
    if b.market == "anytime_td":
        scored = (r.rushing_tds or 0) + (r.receiving_tds or 0) > 0
        yes = b.side == "Over"
        return "win" if scored == yes else "loss"
    v = r[STAT_COL[b.market]]
    x = (v - b.line) if b.side == "Over" else (b.line - v)
    return "win" if x > 0 else "loss" if x < 0 else "push"


def main():
    import sys
    if not config.BET_LOG.exists():
        raise SystemExit("No bet_log.csv yet. Run: python picks.py --log")
    if "--no-update" not in sys.argv:
        data.update()
    log = pd.read_csv(config.BET_LOG, dtype={"result": "object", "profit": "float64"})
    log["result"] = log["result"].fillna("").astype(object)
    games = data.load_games().set_index("game_id")
    stats = data.load_player_stats()
    stats["key"] = stats.player_display_name.map(norm_name)

    clvs = []
    for i, b in log.iterrows():
        clv = np.nan
        if b.kind == "game" and b.game_id in games.index:
            res, clv = grade_game(b, games.loc[b.game_id])
        elif b.kind == "prop":
            res = grade_prop(b, stats)
        else:
            res = ""
        clvs.append(clv)
        if res:
            log.at[i, "result"] = res
            dec = american_to_decimal(b.odds)
            log.at[i, "profit"] = {"win": b.stake * (dec - 1), "loss": -b.stake,
                                   "push": 0.0}[res]
    log["clv"] = clvs
    log.drop(columns="clv").to_csv(config.BET_LOG, index=False)

    done = log[log.result.isin(["win", "loss", "push"])].copy()
    done["profit"] = done.profit.astype(float)
    print(f"\nGraded {len(done)} of {len(log)} bets\n")
    if done.empty:
        return
    for name, d in [("ALL", done)] + list(done.groupby("market")):
        w, l_, p = (d.result == "win").sum(), (d.result == "loss").sum(), (d.result == "push").sum()
        roi = d.profit.sum() / d.stake.sum() if d.stake.sum() else np.nan
        exp = d.hit_pct.mean()
        print(f"{name:12s} {w}-{l_}-{p}  win {w / max(w + l_, 1):.1%} (model expected {exp:.1%})"
              f"  profit ${d.profit.sum():+,.2f}  ROI {roi:+.1%}")
    g = log[log.kind == "game"]
    if g.clv.notna().any():
        sp = g[g.market.isin(["spreads", "totals"])].clv.dropna()
        ml = g[g.market == "h2h"].clv.dropna()
        if len(sp):
            print(f"\nCLV spreads/totals: avg {sp.mean():+.2f} pts vs the closing line;"
                  f" beat the close {(sp > 0).mean():.0%} of the time")
        if len(ml):
            print(f"CLV moneylines: avg {ml.mean():+.1%} closing win-prob above your price")
        print("Positive CLV over many bets = you are getting better numbers than the market"
              " settles on.")


if __name__ == "__main__":
    main()
