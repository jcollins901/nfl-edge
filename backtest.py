#!/usr/bin/env python3
"""Honest report card. Everything here is out-of-sample: each season is predicted using
only data from before it. Writes output/backtest_report.txt and refits prop distributions.

  python backtest.py            # full report (~2 minutes)
  python backtest.py --no-props # game side only (~15 seconds)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from edge import config, data, evaluate as E, pipeline, props as P
from edge.game_model import GameModel
from edge.odds import devig_two_way, expected_value

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def record(win, push=None):
    win = np.asarray(win, float)
    return f"{np.nanmean(win):.1%} over {np.isfinite(win).sum()} bets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-props", action="store_true")
    ap.add_argument("--first-test", type=int, default=2018)
    a = ap.parse_args()

    games = data.load_games()
    pbp = data.load_pbp()
    gm, feats, oof = pipeline.game_model(games, pbp, a.first_test)
    o = oof[oof.played & oof.spread_line.notna() & oof.total_line.notna()].copy()

    say("=" * 78)
    say("1. STATS MODEL vs CLOSING LINE  (seasons %d-%d, %d games)"
        % (o.season.min(), o.season.max(), len(o)))
    say("=" * 78)
    say(f"Avg miss on final margin:  model {np.abs(o.result - o.model_margin).mean():.2f} pts"
        f"   market {np.abs(o.result - o.spread_line).mean():.2f} pts")
    say(f"Avg miss on total points:  model {np.abs(o.total - o.model_total).mean():.2f} pts"
        f"   market {np.abs(o.total - o.total_line).mean():.2f} pts")
    x, y = o.model_margin - o.spread_line, o.result - o.spread_line
    for k in (2, 4):
        sel = (x.abs() > k) & (y != 0)
        win = ((x[sel] > 0) == (y[sel] > 0)).astype(float)
        say(f"Betting the model's side when it disagrees with the line by {k}+ pts: {record(win)}"
            f"  (need 52.4% at -110)")
    say(f"Learned blend weight: spreads {gm.w_margin:.2f}, totals {gm.w_total:.2f}"
        "  (0 = model adds nothing beyond the closing line)")
    say("Takeaway: closing lines already contain everything a stats model knows plus injury\n"
        "and news it doesn't. The tool therefore prices bets off the market and looks for\n"
        "books that are out of line, instead of trusting its own spread.")

    # ------------------------------------------------------------------ probability engine
    say("\n" + "=" * 78)
    say("2. ARE THE HIT % NUMBERS TRUE?  (calibration, market-anchored)")
    say("=" * 78)
    rows = []
    for r in o[o.home_spread_odds.notna() & o.home_moneyline.notna()].itertuples(index=False):
        odds = E.games_to_odds_rows(pd.DataFrame([r._asdict()]).assign(
            gameday=pd.to_datetime(r.gameday)))
        mm = E.market_means(odds, gm).iloc[0]
        rows.append({"season": r.season, "result": r.result, "total": r.total,
                     "spread_line": r.spread_line, "total_line": r.total_line,
                     "mu": mm.mkt_margin, "mu_t": mm.mkt_total,
                     "home_ml": r.home_moneyline, "away_ml": r.away_moneyline})
    c = pd.DataFrame(rows)
    # refit key-number weights on seasons before the test window only
    tr = c[c.season < c.season.min() + 3]
    te = c[c.season >= c.season.min() + 3].copy()
    gm2 = GameModel(margin_model=gm.margin_model, total_model=gm.total_model,
                    sigma_margin=gm.sigma_margin, sigma_total=gm.sigma_total)
    from edge.game_model import _key_number_weights
    gm2.key_weights = _key_number_weights(tr.result, tr.mu, gm2.sigma_margin)

    def calib(name, p, y):
        d = pd.DataFrame({"p": p, "y": y}).dropna()
        d["b"] = pd.cut(d.p, [0, .5, .6, .7, .8, .9, 1.0])
        t = d.groupby("b", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"),
                                              n=("y", "size"))
        say(f"{name}: Brier {((d.p - d.y) ** 2).mean():.4f}")
        for r in t.itertuples():
            say(f"   predicted {r.pred:5.1%}  ->  actually hit {r.actual:5.1%}   ({r.n} bets)")

    # moneyline favorites
    pw = []
    for r in te.itertuples():
        w, t_, l = gm2.moneyline_probs(r.mu)
        pw.append(max(w, l) / (1 - t_))
    fav_home = [gm2.moneyline_probs(m)[0] >= gm2.moneyline_probs(m)[2] for m in te.mu]
    fav_won = np.where(te.result == 0, np.nan,
                       np.where(fav_home, te.result > 0, te.result < 0).astype(float))
    calib("Moneyline favorite win %", pw, fav_won)

    # 6-point teaser legs, both sides of every game
    tp, ty, wong = [], [], []
    for r in te.itertuples():
        hl = -r.spread_line  # home line in betting convention
        for is_home, line in ((True, hl), (False, -hl)):
            t = line + 6
            pc, pp, po = gm2.spread_probs(r.mu, t if is_home else -t)
            tp.append((pc if is_home else po) / (1 - pp))
            m = r.result if is_home else -r.result
            ty.append(np.nan if m + t == 0 else float(m + t > 0))
            wong.append((-8.5 <= line <= -7.5) or (1.5 <= line <= 2.5))
    tp, ty, wong = np.array(tp), np.array(ty), np.array(wong)
    calib("6-pt teaser legs (all)", tp, ty)
    say(f"   Wong legs only: predicted {tp[wong].mean():.1%}, actually hit {np.nanmean(ty[wong]):.1%}"
        f" over {np.isfinite(ty[wong]).sum()} legs")
    say(f"   Break-even per leg for a 2-team teaser: -110 {E.teaser_breakeven(-110):.1%}, "
        f"-120 {E.teaser_breakeven(-120):.1%}, -130 {E.teaser_breakeven(-130):.1%}")

    # ------------------------------------------------------------------ high-hit strategies
    say("\n" + "=" * 78)
    say("3. 'HIGH HIT RATE' BETS AT CLOSING PRICES  (why hit % alone isn't enough)")
    say("=" * 78)
    rows = []
    for r in c.itertuples():
        if r.result == 0:
            continue
        ph, pa = devig_two_way(r.home_ml, r.away_ml)
        fav_is_home = ph >= pa
        p = max(ph, pa)
        won = (r.result > 0) if fav_is_home else (r.result < 0)
        price = r.home_ml if fav_is_home else r.away_ml
        rows.append({"p": p, "won": won, "profit": expected_value(float(won), price)})
    f = pd.DataFrame(rows)
    f["bucket"] = pd.cut(f.p, [0.5, 0.6, 0.7, 0.8, 1.0],
                         labels=["50-60%", "60-70%", "70-80%", "80%+"])
    for b, d in f.groupby("bucket", observed=True):
        say(f"Moneyline favorites {b:>7}: won {d.won.mean():.1%} of {len(d)} bets,"
            f" return per $1: {d.profit.mean():+.3f}")
    say("Big favorites win most of the time but still lose money: the price already charges\n"
        "you for that certainty, plus the vig. Look for EV > 0, not just a high hit %.")

    # ------------------------------------------------------------------ props
    if not a.no_props:
        say("\n" + "=" * 78)
        say("4. PLAYER PROP PROJECTIONS  (2023 onward, starters/real roles only)")
        say("=" * 78)
        stats = data.load_player_stats()
        disp, pr, act = pipeline.fit_dispersion(games, stats, save=True)
        s = P.prepare(stats, games)
        # naive baseline: season-to-date average before the game (what casual bettors use)
        for m in ["pass_yds", "rush_yds", "rec_yds", "receptions", "pass_tds"]:
            col = P.ACTUAL_COLS[m]
            s[f"naive_{m}"] = (s.groupby(["player_id", "season"])[col]
                               .transform(lambda v: v.shift().expanding().mean()))
        base = pr.merge(s[["player_id", "game_id"] + [c for c in s if c.startswith("naive_")]],
                        on=["player_id", "game_id"], how="left")
        rng = np.random.default_rng(0)
        for m in ["pass_yds", "pass_tds", "rush_yds", "rec_yds", "receptions", "anytime_td"]:
            rel = P.relevant(base, m) & base[f"mu_{m}"].notna()
            y = act.loc[rel.values, f"y_{m}"].values
            mu = base.loc[rel, f"mu_{m}"].values
            line = np.where(m == "anytime_td", 0.5,
                            np.floor(mu * (1 + rng.uniform(-.25, .25, len(mu)))) + 0.5)
            p = np.array([disp.prob_over(m, u, l)[0] for u, l in zip(mu, line)])
            hit = (y > line).astype(float)
            msg = f"{m:11s} n={rel.sum():5d}  avg miss {np.mean(np.abs(mu - y)):6.2f}"
            if f"naive_{m}" in base:
                nv = base.loc[rel, f"naive_{m}"].values
                ok = np.isfinite(nv)
                msg += (f"  (season-avg baseline {np.mean(np.abs(nv[ok] - y[ok])):6.2f} on"
                        f" same games: model {np.mean(np.abs(mu[ok] - y[ok])):6.2f})")
            say(msg)
            d = pd.DataFrame({"p": p, "y": hit})
            d["b"] = pd.cut(d.p, [0, .35, .45, .55, .65, 1])
            t = d.groupby("b", observed=True).agg(pred=("p", "mean"), act=("y", "mean"),
                                                  n=("y", "size"))
            say("            calibration: " + " | ".join(
                f"{r.pred:.0%}->{r.act:.0%}" for r in t.itertuples()))
        say("Prop lines aren't in any free historical dataset, so profit vs real books can't\n"
            "be backtested. Track it going forward with grade.py; closing line value (CLV)\n"
            "is the fastest honest signal of whether you're beating the books.")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "backtest_report.txt").write_text("\n".join(LINES))
    print(f"\nSaved to {config.OUTPUT_DIR / 'backtest_report.txt'}")


if __name__ == "__main__":
    main()
