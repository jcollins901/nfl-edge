"""Turn odds + model into a ranked list of bets with hit %, edge, EV and stake size."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .game_model import GameModel
from .odds import (american_to_prob, devig_two_way, expected_value, fmt_american,
                   kelly_fraction)
from .oddsapi import SHARP_BOOKS
from .props import Dispersion


# ------------------------------------------------------------------ helpers
def _solve(f, target, lo, hi, iters=50):
    """Find x with f(x) == target for an increasing f."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def _consensus(values: pd.Series, books: pd.Series) -> float:
    sharp = values[books.isin(SHARP_BOOKS)].dropna()
    if len(sharp):
        return float(sharp.mean())
    v = values.dropna()
    return float(v.median()) if len(v) else np.nan


def stake(p_win, odds, p_push, bankroll):
    f = kelly_fraction(p_win, odds, p_push) * config.KELLY_FRACTION
    return round(min(f, config.MAX_STAKE_PCT) * bankroll, 2)


def games_to_odds_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Fallback when there is no API key: the consensus lines stored in nflverse."""
    rows = []
    for g in games.itertuples(index=False):
        base = dict(event_id=g.game_id, commence=str(g.gameday.date()), home_team=g.home_team,
                    away_team=g.away_team, book="consensus", player=None)
        if pd.notna(g.spread_line):
            rows += [dict(base, market="spreads", name=g.home_team, point=-g.spread_line,
                          price=g.home_spread_odds if pd.notna(g.home_spread_odds) else -110),
                     dict(base, market="spreads", name=g.away_team, point=g.spread_line,
                          price=g.away_spread_odds if pd.notna(g.away_spread_odds) else -110)]
        if pd.notna(g.home_moneyline):
            rows += [dict(base, market="h2h", name=g.home_team, point=None, price=g.home_moneyline),
                     dict(base, market="h2h", name=g.away_team, point=None, price=g.away_moneyline)]
        if pd.notna(g.total_line):
            rows += [dict(base, market="totals", name="Over", point=g.total_line,
                          price=g.over_odds if pd.notna(g.over_odds) else -110),
                     dict(base, market="totals", name="Under", point=g.total_line,
                          price=g.under_odds if pd.notna(g.under_odds) else -110)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ game lines
def market_means(odds: pd.DataFrame, gm: GameModel) -> pd.DataFrame:
    """For each game, the expected home margin and total the market is pricing in.

    Every book's no-vig price is converted back into an expected margin/total using the
    same key-number-aware distribution, so books hanging different numbers (-2.5 -120 vs
    -3 +100) can be compared on one scale.
    """
    out = []
    for (eid, home, away), d in odds.groupby(["event_id", "home_team", "away_team"]):
        per_book = []
        for book, b in d.groupby("book"):
            rec = {"book": book, "mu_sp": np.nan, "mu_ml": np.nan, "mu_t": np.nan,
                   "line": np.nan, "total_line": np.nan}
            sp = b[b.market == "spreads"]
            h, a = sp[sp.name == home], sp[sp.name == away]
            if len(h) and len(a) and h.point.iloc[0] == -a.point.iloc[0]:
                p_home, _ = devig_two_way(h.price.iloc[0], a.price.iloc[0])
                line = h.point.iloc[0]
                rec["line"] = line

                def cover(mu, line=line):
                    pc, pp, _ = gm.spread_probs(mu, line)
                    return pc / (1 - pp)
                rec["mu_sp"] = _solve(cover, p_home, -40, 40)
            ml = b[b.market == "h2h"]
            h, a = ml[ml.name == home], ml[ml.name == away]
            if len(h) and len(a):
                p_home, _ = devig_two_way(h.price.iloc[0], a.price.iloc[0])

                def win(mu):
                    pw, pt, _ = gm.moneyline_probs(mu)
                    return pw / (1 - pt)
                rec["mu_ml"] = _solve(win, p_home, -40, 40)
            to = b[b.market == "totals"]
            o, u = to[to.name == "Over"], to[to.name == "Under"]
            if len(o) and len(u) and o.point.iloc[0] == u.point.iloc[0]:
                p_over, _ = devig_two_way(o.price.iloc[0], u.price.iloc[0])
                line = o.point.iloc[0]
                rec["total_line"] = line

                def over(mu, line=line):
                    po, pp, _ = gm.total_probs(mu, line)
                    return po / (1 - pp)
                rec["mu_t"] = _solve(over, p_over, 10, 90)
            per_book.append(rec)
        pb = pd.DataFrame(per_book)
        mu_sp, mu_ml = _consensus(pb.mu_sp, pb.book), _consensus(pb.mu_ml, pb.book)
        out.append({"event_id": eid, "home_team": home, "away_team": away,
                    # spread-market view (used for spreads, teasers) and moneyline-market
                    # view (used for moneylines); each falls back to the other if missing
                    "mkt_margin": mu_sp if np.isfinite(mu_sp) else mu_ml,
                    "mkt_margin_ml": mu_ml if np.isfinite(mu_ml) else mu_sp,
                    "mkt_total": _consensus(pb.mu_t, pb.book),
                    "mkt_line": _consensus(pb.line, pb.book),
                    "mkt_total_line": _consensus(pb.total_line, pb.book),
                    "n_books": len(pb)})
    return pd.DataFrame(out)


def evaluate_games(odds: pd.DataFrame, preds: pd.DataFrame, gm: GameModel,
                   bankroll: float, bet_books: list[str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """preds: one row per game with home_team, away_team, model_margin, model_total."""
    mk = market_means(odds, gm).merge(
        preds[["home_team", "away_team", "game_id", "model_margin", "model_total"]],
        on=["home_team", "away_team"], how="left")
    mk["fair_margin"] = mk.mkt_margin + gm.w_margin * (mk.model_margin - mk.mkt_margin)
    mk["fair_margin_ml"] = mk.mkt_margin_ml + gm.w_margin * (mk.model_margin - mk.mkt_margin_ml)
    mk["fair_total"] = mk.mkt_total + gm.w_total * (mk.model_total - mk.mkt_total)
    fair = mk.set_index("event_id")

    bets = []
    for r in odds.itertuples(index=False):
        f = fair.loc[r.event_id]
        mu_m, mu_t = f.fair_margin, f.fair_total
        if r.market == "spreads" and np.isfinite(mu_m):
            is_home = r.name == r.home_team
            pc, pp, po = gm.spread_probs(mu_m, r.point if is_home else -r.point)
            p_win, p_push = (pc, pp) if is_home else (po, pp)
            sel = f"{r.name} {r.point:+g}"
            btype = "Spread"
        elif r.market == "h2h" and np.isfinite(f.fair_margin_ml):
            pw, pt, pl = gm.moneyline_probs(f.fair_margin_ml)
            p_win, p_push = (pw, pt) if r.name == r.home_team else (pl, pt)
            sel, btype = f"{r.name} ML", "Moneyline"
        elif r.market == "totals" and np.isfinite(mu_t):
            po, pp, pu = gm.total_probs(mu_t, r.point)
            p_win, p_push = (po, pp) if r.name == "Over" else (pu, pp)
            sel, btype = f"{r.name} {r.point:g}", "Total"
        else:
            continue
        hit = p_win / (1 - p_push) if p_push < 1 else np.nan
        be = american_to_prob(r.price)
        ev = expected_value(p_win, r.price, p_push)
        bets.append({
            "game": f"{r.away_team} @ {r.home_team}", "game_id": f.game_id, "type": btype,
            "bet": sel, "market": r.market, "side": r.name, "line": r.point, "book": r.book,
            "odds": r.price, "hit_pct": hit, "breakeven_pct": be, "edge_pct": hit - be,
            "ev_per_$1": ev, "stake": stake(p_win, r.price, p_push, bankroll),
            "p_win": p_win, "p_push": p_push,
            "commence": r.commence,
        })
    bets = pd.DataFrame(bets)
    if not bets.empty and bet_books:
        # every book still sets fair value; only recommend books you can actually use
        bets = bets[bets.book.isin(bet_books)]
    if not bets.empty:
        # keep the best price for each distinct bet
        bets = (bets.sort_values("ev_per_$1", ascending=False)
                .drop_duplicates(["game", "market", "side", "line"]))
    return bets, mk


def teaser_legs(mk: pd.DataFrame, gm: GameModel, points: float = 6.0) -> pd.DataFrame:
    """Probability each side covers after being teased. 'Wong' legs move a line through
    both 3 and 7 (favorites of -7.5 to -8.5, underdogs of +1.5 to +2.5)."""
    rows = []
    for r in mk.itertuples(index=False):
        if not (np.isfinite(r.fair_margin) and np.isfinite(r.mkt_line)):
            continue
        home_line = round(r.mkt_line * 2) / 2  # consensus spread, to the half point
        for team, line, is_home in ((r.home_team, home_line, True),
                                    (r.away_team, -home_line, False)):
            teased = line + points
            hl = teased if is_home else -teased
            pc, pp, po = gm.spread_probs(r.fair_margin, hl)
            p = (pc if is_home else po) / (1 - pp)
            wong = (-8.5 <= line <= -7.5) or (1.5 <= line <= 2.5)
            rows.append({"game": f"{r.away_team} @ {r.home_team}", "team": team,
                         "line": line, "teased_line": teased, "leg_hit_pct": p, "wong": wong})
    return pd.DataFrame(rows).sort_values("leg_hit_pct", ascending=False)


def teaser_breakeven(price: float, legs: int = 2) -> float:
    return american_to_prob(price) ** (1 / legs)


# ------------------------------------------------------------------ player props
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(n: str) -> str:
    n = re.sub(r"[^a-z ]", "", str(n).lower().replace("-", " "))
    return re.sub(r"\s+", " ", _SUFFIX.sub("", n)).strip()


def evaluate_props(prop_odds: pd.DataFrame, proj: pd.DataFrame, disp: Dispersion,
                   bankroll: float, bet_books: list[str] | None = None) -> pd.DataFrame:
    if prop_odds.empty or proj.empty:
        return pd.DataFrame()
    po = prop_odds.copy()
    po.loc[po.name.isin(["Yes"]), "name"] = "Over"
    po.loc[po.name.isin(["No"]), "name"] = "Under"
    po["point"] = po.point.fillna(0.5)
    po["key"] = po.player.map(norm_name)
    pj = proj.copy()
    pj["key"] = pj.player_display_name.map(norm_name)

    rows = []
    for (key, market, eid), d in po.groupby(["key", "market", "event_id"]):
        cand = pj[(pj.key == key) & pj.team.isin([d.home_team.iloc[0], d.away_team.iloc[0]])]
        col = f"mu_{market}"
        if cand.empty or col not in cand or pd.isna(cand[col].iloc[0]):
            continue
        p = cand.iloc[0]
        model_mu = float(p[col])
        # market-implied projection from each book's no-vig price
        impl = []
        for (book, point), b in d.groupby(["book", "point"]):
            o, u = b[b.name == "Over"], b[b.name == "Under"]
            if len(o) and len(u):
                pov, _ = devig_two_way(o.price.iloc[0], u.price.iloc[0])
                impl.append({"book": book, "mu": disp.implied_mu(market, point, pov)})
        impl = pd.DataFrame(impl, columns=["book", "mu"])
        mkt_mu = _consensus(impl.mu, impl.book) if len(impl) else np.nan
        if np.isfinite(mkt_mu):
            w = config.PROP_MODEL_WEIGHT
            fair_mu = w * model_mu + (1 - w) * mkt_mu
            gap = (model_mu - mkt_mu) / max(mkt_mu, 1e-6)
            basis = "model+market"
        else:
            fair_mu, gap, basis = model_mu, np.nan, "model only"
        p_over, p_push = disp.prob_over(market, fair_mu, d.point.iloc[0])
        for r in d.itertuples(index=False):
            po_, pp = disp.prob_over(market, fair_mu, r.point)
            if np.isnan(po_):
                continue
            p_win = po_ if r.name == "Over" else 1 - po_ - pp
            hit = p_win / (1 - pp)
            be = american_to_prob(r.price)
            rows.append({
                "player": p.player_display_name, "team": p.team, "pos": p.position,
                "game": f"{r.away_team} @ {r.home_team}", "market": market,
                "bet": (f"Anytime TD: {'Yes' if r.name == 'Over' else 'No'}"
                        if market == "anytime_td" else f"{r.name} {r.point:g}"),
                "side": r.name, "line": r.point,
                "book": r.book, "odds": r.price, "model_proj": model_mu, "market_proj": mkt_mu,
                "fair_proj": fair_mu, "model_vs_market": gap, "hit_pct": hit,
                "breakeven_pct": be, "edge_pct": hit - be,
                "ev_per_$1": expected_value(p_win, r.price, pp),
                "p_win": p_win, "p_push": pp,
                "stake": stake(p_win, r.price, pp, bankroll), "basis": basis,
                "games_played": p.n_games, "commence": r.commence,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["news_risk"] = out.model_vs_market.abs() > config.PROP_MAX_DISAGREEMENT
    if bet_books:
        out = out[out.book.isin(bet_books)]
    return (out.sort_values("ev_per_$1", ascending=False)
            .drop_duplicates(["player", "market", "side", "line"]))


def manual_props_to_rows(path, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """CSV columns: player, team, opponent, market, line, over_odds, under_odds[, book].
    `team`/`opponent` are abbreviations (e.g. IND, WAS); home/away is looked up from the
    schedule. Leave under_odds blank if your book only offers one side."""
    m = pd.read_csv(path)
    rows = []
    for r in m.itertuples(index=False):
        book = getattr(r, "book", "manual")
        home, away = r.team, r.opponent
        if schedule is not None:
            hit = schedule[((schedule.home_team == r.team) & (schedule.away_team == r.opponent)) |
                           ((schedule.home_team == r.opponent) & (schedule.away_team == r.team))]
            if len(hit):
                home, away = hit.home_team.iloc[0], hit.away_team.iloc[0]
        base = dict(event_id=f"{away}@{home}", commence="", home_team=home,
                    away_team=away, book=book, market=r.market, player=r.player,
                    point=r.line)
        if pd.notna(r.over_odds):
            rows.append(dict(base, name="Over", price=r.over_odds))
        if pd.notna(getattr(r, "under_odds", np.nan)):
            rows.append(dict(base, name="Under", price=r.under_odds))
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    t = df[cols].copy()
    for c in t.columns:
        if c in ("hit_pct", "breakeven_pct", "edge_pct", "leg_hit_pct", "model_vs_market"):
            t[c] = (t[c] * 100).map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
        elif c == "odds":
            t[c] = t[c].map(fmt_american)
        elif c == "ev_per_$1":
            t[c] = t[c].map(lambda x: f"{x:+.3f}")
        elif c in ("model_proj", "market_proj", "fair_proj"):
            t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        elif c == "stake":
            t[c] = t[c].map(lambda x: f"${x:,.2f}")
    return t
