"""Write the week's full board to one JSON file for the Bet Builder app."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from . import config

# DraftKings 6-point NFL teaser payouts as last seen on the slip (editable in the app).
TEASER_PAYOUTS = {2: -120, 3: 140, 4: 240, 5: 333, 6: 500}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return _clean(v.item())
    return v


def _rows(df: pd.DataFrame, cols: dict) -> list[dict]:
    out = []
    if df is None or df.empty:
        return out
    for r in df.to_dict("records"):
        out.append({k: _clean(r.get(src)) for k, src in cols.items()})
    return out


def export_board(path: Path, *, season, week, source, books, upcoming, mk, bets, props, legs,
                 bankroll) -> Path:
    games = []
    up = upcoming.set_index("game_id")
    for r in mk.to_dict("records"):
        gid = r.get("game_id")
        g = up.loc[gid] if gid in up.index else None
        kickoff = None
        if g is not None:
            kickoff = f"{pd.Timestamp(g.gameday).date()}T{g.gametime if isinstance(g.gametime, str) else '13:00'}"
        ml = bets[(bets.game_id == gid) & (bets.market == "h2h")] if not bets.empty else bets
        fav, fav_win = None, None
        if len(ml):
            top = ml.sort_values("hit_pct", ascending=False).iloc[0]
            fav, fav_win = top.side, float(top.hit_pct)
        games.append({
            "game_id": gid, "game": f"{r['away_team']} @ {r['home_team']}",
            "away": r["away_team"], "home": r["home_team"], "kickoff": kickoff,
            "fav": fav, "fav_win": _clean(fav_win),
            "home_line": _clean(r.get("mkt_line")), "total": _clean(r.get("mkt_total_line")),
            "model_line": _clean(-r["model_margin"]) if r.get("model_margin") is not None else None,
        })

    bet_cols = {"game": "game", "game_id": "game_id", "type": "type", "bet": "bet",
                "market": "market", "side": "side", "line": "line", "book": "book",
                "odds": "odds", "p_win": "p_win", "p_push": "p_push", "hit": "hit_pct",
                "breakeven": "breakeven_pct", "edge": "edge_pct", "ev": "ev_per_$1"}
    game_bets = _rows(bets, bet_cols)
    for i, b in enumerate(game_bets):
        b.update(id=f"g{i}", kind="game", flags=[])

    prop_cols = dict(bet_cols, player="player", team="team", pos="pos",
                     model_proj="model_proj", market_proj="market_proj", basis="basis",
                     news_risk="news_risk")
    prop_cols.pop("game_id"), prop_cols.pop("type")
    prop_rows = _rows(props, prop_cols)
    for i, b in enumerate(prop_rows):
        flags = []
        if b.get("basis") == "model only":
            flags.append("one_sided")
        if b.get("news_risk"):
            flags.append("news_risk")
        b.update(id=f"p{i}", kind="prop", type="Prop", flags=flags)

    leg_rows = _rows(legs, {"game": "game", "team": "team", "line": "line",
                            "teased_line": "teased_line", "hit": "leg_hit_pct", "wong": "wong"})
    for i, b in enumerate(leg_rows):
        b["id"] = f"t{i}"

    board = {
        "meta": {"season": season, "week": week, "source": source,
                 "books": books or "all", "generated": dt.datetime.now().isoformat(timespec="minutes"),
                 "bankroll": bankroll, "min_edge": config.MIN_EDGE, "min_ev": config.MIN_EV,
                 "kelly_fraction": config.KELLY_FRACTION, "max_stake_pct": config.MAX_STAKE_PCT},
        "games": games, "bets": game_bets, "props": prop_rows, "teaser_legs": leg_rows,
        "teaser_payouts": TEASER_PAYOUTS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, separators=(",", ":")), encoding="utf-8")
    return path
