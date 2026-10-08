"""Glue: build the fitted models and this week's projections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, props as P
from .config import DATA_DIR
from .game_model import GameModel, make_features, walk_forward
from .ratings import build_ratings

DISP_PATH = DATA_DIR / "prop_dispersion.json"


def game_model(games: pd.DataFrame, pbp: pd.DataFrame, first_test: int = 2018):
    ratings = build_ratings(games, pbp)
    feats = make_features(games, ratings)
    oof = walk_forward(feats, first_test)
    gm = GameModel().fit(feats, oof)
    return gm, feats, oof


def next_week(games: pd.DataFrame, season: int | None = None, week: int | None = None):
    season = season or data.current_season()
    g = games[(games.season == season)]
    if week is None:
        upcoming = g[~g.played]
        if upcoming.empty:
            raise SystemExit(f"No unplayed games left in {season}.")
        week = int(upcoming.week.min())
    return g[g.week == week].copy(), week


def _future_rows(stats: pd.DataFrame, games: pd.DataFrame, upcoming: pd.DataFrame):
    """Candidate players for upcoming games: anyone who played for the team in its last
    two games this season (the model cannot see injuries, check the news)."""
    season = int(upcoming.season.iloc[0])
    cur = stats[(stats.season == season) & stats.position.isin(P.SKILL)]
    rows = []
    for g in upcoming.itertuples(index=False):
        for team, opp, implied in (
                (g.home_team, g.away_team, (g.total_line + g.spread_line) / 2),
                (g.away_team, g.home_team, (g.total_line - g.spread_line) / 2)):
            t = cur[cur.team == team]
            recent_weeks = sorted(t.week.unique())[-2:]
            ppl = t[t.week.isin(recent_weeks)].sort_values("week").drop_duplicates(
                "player_id", keep="last")
            for p in ppl.itertuples(index=False):
                rows.append({"player_id": p.player_id, "player_display_name": p.player_display_name,
                             "position": p.position, "team": team, "opponent_team": opp,
                             "season": season, "week": g.week, "game_id": g.game_id,
                             "gameday": g.gameday, "implied": implied, "future": True})
    return pd.DataFrame(rows)


def player_projections(games, stats, upcoming) -> pd.DataFrame:
    s = P.prepare(stats, games)
    s["future"] = False
    fut = _future_rows(stats, games, upcoming)
    if fut.empty:
        return pd.DataFrame()
    both = pd.concat([s, fut], ignore_index=True).sort_values("gameday", kind="stable")
    # records each defense's pre-game rating for every game, including the upcoming ones
    dfac = P.defense_factors(both)
    pri = P._pos_priors(s)
    pr = P.project_rows(s, fut, pri)
    pr = pr.merge(fut[["player_id", "game_id", "player_display_name", "position", "team",
                       "opponent_team"]], on=["player_id", "game_id"])
    pr = pr.merge(dfac, on=["game_id", "team"], how="left")
    return P.apply_adjustments(pr)


def load_dispersion(games=None, stats=None) -> P.Dispersion:
    if DISP_PATH.exists():
        return P.Dispersion.load(DISP_PATH)
    print("Fitting prop outcome distributions (one-time, ~1 minute)...")
    return fit_dispersion(games, stats, save=True)[0]


def prop_walk_forward(games, stats, first_season: int):
    s = P.prepare(stats, games)
    tg = s[s.season >= first_season]
    dfac = P.defense_factors(s)
    pri = P._pos_priors(s[s.season < first_season])
    pr = P.project_rows(s, tg, pri)
    cols = ["player_id", "game_id", "team", "position", "season", "week",
            "player_display_name", "tds"] + list(P.ACTUAL_COLS.values())
    pr = pr.merge(tg[cols], on=["player_id", "game_id"]).merge(
        dfac, on=["game_id", "team"], how="left")
    pr = P.apply_adjustments(pr)
    return pr, P.actuals(pr)


def fit_dispersion(games, stats, save=True, first_season: int = 2023):
    pr, act = prop_walk_forward(games, stats, first_season)
    disp = P.Dispersion.fit(pr, act)
    if save:
        disp.save(DISP_PATH)
    return disp, pr, act
