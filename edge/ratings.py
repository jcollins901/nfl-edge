"""Opponent-adjusted, recency-weighted team ratings built from play-by-play.

For every game we record each team's rating *before* kickoff, so nothing the model
trains on or predicts with ever uses information from the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RATING_ALPHA, SEASON_CARRYOVER

HFA_POINTS = 1.5  # used only to de-bias points before they feed the ratings

# (stat, starting value). Offense stats: higher = better. Defense stats: EPA allowed, lower = better.
STATE_VARS = {
    "off_epa": 0.0, "off_pass_epa": 0.05, "off_rush_epa": -0.08, "off_sr": 0.44,
    "def_epa": 0.0, "def_pass_epa": 0.05, "def_rush_epa": -0.08, "def_sr": 0.44,
    "pf": 22.0, "pa": 22.0, "plays": 62.0, "pass_rate": 0.58,
}
# Which opponent stat is used to adjust each of ours for strength of schedule.
OPP_ADJUST = {
    "off_epa": "def_epa", "off_pass_epa": "def_pass_epa", "off_rush_epa": "def_rush_epa",
    "off_sr": "def_sr", "def_epa": "off_epa", "def_pass_epa": "off_pass_epa",
    "def_rush_epa": "off_rush_epa", "def_sr": "off_sr", "pf": "pa", "pa": "pf",
}


def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with offensive and defensive efficiency."""
    grp = pbp.groupby(["game_id", "posteam"])
    off = pd.DataFrame({
        "off_epa": grp.epa.mean(),
        "off_sr": grp.success.mean(),
        "plays": grp.size(),
        "pass_rate": grp.qb_dropback.mean(),
    })
    p = pbp[pbp["pass"] == 1].groupby(["game_id", "posteam"]).epa.mean().rename("off_pass_epa")
    r = pbp[pbp["rush"] == 1].groupby(["game_id", "posteam"]).epa.mean().rename("off_rush_epa")
    off = off.join(p).join(r).reset_index().rename(columns={"posteam": "team"})

    dgrp = pbp.groupby(["game_id", "defteam"])
    dfn = pd.DataFrame({"def_epa": dgrp.epa.mean(), "def_sr": dgrp.success.mean()})
    dp = pbp[pbp["pass"] == 1].groupby(["game_id", "defteam"]).epa.mean().rename("def_pass_epa")
    dr = pbp[pbp["rush"] == 1].groupby(["game_id", "defteam"]).epa.mean().rename("def_rush_epa")
    dfn = dfn.join(dp).join(dr).reset_index().rename(columns={"defteam": "team"})
    return off.merge(dfn, on=["game_id", "team"], how="outer")


def build_ratings(games: pd.DataFrame, pbp: pd.DataFrame,
                  alpha: float = RATING_ALPHA, carryover: float = SEASON_CARRYOVER) -> pd.DataFrame:
    """Walk through games in date order, recording pre-game ratings and then updating."""
    tgs = team_game_stats(pbp).set_index(["game_id", "team"])
    state: dict[str, dict] = {}
    rows = []
    keys = list(STATE_VARS)

    def get(team, season):
        s = state.get(team)
        if s is None:
            s = {"v": dict(STATE_VARS), "season": season, "n": 0}
            state[team] = s
        if s["season"] != season:  # new season: regress toward league average
            lg = league_mean()
            s["v"] = {k: lg[k] + carryover * (s["v"][k] - lg[k]) for k in keys}
            s["season"], s["n"] = season, 0
        return s

    def league_mean():
        if not state:
            return dict(STATE_VARS)
        return {k: float(np.mean([t["v"][k] for t in state.values()])) for k in keys}

    for g in games.itertuples(index=False):
        h, a = get(g.home_team, g.season), get(g.away_team, g.season)
        row = {"game_id": g.game_id, "home_n": h["n"], "away_n": a["n"]}
        for k in keys:
            row[f"home_{k}"] = h["v"][k]
            row[f"away_{k}"] = a["v"][k]
        rows.append(row)

        if not g.played or (g.game_id, g.home_team) not in tgs.index or \
                (g.game_id, g.away_team) not in tgs.index:
            continue
        lg = league_mean()
        hfa = 0 if g.neutral else HFA_POINTS
        obs = {
            g.home_team: dict(tgs.loc[(g.game_id, g.home_team)], pf=g.home_score - hfa / 2,
                              pa=g.away_score + hfa / 2),
            g.away_team: dict(tgs.loc[(g.game_id, g.away_team)], pf=g.away_score + hfa / 2,
                              pa=g.home_score - hfa / 2),
        }
        pre = {g.home_team: dict(h["v"]), g.away_team: dict(a["v"])}
        for team, opp, s in ((g.home_team, g.away_team, h), (g.away_team, g.home_team, a)):
            eff_alpha = max(alpha, 1.0 / (s["n"] + 3))  # learn faster early in a season
            for k in keys:
                x = obs[team].get(k)
                if x is None or pd.isna(x):
                    continue
                if k in OPP_ADJUST:
                    ok = OPP_ADJUST[k]
                    x = x - (pre[opp][ok] - lg[ok])
                s["v"][k] += eff_alpha * (x - s["v"][k])
            s["n"] += 1
    return pd.DataFrame(rows)
