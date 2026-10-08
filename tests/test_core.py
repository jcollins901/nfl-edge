"""Run:  python -m pytest tests   (or  python tests/test_core.py)

Uses a hand-made response in The Odds API's documented format, so the live-odds path is
tested without an API key.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge import odds as O  # noqa: E402
from edge.evaluate import evaluate_games, evaluate_props, market_means, norm_name  # noqa: E402
from edge.game_model import GameModel  # noqa: E402
from edge.oddsapi import parse_events  # noqa: E402
from edge.props import Dispersion  # noqa: E402

FIXTURE = [{
    "id": "ev1", "sport_key": "americanfootball_nfl", "commence_time": "2026-10-04T13:30:00Z",
    "home_team": "Washington Commanders", "away_team": "Indianapolis Colts",
    "bookmakers": [
        {"key": "pinnacle", "title": "Pinnacle", "markets": [
            {"key": "spreads", "outcomes": [
                {"name": "Washington Commanders", "price": -104, "point": 3.5},
                {"name": "Indianapolis Colts", "price": -106, "point": -3.5}]},
            {"key": "h2h", "outcomes": [
                {"name": "Washington Commanders", "price": 158},
                {"name": "Indianapolis Colts", "price": -172}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": -105, "point": 47.5},
                {"name": "Under", "price": -105, "point": 47.5}]}]},
        {"key": "draftkings", "title": "DraftKings", "markets": [
            {"key": "spreads", "outcomes": [
                # soft book hanging the key number +4 at a normal price -> should be flagged
                {"name": "Washington Commanders", "price": -110, "point": 4.5},
                {"name": "Indianapolis Colts", "price": -110, "point": -4.5}]},
            {"key": "h2h", "outcomes": [
                {"name": "Washington Commanders", "price": 150},
                {"name": "Indianapolis Colts", "price": -180}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": -110, "point": 47.5},
                {"name": "Under", "price": -110, "point": 47.5}]}]},
    ]}]

PROP_FIXTURE = [{
    "id": "ev1", "home_team": "Washington Commanders", "away_team": "Indianapolis Colts",
    "commence_time": "2026-10-04T13:30:00Z",
    "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "player_rush_yds", "outcomes": [
            {"name": "Over", "description": "Jonathan Taylor", "price": -110, "point": 84.5},
            {"name": "Under", "description": "Jonathan Taylor", "price": -110, "point": 84.5}]}]},
        {"key": "fanduel", "markets": [{"key": "player_rush_yds", "outcomes": [
            {"name": "Over", "description": "Jonathan Taylor", "price": -114, "point": 79.5},
            {"name": "Under", "description": "Jonathan Taylor", "price": -106, "point": 79.5}]}]},
    ]}]


def gm_default():
    gm = GameModel()
    gm.sigma_margin, gm.sigma_total = 12.8, 13.2
    return gm


def test_odds_math():
    assert abs(O.american_to_prob(-110) - 0.5238) < 1e-3
    assert abs(O.american_to_prob(+150) - 0.4) < 1e-9
    a, b = O.devig_two_way(-110, -110)
    assert abs(a - 0.5) < 1e-9 and abs(a + b - 1) < 1e-12
    assert abs(O.expected_value(0.5, -110) - (-0.04545)) < 1e-4
    assert O.kelly_fraction(0.5, -110) == 0.0
    assert O.kelly_fraction(0.6, +100) > 0.19
    assert abs(O.prob_to_american(0.5) + 100) < 1e-9


def test_parse_and_line_shopping():
    df = parse_events(FIXTURE)
    assert set(df.book) == {"pinnacle", "draftkings"}
    assert set(df[df.market == "spreads"].name) == {"WAS", "IND"}
    gm = gm_default()
    mk = market_means(df, gm)
    # Pinnacle is the anchor: WAS +3.5 roughly a coin flip -> IND favored by ~3.5
    assert -5 < mk.mkt_margin.iloc[0] < -2.5
    preds = pd.DataFrame({"home_team": ["WAS"], "away_team": ["IND"], "game_id": ["g"],
                          "model_margin": [0.0], "model_total": [47.0]})
    bets, _ = evaluate_games(df, preds, gm, 1000)
    was = bets[(bets.market == "spreads") & (bets.side == "WAS")].iloc[0]
    assert was.book == "draftkings" and was.line == 4.5
    assert was["ev_per_$1"] > 0.0          # an extra point through 4 at -110 is +EV
    assert was.hit_pct > 0.52


def test_probabilities_sum_to_one():
    gm = gm_default()
    for mu in (-10, -3, 0, 2.5, 7):
        assert abs(sum(gm.moneyline_probs(mu)) - 1) < 1e-9
        assert abs(sum(gm.spread_probs(mu, -3)) - 1) < 1e-9
        assert abs(sum(gm.total_probs(45, 44)) - 1) < 1e-9


def test_props_blend_and_labels():
    df = parse_events(PROP_FIXTURE)
    assert set(df.market) == {"rush_yds"} and df.player.iloc[0] == "Jonathan Taylor"
    disp = Dispersion({"rush_yds": {"kind": "yards", "a": 3.6, "b": 0.56}})
    proj = pd.DataFrame({"player_display_name": ["Jonathan Taylor"], "team": ["IND"],
                         "position": ["RB"], "mu_rush_yds": [92.0], "n_games": [20]})
    out = evaluate_props(df, proj, disp, 1000)
    over = out[out.side == "Over"].iloc[0]
    assert over.book == "fanduel" and over.line == 79.5   # lower line is the better over
    assert over.market_proj < over.fair_proj < over.model_proj
    assert not over.news_risk


def test_name_normalizing():
    assert norm_name("Marvin Harrison Jr.") == norm_name("Marvin Harrison")
    assert norm_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_implied_mu_inverts_prob_over():
    disp = Dispersion({"rec_yds": {"kind": "yards", "a": 3.2, "b": 0.6},
                       "receptions": {"kind": "count", "c": 1.37, "d": 0.0}})
    for m, mu, line in (("rec_yds", 64.0, 58.5), ("receptions", 5.1, 4.5)):
        p, _ = disp.prob_over(m, mu, line)
        assert abs(disp.implied_mu(m, line, p) - mu) < 0.05 * mu


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
