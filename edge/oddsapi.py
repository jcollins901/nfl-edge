"""Live odds from The Odds API (https://the-odds-api.com). Free tier: 500 credits/month.

Credit cost = (#markets) x (#regions) per call, and a list of up to 10 named bookmakers
counts as one region. So game lines for the whole slate cost 3 credits, and props cost
6 credits per game (~96 for a full 16-game week). Use --props-games to pull fewer games.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from .teams import to_abbr

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
GAME_MARKETS = "h2h,spreads,totals"
PROP_MARKETS = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds",
    "player_pass_completions": "completions", "player_pass_attempts": "pass_att",
    "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_att",
    "player_reception_yds": "rec_yds", "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
}
DEFAULT_PROP_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds",
                        "player_receptions", "player_pass_tds", "player_anytime_td"]
# Books whose prices are closest to "true" odds; used as the fair-value anchor when present.
SHARP_BOOKS = ["pinnacle", "circasports", "lowvig", "betonlineag"]
# The 10 books requested (10 = billed as a single region). Sharp books set fair value;
# the US books give a market consensus and the prices you can actually bet.
BOOKMAKERS = ["pinnacle", "lowvig", "betonlineag", "draftkings", "fanduel", "betmgm",
              "williamhill_us", "fanatics", "espnbet", "betrivers"]


class OddsAPI:
    def __init__(self, api_key: str | None = None, bookmakers: list[str] | None = None):
        self.key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.key:
            raise RuntimeError("Set ODDS_API_KEY (free key at https://the-odds-api.com).")
        self.bookmakers = ",".join((bookmakers or BOOKMAKERS)[:10])
        self.remaining = None
        self.used = None

    def _get(self, url: str, **params):
        params.update(apiKey=self.key, bookmakers=self.bookmakers, oddsFormat="american")
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:300]}")
        self.remaining = r.headers.get("x-requests-remaining")
        self.used = r.headers.get("x-requests-used")
        return r.json()

    def game_odds(self) -> pd.DataFrame:
        return parse_events(self._get(f"{BASE}/odds", markets=GAME_MARKETS))

    def prop_odds(self, event_ids: list[str], markets: list[str] | None = None) -> pd.DataFrame:
        markets = markets or DEFAULT_PROP_MARKETS
        frames = []
        for eid in event_ids:
            data = self._get(f"{BASE}/events/{eid}/odds", markets=",".join(markets))
            frames.append(parse_events([data]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_events(events: list[dict]) -> pd.DataFrame:
    """Flatten the API's nested JSON into one row per (book, market, outcome)."""
    rows = []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": ev["id"], "commence": ev.get("commence_time"),
                        "home_team": to_abbr(home), "away_team": to_abbr(away),
                        "book": bk["key"], "market": mk["key"],
                        "name": oc["name"], "player": oc.get("description"),
                        "point": oc.get("point"), "price": oc["price"],
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    team_side = df.market.isin(["h2h", "spreads"])
    df.loc[team_side, "name"] = df.loc[team_side, "name"].map(to_abbr)
    df["market"] = df.market.map(lambda m: PROP_MARKETS.get(m, m))
    return df
