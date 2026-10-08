"""Download and load nflverse data (free, updated nightly during the season)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, FIRST_SEASON
from .teams import norm_abbr

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
PSTATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "stats_player/stats_player_week_{y}.parquet")

PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "epa",
            "success", "pass", "rush", "qb_dropback"]


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _download(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def update(seasons: list[int] | None = None, force: bool = False) -> None:
    """Fetch data files. The current season is always re-downloaded; older ones only if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cur = current_season()
    seasons = seasons or list(range(FIRST_SEASON, cur + 1))
    print("Updating schedule / lines ...")
    _download(GAMES_URL, DATA_DIR / "games.csv")
    for y in seasons:
        for url, name in ((PBP_URL, f"pbp_{y}.parquet"),
                          (PSTATS_URL, f"stats_player_week_{y}.parquet")):
            if name.startswith("stats_player") and y < cur - 5:
                continue
            dest = DATA_DIR / name
            if force or y == cur or not dest.exists():
                print(f"  downloading {name}")
                try:
                    _download(url.format(y=y), dest)
                except requests.HTTPError as e:
                    print(f"  (skipped {name}: {e})")


def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA_DIR / "games.csv", low_memory=False)
    g = g[g.season >= FIRST_SEASON].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(norm_abbr)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["played"] = g["result"].notna()
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_pbp(seasons: list[int] | None = None) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("pbp_*.parquet"))
    frames = []
    for f in files:
        y = int(f.stem.split("_")[1])
        if y < FIRST_SEASON or (seasons and y not in seasons):
            continue
        frames.append(pd.read_parquet(f, columns=PBP_COLS))
    p = pd.concat(frames, ignore_index=True)
    p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna()].copy()
    p["posteam"] = p["posteam"].map(norm_abbr)
    p["defteam"] = p["defteam"].map(norm_abbr)
    return p


def load_player_stats() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(DATA_DIR.glob("stats_player_week_*.parquet"))]
    s = pd.concat(frames, ignore_index=True)
    s = s[s.season_type == "REG"].copy() if "season_type" in s else s
    s["team"] = s["team"].map(norm_abbr)
    s["opponent_team"] = s["opponent_team"].map(norm_abbr)
    return s
