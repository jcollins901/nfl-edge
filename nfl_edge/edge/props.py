"""Player prop projections.

Each stat is split into VOLUME (targets, carries, pass attempts) and EFFICIENCY
(yards per target, catch rate, TD rate...). Volume is fairly stable week to week, so it
uses the player's recent games. Efficiency is noisy, so it is pulled strongly toward the
position average until the player has a large sample. Then two adjustments:
  * opponent: how much this defense has allowed to the position, vs league average
  * game environment: the team's implied points from the betting line vs usual
Finally a fitted distribution turns the projection into P(over) / P(under).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from .config import (PROP_HALF_LIFE_GAMES, PROP_MIN_GAMES, PROP_PRIOR_SEASON_WEIGHT,
                     DATA_DIR)

SKILL = ["QB", "RB", "WR", "TE"]
WINDOW = 20  # games of history considered

# market -> (volume column, numerator column, prior strength in volume units, positions)
MARKETS = {
    "pass_yds":    ("attempts", "passing_yards", 150, ["QB"]),
    "pass_tds":    ("attempts", "passing_tds", 300, ["QB"]),
    "completions": ("attempts", "completions", 150, ["QB"]),
    "pass_att":    ("attempts", None, 0, ["QB"]),
    "rush_yds":    ("carries", "rushing_yards", 60, ["QB", "RB", "WR"]),
    "rush_att":    ("carries", None, 0, ["QB", "RB"]),
    "rec_yds":     ("targets", "receiving_yards", 40, ["RB", "WR", "TE"]),
    "receptions":  ("targets", "receptions", 40, ["RB", "WR", "TE"]),
    "anytime_td":  (None, None, 0, ["QB", "RB", "WR", "TE"]),
}
# which markets are counts (discrete distributions) vs yardage (continuous)
COUNT_MARKETS = {"pass_tds", "completions", "receptions", "pass_att", "rush_att"}
# how strongly the betting-market environment (implied team points) moves each market
# (exponents chosen by walk-forward test on 2023-2026 games; see backtest.py)
ENV_EXP = {"pass_yds": 0.25, "pass_tds": 0.75, "completions": 0.0, "pass_att": 0.1,
           "rush_yds": 0.25, "rush_att": 0.25, "rec_yds": 0.25, "receptions": 0.0,
           "anytime_td": 1.0}
# opponent defense stat used for each market and which positions it tracks
# (receiving and TD opponent adjustments tested as noise, so they are left out)
OPP_STAT = {"pass_yds": ("passing_yards", "QB"), "rush_yds": ("rushing_yards", "RB")}
OPP_SHRINK = 0.5


def relevant(p: pd.DataFrame, market: str) -> pd.Series:
    """Players a sportsbook would actually post this prop for (starters / real roles)."""
    pos = p.position
    att = p.get("mu_pass_att", pd.Series(0, index=p.index)).fillna(0)
    car = p.get("mu_rush_att", pd.Series(0, index=p.index)).fillna(0)
    rec = p.get("mu_receptions", pd.Series(0, index=p.index)).fillna(0)
    if market in ("pass_yds", "pass_tds", "completions", "pass_att"):
        return pos.eq("QB") & (att >= 20)
    if market in ("rush_yds", "rush_att"):
        return (pos.eq("RB") & (car >= 5)) | (pos.eq("QB") & (att >= 20))
    if market in ("rec_yds", "receptions"):
        return pos.isin(["WR", "TE", "RB"]) & (rec >= 1.5)
    if market == "anytime_td":
        return (pos.isin(["RB", "WR", "TE"]) & ((car + rec) >= 2.5))
    return pd.Series(True, index=p.index)


def _implied_points(games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "home_team", "away_team", "spread_line", "total_line"]].dropna()
    home = pd.DataFrame({"game_id": g.game_id, "team": g.home_team,
                         "implied": (g.total_line + g.spread_line) / 2})
    away = pd.DataFrame({"game_id": g.game_id, "team": g.away_team,
                         "implied": (g.total_line - g.spread_line) / 2})
    return pd.concat([home, away])


def prepare(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    s = stats[stats.position.isin(SKILL)].copy()
    s["tds"] = s.rushing_tds.fillna(0) + s.receiving_tds.fillna(0)
    s = s.merge(_implied_points(games), on=["game_id", "team"], how="left")
    order = games[["game_id", "gameday"]]
    s = s.merge(order, on="game_id", how="left")
    s["gameday"] = s["gameday"].fillna(pd.to_datetime(s.season.astype(str) + "-09-01")
                                       + pd.to_timedelta(s.week * 7, "D"))
    return s.sort_values(["gameday", "player_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- defense factors
def defense_factors(s: pd.DataFrame, alpha: float = 0.15, carry: float = 0.6) -> pd.DataFrame:
    """Pre-game, recency-weighted 'allowed vs league average' ratio per defense and stat."""
    keys = []
    for m, (col, pos) in OPP_STAT.items():
        keys.append((m, col, pos))
    rows = []
    allowed = {}
    real = s[~s["future"].astype(bool)] if "future" in s else s  # future rows: no stats yet
    for m, col, pos in keys:
        sub = real if pos is None else real[real.position == pos]
        allowed[m] = sub.groupby(["game_id", "opponent_team"])[col].sum()
    games = s[["game_id", "season", "gameday", "team", "opponent_team"]].drop_duplicates(
        ["game_id", "team"]).sort_values("gameday")
    state = {}
    lg_hist = {m: [] for m, _, _ in keys}
    for g in games.itertuples(index=False):
        d = g.opponent_team  # the defense facing `team`
        st_ = state.setdefault(d, {"season": g.season, "v": {}})
        if st_["season"] != g.season:
            for m in st_["v"]:
                lg = np.mean(lg_hist[m][-500:]) if lg_hist[m] else st_["v"][m]
                st_["v"][m] = lg + carry * (st_["v"][m] - lg)
            st_["season"] = g.season
        row = {"game_id": g.game_id, "team": g.team}
        for m, _, _ in keys:
            lg = np.mean(lg_hist[m][-500:]) if lg_hist[m] else np.nan
            v = st_["v"].get(m)
            row[f"opp_{m}"] = (v / lg) if (v is not None and lg and lg > 0) else 1.0
        rows.append(row)
        for m, _, _ in keys:
            x = allowed[m].get((g.game_id, d))
            if x is None or pd.isna(x):
                continue
            lg_hist[m].append(x)
            v = st_["v"].get(m, x)
            st_["v"][m] = v + alpha * (x - v)
    f = pd.DataFrame(rows)
    for m in OPP_STAT:
        f[f"opp_{m}"] = 1 + OPP_SHRINK * (f[f"opp_{m}"] - 1)
    return f


# ---------------------------------------------------------------- projections
def _pos_priors(s: pd.DataFrame) -> dict:
    pri = {}
    for m, (vol, num, _, _) in MARKETS.items():
        if vol is None or num is None:
            continue
        for pos, d in s.groupby("position"):
            v = d[vol].sum()
            if v > 0:
                pri[(m, pos)] = d[num].sum() / v
    for pos, d in s.groupby("position"):
        pri[("rush_td", pos)] = d.rushing_tds.sum() / max(d.carries.sum(), 1)
        pri[("rec_td", pos)] = d.receiving_tds.sum() / max(d.targets.sum(), 1)
    return pri


def project_rows(hist: pd.DataFrame, targets: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """For each target row (player, season, week, implied points), project every market
    using only that player's games strictly before the target game."""
    hl = PROP_HALF_LIFE_GAMES
    by_player = {pid: d for pid, d in hist.groupby("player_id")}
    out = []
    for t in targets.itertuples(index=False):
        h = by_player.get(t.player_id)
        if h is None:
            continue
        h = h[h.gameday < t.gameday].tail(WINDOW)
        if len(h) < PROP_MIN_GAMES:
            continue
        ago = np.arange(len(h))[::-1]
        w = 0.5 ** (ago / hl) * np.where(h.season.values == t.season, 1.0,
                                         PROP_PRIOR_SEASON_WEIGHT)
        W = w.sum()
        pos = t.position
        row = {"player_id": t.player_id, "game_id": t.game_id, "n_games": len(h),
               "games_this_season": int((h.season == t.season).sum())}
        vols = {c: float((h[c].fillna(0).values * w).sum() / W)
                for c in ("attempts", "carries", "targets")}
        imp_hist = h.implied.values
        env = np.nan
        if not pd.isna(getattr(t, "implied", np.nan)) and np.isfinite(imp_hist).any():
            ok = np.isfinite(imp_hist)
            env = t.implied / ((imp_hist[ok] * w[ok]).sum() / w[ok].sum())
        row["env"] = env
        for m, (vol, num, k, poss) in MARKETS.items():
            if pos not in poss:
                continue
            if m == "anytime_td":
                rr = _rate(h, w, "rushing_tds", "carries", priors.get(("rush_td", pos), 0), 60)
                rc = _rate(h, w, "receiving_tds", "targets", priors.get(("rec_td", pos), 0), 60)
                lam = rr * vols["carries"] + rc * vols["targets"]
                if pos == "QB":
                    lam = rr * vols["carries"]
                row[f"mu_{m}"] = lam
                continue
            v = vols[vol]
            if num is None:
                row[f"mu_{m}"] = v
                continue
            rate = _rate(h, w, num, vol, priors.get((m, pos), 0), k)
            row[f"mu_{m}"] = v * rate
        out.append(row)
    return pd.DataFrame(out)


def _rate(h, w, num, den, prior, k):
    n = (h[num].fillna(0).values * w).sum()
    d = (h[den].fillna(0).values * w).sum()
    return (n + k * prior) / (d + k) if (d + k) > 0 else prior


def apply_adjustments(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    env = p["env"].clip(0.6, 1.6).fillna(1.0)
    for m in MARKETS:
        c = f"mu_{m}"
        if c not in p:
            continue
        adj = env ** ENV_EXP[m]
        if f"opp_{m}" in p:
            adj = adj * p[f"opp_{m}"].fillna(1.0)
        p[c] = p[c] * adj
    return p


# ---------------------------------------------------------------- distributions
@dataclass
class Dispersion:
    """Spread of outcomes around the projection, fitted from history.
    sd = a * mu ** b  (yards) ; var = c * mu + d * mu^2 (counts)"""
    params: dict

    def save(self, path=DATA_DIR / "prop_dispersion.json"):
        path.write_text(json.dumps(self.params, indent=1))

    @classmethod
    def load(cls, path=DATA_DIR / "prop_dispersion.json"):
        return cls(json.loads(path.read_text()))

    @classmethod
    def fit(cls, proj: pd.DataFrame, actual: pd.DataFrame) -> "Dispersion":
        params = {}
        for m in MARKETS:
            c = f"mu_{m}"
            if c not in proj or f"y_{m}" not in actual:
                continue
            keep = relevant(proj, m).values
            d = pd.DataFrame({"mu": proj[c].values[keep],
                              "y": actual[f"y_{m}"].values[keep]}).dropna()
            d = d[d.mu > 0.05]
            if len(d) < 200:
                continue
            d["bin"] = pd.qcut(d.mu, 12, duplicates="drop")
            b = d.groupby("bin", observed=True).agg(mu=("mu", "mean"), var=("y", "var"),
                                                    sd=("y", "std"))
            if m == "anytime_td":
                continue
            if m in COUNT_MARKETS:
                X = np.column_stack([b.mu, b.mu ** 2])
                coef, *_ = np.linalg.lstsq(X, b["var"].values, rcond=None)
                params[m] = {"kind": "count", "c": float(max(coef[0], 0.05)),
                             "d": float(max(coef[1], 0.0))}
            else:
                bb, la = np.polyfit(np.log(b.mu), np.log(b.sd), 1)
                params[m] = {"kind": "yards", "a": float(np.exp(la)), "b": float(bb)}
        return cls(params)

    def prob_over(self, market: str, mu: float, line: float) -> tuple[float, float]:
        """Returns (P(over), P(push))."""
        if mu is None or not np.isfinite(mu) or mu <= 0:
            return np.nan, np.nan
        if market == "anytime_td":
            # line is 0.5 for 'anytime'; Poisson in expected TDs
            p0 = np.exp(-mu)
            return 1 - p0, 0.0
        prm = self.params.get(market)
        if prm is None:
            return np.nan, np.nan
        if prm["kind"] == "yards":
            sd = prm["a"] * mu ** prm["b"]
            shape = (mu / sd) ** 2
            dist = st.gamma(a=shape, scale=mu / shape)
            if float(line).is_integer():
                return 1 - dist.cdf(line + 0.5), dist.cdf(line + 0.5) - dist.cdf(line - 0.5)
            return 1 - dist.cdf(line), 0.0
        var = prm["c"] * mu + prm["d"] * mu ** 2
        if var > mu * 1.02:  # over-dispersed -> negative binomial
            r = mu ** 2 / (var - mu)
            dist = st.nbinom(r, r / (r + mu))
        elif var < mu * 0.98:  # under-dispersed -> binomial
            p = 1 - var / mu
            n = max(int(round(mu / p)), 1)
            dist = st.binom(n, min(mu / n, 1.0))
        else:
            dist = st.poisson(mu)
        k = np.floor(line)
        if float(line).is_integer():
            return float(dist.sf(line)), float(dist.pmf(line))
        return float(dist.sf(k)), 0.0


    def implied_mu(self, market: str, line: float, p_over: float) -> float:
        """The projection a sportsbook's no-vig price implies (inverts prob_over)."""
        if not (0.02 < p_over < 0.98):
            return np.nan
        if market == "anytime_td":
            return -np.log(1 - p_over)
        lo, hi = 1e-3, max(line * 4, 5.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            po, pp = self.prob_over(market, mid, line)
            po = po / (1 - pp) if pp < 1 else po
            if np.isnan(po):
                return np.nan
            lo, hi = (mid, hi) if po < p_over else (lo, mid)
        return (lo + hi) / 2


ACTUAL_COLS = {"pass_yds": "passing_yards", "pass_tds": "passing_tds",
               "completions": "completions", "pass_att": "attempts",
               "rush_yds": "rushing_yards", "rush_att": "carries",
               "rec_yds": "receiving_yards", "receptions": "receptions"}


def actuals(rows: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame({"player_id": rows.player_id.values})
    for m, c in ACTUAL_COLS.items():
        a[f"y_{m}"] = rows[c].fillna(0).values
    a["y_anytime_td"] = (rows["tds"].fillna(0).values > 0).astype(float)
    return a
