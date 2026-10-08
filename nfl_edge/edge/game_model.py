"""Game model: predicts home margin and total points, then turns them into
probabilities for spreads, moneylines and totals.

Two layers:
 1. A ridge regression on opponent-adjusted team ratings (the "model" opinion).
 2. A blend with the betting market. Closing lines are very efficient, so the model
    is only allowed to move the market number by the weight that historically helped
    out-of-sample. That weight is learned, not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

MARGIN_FEATURES = ["d_off_epa", "d_def_epa", "d_off_pass", "d_def_pass", "d_off_rush",
                   "d_def_rush", "d_net_pts", "hfa", "rest_diff"]
TOTAL_FEATURES = ["s_off_epa", "s_def_epa", "s_pf", "s_pa", "s_plays", "s_pass_rate", "dome"]
MARGINS = np.arange(-70, 71)


def make_features(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    df = games.merge(ratings, on="game_id", how="left")
    f = pd.DataFrame(index=df.index)
    f["d_off_epa"] = df.home_off_epa - df.away_off_epa
    f["d_def_epa"] = df.home_def_epa - df.away_def_epa
    f["d_off_pass"] = df.home_off_pass_epa - df.away_off_pass_epa
    f["d_def_pass"] = df.home_def_pass_epa - df.away_def_pass_epa
    f["d_off_rush"] = df.home_off_rush_epa - df.away_off_rush_epa
    f["d_def_rush"] = df.home_def_rush_epa - df.away_def_rush_epa
    f["d_net_pts"] = (df.home_pf - df.home_pa) - (df.away_pf - df.away_pa)
    f["hfa"] = 1 - df.neutral
    f["rest_diff"] = (df.home_rest - df.away_rest).clip(-7, 7).fillna(0)
    f["s_off_epa"] = df.home_off_epa + df.away_off_epa
    f["s_def_epa"] = df.home_def_epa + df.away_def_epa
    f["s_pf"] = df.home_pf + df.away_pf
    f["s_pa"] = df.home_pa + df.away_pa
    f["s_plays"] = df.home_plays + df.away_plays
    f["s_pass_rate"] = df.home_pass_rate + df.away_pass_rate
    f["dome"] = df.dome
    keep = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
            "result", "total", "spread_line", "total_line", "home_moneyline", "away_moneyline",
            "home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "played",
            "neutral", "home_n", "away_n"]
    return pd.concat([df[keep], f], axis=1)


def _fit_ridge(X, y, alpha=5.0):
    m = Ridge(alpha=alpha)
    m.fit(X, y)
    return m


def walk_forward(feats: pd.DataFrame, first_test: int) -> pd.DataFrame:
    """Out-of-sample predictions: for each season, train only on earlier seasons."""
    out = []
    played = feats[feats.played & feats.spread_line.notna()]
    for s in sorted(feats.season.unique()):
        if s < first_test:
            continue
        tr = played[played.season < s]
        te = feats[feats.season == s].copy()
        mm = _fit_ridge(tr[MARGIN_FEATURES], tr.result)
        tm = _fit_ridge(tr[TOTAL_FEATURES], tr.total)
        te["model_margin"] = mm.predict(te[MARGIN_FEATURES])
        te["model_total"] = tm.predict(te[TOTAL_FEATURES])
        out.append(te)
    return pd.concat(out)


def _key_number_weights(results: pd.Series, mus: pd.Series, sigma: float) -> np.ndarray:
    """NFL margins pile up on 3, 7, 6, 10, 14... A plain bell curve misses that.

    Compare how often each final margin actually happened against how often a smooth
    normal curve says it should, and keep the ratio as a per-margin weight.
    """
    absk = np.abs(MARGINS)
    emp = np.array([(results.abs() == k).mean() for k in range(71)])
    smooth = np.zeros(71)
    for mu in mus.values:
        p = norm.pdf(MARGINS, mu, sigma)
        p /= p.sum()
        smooth += np.bincount(absk, weights=p, minlength=71)[:71]
    smooth /= len(mus)
    n = len(results)
    # shrink toward 1 where there is little data
    w = (emp * n + 20 * smooth) / (smooth * n + 20 * smooth)
    w = np.clip(w, 0.02, 3.0)
    w[25:] = 1.0
    return w[absk]


@dataclass
class GameModel:
    margin_model: Ridge | None = None
    total_model: Ridge | None = None
    w_margin: float = 0.0     # how far the model may pull the spread (0 = pure market)
    w_total: float = 0.0
    sigma_margin: float = 13.3
    sigma_total: float = 13.5
    key_weights: np.ndarray = field(default_factory=lambda: np.ones(len(MARGINS)))

    # ---------- fitting ----------
    def fit(self, feats: pd.DataFrame, oof: pd.DataFrame) -> "GameModel":
        tr = feats[feats.played & feats.spread_line.notna()]
        self.margin_model = _fit_ridge(tr[MARGIN_FEATURES], tr.result)
        self.total_model = _fit_ridge(tr[TOTAL_FEATURES], tr.total)
        self.calibrate(oof)
        return self

    def calibrate(self, oof: pd.DataFrame) -> "GameModel":
        o = oof[oof.played & oof.spread_line.notna() & oof.total_line.notna()]
        o = o[(o.home_n + o.away_n) >= 0]
        self.w_margin = _blend_weight(o.result - o.spread_line, o.model_margin - o.spread_line)
        self.w_total = _blend_weight(o.total - o.total_line, o.model_total - o.total_line)
        mu_m = o.spread_line + self.w_margin * (o.model_margin - o.spread_line)
        mu_t = o.total_line + self.w_total * (o.model_total - o.total_line)
        self.sigma_margin = float(np.std(o.result - mu_m))
        self.sigma_total = float(np.std(o.total - mu_t))
        self.key_weights = _key_number_weights(o.result, mu_m, self.sigma_margin)
        return self

    # ---------- predicting ----------
    def predict(self, feats: pd.DataFrame, market_spread=None, market_total=None,
                margin_adj=None) -> pd.DataFrame:
        df = feats.copy()
        df["model_margin"] = self.margin_model.predict(df[MARGIN_FEATURES])
        df["model_total"] = self.total_model.predict(df[TOTAL_FEATURES])
        if margin_adj is not None:
            df["model_margin"] += margin_adj
        ms = df.spread_line if market_spread is None else market_spread
        mt = df.total_line if market_total is None else market_total
        df["mkt_margin"] = ms
        df["mkt_total"] = mt
        df["pred_margin"] = np.where(ms.notna(), ms + self.w_margin * (df.model_margin - ms),
                                     df.model_margin)
        df["pred_total"] = np.where(mt.notna(), mt + self.w_total * (df.model_total - mt),
                                    df.model_total)
        return df

    def margin_dist(self, mu: float) -> np.ndarray:
        p = norm.pdf(MARGINS, mu, self.sigma_margin) * self.key_weights
        return p / p.sum()

    def spread_probs(self, mu: float, home_line: float) -> tuple[float, float, float]:
        """P(home covers), P(push), P(away covers) where home_line is the home handicap
        in betting convention (e.g. -3.5 means home gives 3.5)."""
        p = self.margin_dist(mu)
        need = -home_line
        return p[MARGINS > need].sum(), p[MARGINS == need].sum(), p[MARGINS < need].sum()

    def moneyline_probs(self, mu: float) -> tuple[float, float, float]:
        p = self.margin_dist(mu)
        return p[MARGINS > 0].sum(), p[MARGINS == 0].sum(), p[MARGINS < 0].sum()

    def total_probs(self, mu: float, line: float) -> tuple[float, float, float]:
        """P(over), P(push), P(under)."""
        s = self.sigma_total
        if float(line).is_integer():
            p_push = norm.cdf(line + 0.5, mu, s) - norm.cdf(line - 0.5, mu, s)
            p_over = 1 - norm.cdf(line + 0.5, mu, s)
        else:
            p_push = 0.0
            p_over = 1 - norm.cdf(line, mu, s)
        return p_over, p_push, 1 - p_over - p_push


def _blend_weight(resid_market: pd.Series, model_minus_market: pd.Series) -> float:
    x, y = model_minus_market.values, resid_market.values
    w = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0
    return float(np.clip(w, 0.0, 1.0))
