"""Odds math: implied probability, removing the vig, expected value, Kelly sizing."""
from __future__ import annotations

import math


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def american_to_prob(odds: float) -> float:
    """Break-even win rate for a price, vig included (-110 -> 52.4%)."""
    return 1 / american_to_decimal(odds)


def prob_to_american(p: float) -> float:
    """Fair (no-vig) American odds for a win probability."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def fmt_american(odds: float) -> str:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return ""
    o = int(round(odds))
    return f"+{o}" if o > 0 else str(o)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Strip the bookmaker margin from a two-sided market.

    Uses the multiplicative method: scale both implied probabilities so they sum to 1.
    Returns the market's 'fair' probability for each side.
    """
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def expected_value(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Expected profit per $1 staked. Pushes return the stake (zero profit)."""
    p_loss = 1 - p_win - p_push
    return p_win * (american_to_decimal(odds) - 1) - p_loss


def kelly_fraction(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """Full-Kelly bankroll fraction; 0 if the bet has no edge."""
    b = american_to_decimal(odds) - 1
    p_loss = 1 - p_win - p_push
    f = (b * p_win - p_loss) / b
    return max(f, 0.0)
