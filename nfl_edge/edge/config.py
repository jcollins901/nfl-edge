"""Central settings. Edit these to tune how picky the model is."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
BET_LOG = ROOT / "bet_log.csv"
ADJUSTMENTS_FILE = ROOT / "adjustments.csv"

# Seasons of play-by-play used to train the game model.
FIRST_SEASON = 2016

# --- Team rating settings -------------------------------------------------
# How fast team ratings react to new games (bigger = more reactive).
RATING_ALPHA = 0.12
# Share of last season's rating kept at the start of a new season.
SEASON_CARRYOVER = 0.60

# --- Your sportsbooks -------------------------------------------------------
# Only bets at these books are recommended. All books (including ones you can't use, like
# Pinnacle) are still used to work out the fair price. Override with --books.
MY_BOOKS = ["draftkings"]

# --- Bet selection defaults -------------------------------------------------
MIN_EDGE = 0.03          # model win prob must beat the de-vigged market by 3+ pts
MIN_EV = 0.02            # expected profit of at least 2 cents per $1 risked
KELLY_FRACTION = 0.25    # quarter-Kelly: full Kelly is far too aggressive
MAX_STAKE_PCT = 0.02     # never suggest more than 2% of bankroll on one bet

# --- Props -------------------------------------------------------------------
PROP_HALF_LIFE_GAMES = 5.0     # recency weighting for player averages
PROP_PRIOR_SEASON_WEIGHT = 0.5  # last-season games count half as much
PROP_MIN_GAMES = 3
# How much the model's projection counts vs. the sportsbooks' consensus projection
# (0 = trust books only, 1 = trust model only). This is a judgment call: there is no free
# history of prop lines to fit it on. Track your results (grade.py) and adjust.
PROP_MODEL_WEIGHT = 0.4
# If model and books disagree by more than this share, it is usually injury or role news
# the model can't see. Those bets are flagged "news_risk" and excluded from picks.
PROP_MAX_DISAGREEMENT = 0.25
