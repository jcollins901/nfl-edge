"""Team name <-> nflverse abbreviation mapping."""

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

NAME_TO_ABBR = {v: k for k, v in TEAM_NAMES.items()}
# Older names that can show up in historical data or at some books.
NAME_TO_ABBR.update({
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
    "Oakland Raiders": "LV", "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
})

# nflverse used different codes for relocated teams in older seasons.
LEGACY_ABBR = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX", "LAR": "LA", "WSH": "WAS"}


def norm_abbr(abbr: str) -> str:
    return LEGACY_ABBR.get(abbr, abbr)


def to_abbr(name: str) -> str:
    if name in NAME_TO_ABBR:
        return NAME_TO_ABBR[name]
    if name in TEAM_NAMES:
        return name
    raise KeyError(f"Unknown team name: {name!r}")
