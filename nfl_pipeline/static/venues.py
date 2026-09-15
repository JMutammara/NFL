"""Static venue geography: team home stadiums by season, neutral sites, time zones.

Coordinates are stadium centroids (decimal degrees). ``tz`` is the standard-time
UTC offset of the team's home market (used for body-clock/kickoff features;
DST shifts affect both teams equally within CONUS, Arizona excepted, so a
standard-time offset is adequate). ``alt_m`` is elevation in metres.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    lat: float
    lon: float
    tz: float
    alt_m: float = 0.0


# (first_season, last_season) -> venue. Use None for open-ended.
TEAM_VENUES: dict[str, list[tuple[int | None, int | None, Venue]]] = {
    "ARI": [(None, None, Venue(33.5276, -112.2626, -7.0, 330))],
    "ATL": [(None, None, Venue(33.7554, -84.4010, -5.0, 300))],
    "BAL": [(None, None, Venue(39.2780, -76.6227, -5.0, 10))],
    "BUF": [(None, None, Venue(42.7738, -78.7870, -5.0, 180))],
    "CAR": [(None, None, Venue(35.2258, -80.8528, -5.0, 220))],
    "CHI": [(None, None, Venue(41.8623, -87.6167, -6.0, 180))],
    "CIN": [(None, None, Venue(39.0954, -84.5160, -5.0, 150))],
    "CLE": [(None, None, Venue(41.5061, -81.6995, -5.0, 180))],
    "DAL": [(None, None, Venue(32.7473, -97.0945, -6.0, 170))],
    "DEN": [(None, None, Venue(39.7439, -105.0201, -7.0, 1609))],
    "DET": [(None, None, Venue(42.3400, -83.0456, -5.0, 180))],
    "GB":  [(None, None, Venue(44.5013, -88.0622, -6.0, 200))],
    "HOU": [(None, None, Venue(29.6847, -95.4107, -6.0, 15))],
    "IND": [(None, None, Venue(39.7601, -86.1639, -5.0, 220))],
    "JAX": [(None, None, Venue(30.3240, -81.6373, -5.0, 5))],
    "KC":  [(None, None, Venue(39.0489, -94.4839, -6.0, 270))],
    "LV":  [(None, 2019, Venue(37.7516, -122.2005, -8.0, 5)),      # Oakland Coliseum
            (2020, None, Venue(36.0909, -115.1833, -8.0, 600))],   # Allegiant
    "LAC": [(None, 2016, Venue(32.7831, -117.1196, -8.0, 90)),     # Qualcomm
            (2017, 2019, Venue(33.8644, -118.2611, -8.0, 20)),     # Dignity Health SP
            (2020, None, Venue(33.9535, -118.3392, -8.0, 30))],    # SoFi
    "LA":  [(None, 2015, Venue(38.6328, -90.1885, -6.0, 140)),     # Edward Jones Dome
            (2016, 2019, Venue(34.0141, -118.2879, -8.0, 50)),     # LA Coliseum
            (2020, None, Venue(33.9535, -118.3392, -8.0, 30))],    # SoFi
    "MIA": [(None, None, Venue(25.9580, -80.2389, -5.0, 3))],
    "MIN": [(None, None, Venue(44.9736, -93.2575, -6.0, 250))],
    "NE":  [(None, None, Venue(42.0909, -71.2643, -5.0, 90))],
    "NO":  [(None, None, Venue(29.9511, -90.0812, -6.0, 1))],
    "NYG": [(None, None, Venue(40.8135, -74.0745, -5.0, 2))],
    "NYJ": [(None, None, Venue(40.8135, -74.0745, -5.0, 2))],
    "PHI": [(None, None, Venue(39.9008, -75.1675, -5.0, 5))],
    "PIT": [(None, None, Venue(40.4468, -80.0158, -5.0, 220))],
    "SF":  [(None, 2013, Venue(37.7136, -122.3861, -8.0, 5)),      # Candlestick
            (2014, None, Venue(37.4030, -121.9700, -8.0, 5))],     # Levi's
    "SEA": [(None, None, Venue(47.5952, -122.3316, -8.0, 5))],
    "TB":  [(None, None, Venue(27.9759, -82.5033, -5.0, 10))],
    "TEN": [(None, None, Venue(36.1665, -86.7713, -6.0, 130))],
    "WAS": [(None, None, Venue(38.9076, -76.8645, -5.0, 60))],
}

# Neutral / international venues matched by case-insensitive substring of the
# schedule's ``stadium`` field.
NEUTRAL_VENUES: list[tuple[str, Venue]] = [
    ("wembley", Venue(51.5560, -0.2795, 0.0, 50)),
    ("tottenham", Venue(51.6042, -0.0662, 0.0, 40)),
    ("twickenham", Venue(51.4559, -0.3415, 0.0, 15)),
    ("azteca", Venue(19.3029, -99.1505, -6.0, 2200)),
    ("allianz", Venue(48.2188, 11.6247, 1.0, 500)),
    ("deutsche bank", Venue(50.0686, 8.6455, 1.0, 100)),
    ("frankfurt", Venue(50.0686, 8.6455, 1.0, 100)),
    ("corinthians", Venue(-23.5453, -46.4742, -3.0, 750)),
    ("neo qu", Venue(-23.5453, -46.4742, -3.0, 750)),
    ("bernab", Venue(40.4531, -3.6883, 1.0, 650)),
    ("croke", Venue(53.3607, -6.2511, 0.0, 20)),
    ("melbourne", Venue(-37.8200, 144.9834, 10.0, 30)),
    ("aviva", Venue(53.3352, -6.2285, 0.0, 10)),
]


def team_venue(team: str, season: int) -> Venue | None:
    spans = TEAM_VENUES.get(team)
    if not spans:
        return None
    for first, last, v in spans:
        if (first is None or season >= first) and (last is None or season <= last):
            return v
    return spans[-1][2]


def neutral_venue(stadium_name: str | None) -> Venue | None:
    if not stadium_name:
        return None
    s = str(stadium_name).lower()
    for key, v in NEUTRAL_VENUES:
        if key in s:
            return v
    return None
