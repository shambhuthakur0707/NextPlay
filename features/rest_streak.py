# -*- coding: utf-8 -*-
"""
NextPlay -- Rest Days & Win Streak Features
============================================
Calculates days between games (fatigue signal) and
running win/loss streak (momentum signal) for each team.
"""

from math import radians, cos, sin, asin, sqrt
import pandas as pd
from utils.helpers import compute_streak


TEAM_COORDS = {
    "BOS": (42.36, -71.06), "NYK": (40.75, -73.99),
    "PHI": (39.90, -75.17), "BKN": (40.68, -73.97),
    "TOR": (43.64, -79.38), "CHI": (41.88, -87.62),
    "MIL": (43.04, -87.92), "CLE": (41.50, -81.69),
    "IND": (39.76, -86.16), "DET": (42.34, -83.05),
    "ATL": (33.75, -84.40), "MIA": (25.78, -80.19),
    "CHA": (35.22, -80.84), "ORL": (28.54, -81.38),
    "WAS": (38.90, -77.02), "MEM": (35.14, -90.05),
    "NOP": (29.95, -90.08), "SAS": (29.43, -98.44),
    "HOU": (29.75, -95.36), "DAL": (32.79, -96.81),
    "DEN": (39.75, -104.99), "UTA": (40.77, -111.90),
    "MIN": (44.98, -93.27), "OKC": (35.46, -97.52),
    "POR": (45.53, -122.67), "GSW": (37.77, -122.39),
    "LAL": (34.04, -118.27), "LAC": (34.04, -118.27),
    "SAC": (38.58, -121.50), "PHX": (33.44, -112.07),
}


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    r_miles = 3956.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r_miles * asin(sqrt(a))


def get_travel_distance(away_team, home_team):
    """Distance in miles from away to home city."""
    if away_team not in TEAM_COORDS or home_team not in TEAM_COORDS:
        return 0.0
    away = TEAM_COORDS[away_team]
    home = TEAM_COORDS[home_team]
    return haversine(away[0], away[1], home[0], home[1])


def add_rest_and_streak(games):
    """
    Add rest days and win streak features to game dataset.

    Args:
        games: game-level DataFrame

    Returns:
        DataFrame with REST_DAYS and STREAK columns added
    """
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)

    # ?? TRAVEL DISTANCE ?????????????????????????????????????
    df["HOME_TRAVEL_DIST"] = 0.0
    df["AWAY_TRAVEL_DIST"] = df.apply(
        lambda r: get_travel_distance(r["AWAY_TEAM"], r["HOME_TEAM"]),
        axis=1,
    )
    df["TRAVEL_DIFF"] = df["HOME_TRAVEL_DIST"] - df["AWAY_TRAVEL_DIST"]

    # ?? REST DAYS ????????????????????????????????????????????
    home_app = df[["GAME_DATE", "GAME_ID", "HOME_TEAM"]].copy()
    home_app.columns = ["GAME_DATE", "GAME_ID", "TEAM"]

    away_app = df[["GAME_DATE", "GAME_ID", "AWAY_TEAM"]].copy()
    away_app.columns = ["GAME_DATE", "GAME_ID", "TEAM"]

    all_app = pd.concat([home_app, away_app], ignore_index=True)
    all_app = all_app.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)

    all_app["PREV_DATE"] = all_app.groupby("TEAM")["GAME_DATE"].shift(1)
    all_app["REST_DAYS"] = (
        all_app["GAME_DATE"] - all_app["PREV_DATE"]
    ).dt.days
    all_app["REST_DAYS"] = all_app["REST_DAYS"].clip(upper=7).fillna(3)

    rest_lookup = all_app[["GAME_ID", "TEAM", "REST_DAYS"]].drop_duplicates()

    # Merge home rest
    df = df.merge(
        rest_lookup.rename(
            columns={"TEAM": "HOME_TEAM", "REST_DAYS": "HOME_REST_DAYS"}
        ),
        on=["GAME_ID", "HOME_TEAM"],
        how="left",
    )
    # Merge away rest
    df = df.merge(
        rest_lookup.rename(
            columns={"TEAM": "AWAY_TEAM", "REST_DAYS": "AWAY_REST_DAYS"}
        ),
        on=["GAME_ID", "AWAY_TEAM"],
        how="left",
    )

    df["REST_DAYS_DIFF"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]

    # ?? WIN STREAK ???????????????????????????????????????????
    home_wl = df[["GAME_DATE", "GAME_ID", "HOME_TEAM", "HOME_WL"]].copy()
    home_wl.columns = ["GAME_DATE", "GAME_ID", "TEAM", "WL"]

    away_wl = df[["GAME_DATE", "GAME_ID", "AWAY_TEAM", "AWAY_WL"]].copy()
    away_wl.columns = ["GAME_DATE", "GAME_ID", "TEAM", "WL"]

    all_wl = pd.concat([home_wl, away_wl], ignore_index=True)
    all_wl = all_wl.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)

    streak_rows = []
    for _, group in all_wl.groupby("TEAM"):
        g = group.copy().reset_index(drop=True)
        g["STREAK"] = compute_streak(g["WL"].tolist())
        streak_rows.append(g)

    streak_df = pd.concat(streak_rows, ignore_index=True)
    streak_lookup = streak_df[["GAME_ID", "TEAM", "STREAK"]].drop_duplicates()

    # Merge home streak
    df = df.merge(
        streak_lookup.rename(
            columns={"TEAM": "HOME_TEAM", "STREAK": "HOME_STREAK"}
        ),
        on=["GAME_ID", "HOME_TEAM"],
        how="left",
    )
    # Merge away streak
    df = df.merge(
        streak_lookup.rename(
            columns={"TEAM": "AWAY_TEAM", "STREAK": "AWAY_STREAK"}
        ),
        on=["GAME_ID", "AWAY_TEAM"],
        how="left",
    )

    df["STREAK_DIFF"] = df["HOME_STREAK"] - df["AWAY_STREAK"]

    return df
