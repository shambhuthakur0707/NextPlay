# -*- coding: utf-8 -*-
"""
NextPlay -- Defensive Rolling Features
========================================
Rolling average of points allowed by each team.
Captures how good each team's defense is going into the game.
"""

import pandas as pd
from config import ROLLING_WINDOW


def add_defensive_features(games, window=ROLLING_WINDOW):
    """
    Add defensive rolling average features.

    For each team, track the rolling average of points scored
    by their opponents (= points they allow).

    Features added:
    - HOME_DEF_ROLL10, AWAY_DEF_ROLL10
    - COMBINED_DEF_ROLL10
    - EXPECTED_TOTAL (avg of offense + defense signals)
    """
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)

    # Build opponent-scored lookup
    home_def = df[["GAME_DATE", "GAME_ID", "HOME_TEAM", "AWAY_PTS"]].copy()
    home_def.columns = ["GAME_DATE", "GAME_ID", "TEAM", "OPP_PTS"]

    away_def = df[["GAME_DATE", "GAME_ID", "AWAY_TEAM", "HOME_PTS"]].copy()
    away_def.columns = ["GAME_DATE", "GAME_ID", "TEAM", "OPP_PTS"]

    all_def = pd.concat([home_def, away_def], ignore_index=True)
    all_def = all_def.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)

    # Rolling avg of points allowed (shifted -- no leakage)
    all_def["DEF_ROLL10"] = all_def.groupby("TEAM")["OPP_PTS"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )

    def_lookup = all_def[["GAME_ID", "TEAM", "DEF_ROLL10"]].drop_duplicates()

    # Merge home defense
    df = df.merge(
        def_lookup.rename(
            columns={"TEAM": "HOME_TEAM", "DEF_ROLL10": "HOME_DEF_ROLL10"}
        ),
        on=["GAME_ID", "HOME_TEAM"],
        how="left",
    )

    # Merge away defense
    df = df.merge(
        def_lookup.rename(
            columns={"TEAM": "AWAY_TEAM", "DEF_ROLL10": "AWAY_DEF_ROLL10"}
        ),
        on=["GAME_ID", "AWAY_TEAM"],
        how="left",
    )

    # Combined defensive signal
    df["COMBINED_DEF_ROLL10"] = df["HOME_DEF_ROLL10"] + df["AWAY_DEF_ROLL10"]

    # Expected total: avg of offense and defense signals
    df["EXPECTED_TOTAL"] = (
        df["HOME_ROLL10_PTS"]
        + df["AWAY_ROLL10_PTS"]
        + df["HOME_DEF_ROLL10"]
        + df["AWAY_DEF_ROLL10"]
    ) / 2

    return df
