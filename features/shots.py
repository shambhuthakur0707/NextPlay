# -*- coding: utf-8 -*-
"""
NextPlay -- Shot Profile Features
==================================
Merges team shot profile data onto games and creates
clash features (what happens when different styles meet).
"""

import pandas as pd


def merge_shot_profiles(games, shot_df):
    """
    Join each team's shot profile onto every game they play.

    Args:
        games: game-level DataFrame
        shot_df: shot profile DataFrame with TEAM_ABBR, SEASON columns

    Returns:
        DataFrame with shot profile + clash features added
    """
    shot_home = shot_df.rename(columns={
        "TEAM_ABBR": "HOME_TEAM",
        "PAINT_RATE": "HOME_PAINT_RATE",
        "THREEPT_RATE": "HOME_3PT_RATE",
        "MIDRANGE_RATE": "HOME_MID_RATE",
        "PAINT_PCT": "HOME_PAINT_PCT",
        "THREEPT_PCT": "HOME_3PT_PCT",
        "PTS_PER_SHOT": "HOME_PTS_PER_SHOT",
    })

    shot_away = shot_df.rename(columns={
        "TEAM_ABBR": "AWAY_TEAM",
        "PAINT_RATE": "AWAY_PAINT_RATE",
        "THREEPT_RATE": "AWAY_3PT_RATE",
        "MIDRANGE_RATE": "AWAY_MID_RATE",
        "PAINT_PCT": "AWAY_PAINT_PCT",
        "THREEPT_PCT": "AWAY_3PT_PCT",
        "PTS_PER_SHOT": "AWAY_PTS_PER_SHOT",
    })

    gws = games.merge(
        shot_home[[
            "HOME_TEAM", "SEASON", "HOME_PAINT_RATE", "HOME_3PT_RATE",
            "HOME_MID_RATE", "HOME_PAINT_PCT", "HOME_3PT_PCT",
            "HOME_PTS_PER_SHOT",
        ]],
        on=["HOME_TEAM", "SEASON"],
        how="left",
    ).merge(
        shot_away[[
            "AWAY_TEAM", "SEASON", "AWAY_PAINT_RATE", "AWAY_3PT_RATE",
            "AWAY_MID_RATE", "AWAY_PAINT_PCT", "AWAY_3PT_PCT",
            "AWAY_PTS_PER_SHOT",
        ]],
        on=["AWAY_TEAM", "SEASON"],
        how="left",
    )

    # Style clash features
    gws["COMBINED_3PT_RATE"] = gws["HOME_3PT_RATE"] + gws["AWAY_3PT_RATE"]
    gws["COMBINED_PAINT_RATE"] = gws["HOME_PAINT_RATE"] + gws["AWAY_PAINT_RATE"]
    gws["COMBINED_PTS_PER_SHOT"] = (
        gws["HOME_PTS_PER_SHOT"] + gws["AWAY_PTS_PER_SHOT"]
    )
    gws["STYLE_MISMATCH"] = gws["HOME_3PT_RATE"] - gws["AWAY_3PT_RATE"]
    gws["EFFICIENCY_EDGE"] = (
        gws["HOME_PTS_PER_SHOT"] - gws["AWAY_PTS_PER_SHOT"]
    )

    return gws
