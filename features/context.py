# -*- coding: utf-8 -*-
"""
NextPlay -- Season Context & Home Court Features
=================================================
Season stage (early/mid/late), home court advantage,
and opponent strength rolling features.
"""

import pandas as pd


def add_context_features(games):
    """
    Add season context and home court features.

    Features added:
    - SEASON_STAGE: 1=early(1-27), 2=mid(28-55), 3=late(56-82)
    - HOME_COURT_STRENGTH: rolling home win rate
    - AWAY_TEAM_STRENGTH: rolling away win rate
    - HOME_ADVANTAGE_SCORE: home strength vs league avg
    - STRENGTH_DIFF: home vs away strength gap
    """
    df = games.copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Season stage
    df["SEASON_GAME_NUM"] = (
        df.groupby(["SEASON", "HOME_TEAM"]).cumcount() + 1
    )
    df["SEASON_STAGE"] = pd.cut(
        df["SEASON_GAME_NUM"],
        bins=[0, 27, 55, 82],
        labels=[1, 2, 3],
    ).astype(float)

    # Home court strength
    df["HOME_WIN_INT"] = (df["HOME_WL"] == "W").astype(int)
    df["HOME_COURT_STRENGTH"] = df.groupby("HOME_TEAM")[
        "HOME_WIN_INT"
    ].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())

    # League average home win rate
    league_avg = df["HOME_WIN_INT"].mean()
    df["HOME_ADVANTAGE_SCORE"] = df["HOME_COURT_STRENGTH"] - league_avg

    # Opponent (away) strength
    df["AWAY_WIN_INT"] = (df["AWAY_WL"] == "W").astype(int)
    df["AWAY_TEAM_STRENGTH"] = df.groupby("AWAY_TEAM")[
        "AWAY_WIN_INT"
    ].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())

    # Strength matchup
    df["STRENGTH_DIFF"] = (
        df["HOME_COURT_STRENGTH"] - df["AWAY_TEAM_STRENGTH"]
    )

    return df
