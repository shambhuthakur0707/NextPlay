# -*- coding: utf-8 -*-
"""
NextPlay -- Exponentially Weighted Moving Average Features
===========================================================
EWM gives more weight to recent games -- the model reacts
faster to current form compared to simple rolling averages.
"""

import pandas as pd
from config import EWM_SPAN


def add_ewm_features(games, span=EWM_SPAN):
    """
    Add exponentially weighted moving average features.

    Features added:
    - HOME/AWAY_EWM_PTS, EWM_FG_PCT, EWM_FG3_PCT, EWM_DEF
    - EWM_EXPECTED_TOTAL
    - HOME/AWAY_MOMENTUM (EWM - rolling avg = recent trend)
    - COMBINED_MOMENTUM
    """
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)

    ewm_stats = {
        "HOME_TEAM": [
            ("HOME_PTS", "HOME_EWM_PTS"),
            ("HOME_FG_PCT", "HOME_EWM_FG_PCT"),
            ("HOME_FG3_PCT", "HOME_EWM_FG3_PCT"),
            ("AWAY_PTS", "HOME_EWM_DEF"),  # pts allowed at home
        ],
        "AWAY_TEAM": [
            ("AWAY_PTS", "AWAY_EWM_PTS"),
            ("AWAY_FG_PCT", "AWAY_EWM_FG_PCT"),
            ("AWAY_FG3_PCT", "AWAY_EWM_FG3_PCT"),
            ("HOME_PTS", "AWAY_EWM_DEF"),  # pts allowed on road
        ],
    }

    for team_col, stat_pairs in ewm_stats.items():
        group = df.groupby(team_col)
        for source_col, new_col in stat_pairs:
            df[new_col] = group[source_col].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
            )

    # EWM expected total
    df["EWM_EXPECTED_TOTAL"] = (
        df["HOME_EWM_PTS"]
        + df["AWAY_EWM_PTS"]
        + df["HOME_EWM_DEF"]
        + df["AWAY_EWM_DEF"]
    ) / 2

    # Momentum = EWM - rolling avg (positive = scoring MORE recently)
    df["HOME_MOMENTUM"] = df["HOME_EWM_PTS"] - df["HOME_ROLL10_PTS"]
    df["AWAY_MOMENTUM"] = df["AWAY_EWM_PTS"] - df["AWAY_ROLL10_PTS"]
    df["COMBINED_MOMENTUM"] = df["HOME_MOMENTUM"] + df["AWAY_MOMENTUM"]

    return df
