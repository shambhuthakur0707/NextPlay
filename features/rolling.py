# -*- coding: utf-8 -*-
"""
NextPlay -- Rolling Average Features
=====================================
For each game, calculates each team's recent form using
their last N games before that game.

KEY RULE: shift(1) before rolling -- never include the
current game in its own features (prevents data leakage).
"""

import pandas as pd
from config import ROLLING_WINDOW


def add_rolling_features(games, window=ROLLING_WINDOW):
    """
    Add rolling averages of the last N games for both teams.

    Args:
        games: game-level DataFrame
        window: number of games to average (default: 10)

    Returns:
        DataFrame with rolling feature columns added
    """
    df = games.copy()

    # Shooting efficiency (per-game)
    df["HOME_EFG"] = (df["HOME_FGM"] + 0.5 * df["HOME_FG3M"]) / df["HOME_FGA"]
    df["AWAY_EFG"] = (df["AWAY_FGM"] + 0.5 * df["AWAY_FG3M"]) / df["AWAY_FGA"]
    df["HOME_TS"] = df["HOME_PTS"] / (2 * (df["HOME_FGA"] + 0.44 * df["HOME_FTA"]))
    df["AWAY_TS"] = df["AWAY_PTS"] / (2 * (df["AWAY_FGA"] + 0.44 * df["AWAY_FTA"]))

    # Possession estimates (per-game)
    df["HOME_POSS_EST"] = (
        df["HOME_FGA"] + 0.44 * df["HOME_FTA"]
        - df["HOME_OREB"] + df["HOME_TOV"]
    )
    df["AWAY_POSS_EST"] = (
        df["AWAY_FGA"] + 0.44 * df["AWAY_FTA"]
        - df["AWAY_OREB"] + df["AWAY_TOV"]
    )
    df["COMBINED_POSS"] = df["HOME_POSS_EST"] + df["AWAY_POSS_EST"]

    # Stats to compute rolling averages for
    home_stats = [
        "HOME_PTS", "HOME_FG_PCT", "HOME_FG3_PCT", "HOME_FT_PCT",
        "HOME_FG3A", "HOME_FTA", "HOME_AST", "HOME_TOV",
        "HOME_OREB", "HOME_REB", "HOME_PF", "TOTAL_PTS",
        "HOME_POSS_EST",
    ]
    away_stats = [
        "AWAY_PTS", "AWAY_FG_PCT", "AWAY_FG3_PCT", "AWAY_FT_PCT",
        "AWAY_FG3A", "AWAY_FTA", "AWAY_AST", "AWAY_TOV",
        "AWAY_OREB", "AWAY_REB", "AWAY_PF",
        "AWAY_POSS_EST",
    ]

    # Home team rolling features
    for stat in home_stats:
        col_name = f'HOME_ROLL10_{stat.replace("HOME_", "").replace("AWAY_", "")}'
        df[col_name] = df.groupby("HOME_TEAM")[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    # eFG/TS rolling features (explicit names)
    df["HOME_EFG_ROLL10"] = df.groupby("HOME_TEAM")["HOME_EFG"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    df["HOME_TS_ROLL10"] = df.groupby("HOME_TEAM")["HOME_TS"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )

    # Away team rolling features
    for stat in away_stats:
        col_name = f'AWAY_ROLL10_{stat.replace("AWAY_", "").replace("HOME_", "")}'
        df[col_name] = df.groupby("AWAY_TEAM")[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    df["AWAY_EFG_ROLL10"] = df.groupby("AWAY_TEAM")["AWAY_EFG"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    df["AWAY_TS_ROLL10"] = df.groupby("AWAY_TEAM")["AWAY_TS"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )

    # Rolling window variations (5 and 15 games) for PTS
    for size, label in [(5, "ROLL5"), (15, "ROLL15")]:
        df[f"HOME_{label}_PTS"] = df.groupby("HOME_TEAM")["HOME_PTS"].transform(
            lambda x: x.shift(1).rolling(size, min_periods=2).mean()
        )
        df[f"AWAY_{label}_PTS"] = df.groupby("AWAY_TEAM")["AWAY_PTS"].transform(
            lambda x: x.shift(1).rolling(size, min_periods=2).mean()
        )

    df["HOME_FORM_TREND"] = df["HOME_ROLL5_PTS"] - df["HOME_ROLL15_PTS"]
    df["AWAY_FORM_TREND"] = df["AWAY_ROLL5_PTS"] - df["AWAY_ROLL15_PTS"]

    # Derived combined features
    df["COMBINED_PTS_ROLL10"] = df["HOME_ROLL10_PTS"] + df["AWAY_ROLL10_PTS"]
    df["COMBINED_FG3A_ROLL10"] = df["HOME_ROLL10_FG3A"] + df["AWAY_ROLL10_FG3A"]
    df["COMBINED_FTA_ROLL10"] = df["HOME_ROLL10_FTA"] + df["AWAY_ROLL10_FTA"]
    df["COMBINED_POSS_ROLL10"] = (
        df["HOME_ROLL10_POSS_EST"] + df["AWAY_ROLL10_POSS_EST"]
    )
    df["TOV_DIFF_ROLL10"] = df["HOME_ROLL10_TOV"] - df["AWAY_ROLL10_TOV"]
    df["OREB_DIFF_ROLL10"] = df["HOME_ROLL10_OREB"] - df["AWAY_ROLL10_OREB"]

    return df
