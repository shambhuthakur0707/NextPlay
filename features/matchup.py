# -*- coding: utf-8 -*-
"""
NextPlay -- Matchup-Specific Features
=====================================
Adds head-to-head historical context for each home/away pairing.
All stats are shifted so the current game is never included.
"""
import pandas as pd


def add_matchup_features(games, min_meetings=3):
    """
    Add matchup-specific historical features.

    Features added:
    - MATCHUP_AVG_TOTAL: average total points in prior meetings
    - MATCHUP_HOME_WIN: average home win rate in prior meetings
    - MATCHUP_MEETINGS: number of prior meetings

    Args:
        games: game-level DataFrame
        min_meetings: minimum prior meetings required to keep features

    Returns:
        DataFrame with matchup features added
    """
    df = games.copy()

    sort_cols = [c for c in ["GAME_DATE", "GAME_ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    matchup = df.groupby(["HOME_TEAM", "AWAY_TEAM"], sort=False)

    df["MATCHUP_MEETINGS"] = matchup.cumcount()
    df["MATCHUP_AVG_TOTAL"] = matchup["TOTAL_PTS"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["MATCHUP_HOME_WIN"] = matchup["HOME_WIN"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    if min_meetings > 0:
        mask = df["MATCHUP_MEETINGS"] >= min_meetings
        df.loc[~mask, ["MATCHUP_AVG_TOTAL", "MATCHUP_HOME_WIN"]] = pd.NA

    # Fill missing values with neutral defaults to avoid row drops
    league_avg_total = df["TOTAL_PTS"].mean()
    df["MATCHUP_AVG_TOTAL"] = df["MATCHUP_AVG_TOTAL"].fillna(league_avg_total)
    df["MATCHUP_HOME_WIN"] = df["MATCHUP_HOME_WIN"].fillna(0.5)
    df["MATCHUP_MEETINGS"] = df["MATCHUP_MEETINGS"].fillna(0)

    return df
