# -*- coding: utf-8 -*-
"""
NextPlay -- Exponentially Weighted Moving Average Features
===========================================================
EWM gives more weight to recent games -- the model reacts
faster to current form compared to simple rolling averages.

FIX (v9): Rewrote to use a unified team-history table instead of
grouping by HOME_TEAM / AWAY_TEAM slot separately. The old approach
computed EWM only from a team's home games (for HOME_EWM_*) or only
from their away games (for AWAY_EWM_*), discarding ~half the data per
team per feature. The new approach mirrors the pattern in rolling.py:
build one row per team-game, compute shift(1).ewm(), then join back.
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

    All EWM features use shift(1) so the current game is never
    included in its own feature (no data leakage).
    """
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)

    # ── Build unified team history (one row per team-game) ──────────────
    # This ensures each team's EWM is computed across ALL their games,
    # not just their home games or just their away games.

    home = df[["GAME_DATE", "GAME_ID", "HOME_TEAM",
               "HOME_PTS", "AWAY_PTS",
               "HOME_FG_PCT", "HOME_FG3_PCT"]].copy()
    home = home.rename(columns={
        "HOME_TEAM":   "TEAM",
        "HOME_PTS":    "PTS",
        "AWAY_PTS":    "OPP_PTS",
        "HOME_FG_PCT": "FG_PCT",
        "HOME_FG3_PCT":"FG3_PCT",
    })

    away = df[["GAME_DATE", "GAME_ID", "AWAY_TEAM",
               "AWAY_PTS", "HOME_PTS",
               "AWAY_FG_PCT", "AWAY_FG3_PCT"]].copy()
    away = away.rename(columns={
        "AWAY_TEAM":   "TEAM",
        "AWAY_PTS":    "PTS",
        "HOME_PTS":    "OPP_PTS",
        "AWAY_FG_PCT": "FG_PCT",
        "AWAY_FG3_PCT":"FG3_PCT",
    })

    team_hist = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["TEAM", "GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )

    # ── Compute EWM per team, shifted to exclude current game ───────────
    for stat, new_col in [
        ("PTS",     "EWM_PTS"),
        ("FG_PCT",  "EWM_FG_PCT"),
        ("FG3_PCT", "EWM_FG3_PCT"),
        ("OPP_PTS", "EWM_DEF"),
    ]:
        team_hist[new_col] = team_hist.groupby("TEAM")[stat].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )

    # ── Keep one snapshot per (GAME_ID, TEAM) ───────────────────────────
    snap = (
        team_hist[["GAME_ID", "TEAM", "EWM_PTS", "EWM_FG_PCT",
                   "EWM_FG3_PCT", "EWM_DEF"]]
        .drop_duplicates(subset=["GAME_ID", "TEAM"])
    )

    # ── Join home snapshot back onto main df ────────────────────────────
    df = df.merge(
        snap.rename(columns={
            "TEAM":      "HOME_TEAM",
            "EWM_PTS":   "HOME_EWM_PTS",
            "EWM_FG_PCT":"HOME_EWM_FG_PCT",
            "EWM_FG3_PCT":"HOME_EWM_FG3_PCT",
            "EWM_DEF":   "HOME_EWM_DEF",
        }),
        on=["GAME_ID", "HOME_TEAM"],
        how="left",
    )

    # ── Join away snapshot back onto main df ────────────────────────────
    df = df.merge(
        snap.rename(columns={
            "TEAM":      "AWAY_TEAM",
            "EWM_PTS":   "AWAY_EWM_PTS",
            "EWM_FG_PCT":"AWAY_EWM_FG_PCT",
            "EWM_FG3_PCT":"AWAY_EWM_FG3_PCT",
            "EWM_DEF":   "AWAY_EWM_DEF",
        }),
        on=["GAME_ID", "AWAY_TEAM"],
        how="left",
    )

    # ── Derived EWM features ─────────────────────────────────────────────
    # EWM expected total: average of both teams' offensive and defensive EWM
    df["EWM_EXPECTED_TOTAL"] = (
        df["HOME_EWM_PTS"]
        + df["AWAY_EWM_PTS"]
        + df["HOME_EWM_DEF"]
        + df["AWAY_EWM_DEF"]
    ) / 2

    # Momentum = EWM - rolling avg (positive = scoring MORE recently than average)
    # Requires HOME/AWAY_ROLL10_PTS to already exist (added in rolling.py, step 1).
    df["HOME_MOMENTUM"] = df["HOME_EWM_PTS"] - df["HOME_ROLL10_PTS"]
    df["AWAY_MOMENTUM"] = df["AWAY_EWM_PTS"] - df["AWAY_ROLL10_PTS"]
    df["COMBINED_MOMENTUM"] = df["HOME_MOMENTUM"] + df["AWAY_MOMENTUM"]

    return df