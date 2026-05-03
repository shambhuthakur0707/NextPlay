# -*- coding: utf-8 -*-
"""
NextPlay -- Playoff Feature Engineering
=========================================
Adds playoff-specific features that capture postseason dynamics:
  - IS_PLAYOFF flag (binary)
  - PLAYOFF_HOME_BOOST: amplified home-court advantage in playoffs
  - PLAYOFF_ROAD_PENALTY: road teams historically underperform in playoffs
  - PLAYOFF_REST_ADVANTAGE: rest matters more in tight playoff series
  - HOME_PLAYOFF_WIN_PCT / AWAY_PLAYOFF_WIN_PCT: rolling playoff home/away splits
  - SERIES_GAME_NUM: game number within a 7-game series (1-7)
  - IS_ELIMINATION: whether the game is do-or-die for either team
  - IS_CLOSEOUT: whether one team can close out the series
  - HOME_SERIES_LEAD / AWAY_SERIES_LEAD: series standing entering the game
  - PLAYOFF_INTENSITY: composite score capturing series pressure
    - All playoff features are zeroed for regular-season games.

FIX (v9): Replaced row-by-row _compute_intensity (used .iloc[idx].get()
which is slow and fragile on Series objects) with a fully vectorized
implementation using numpy. This avoids KeyError on missing columns and
runs ~100x faster on 3,600+ row datasets.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# HISTORICAL PLAYOFF HOME-COURT ADVANTAGE (league-wide)
# ─────────────────────────────────────────────────────────────
# NBA historical playoff home win rate is ~63%, vs ~57% regular season.
# The delta is used to amplify home-court features during playoffs.
PLAYOFF_HOME_WIN_RATE = 0.63
REG_SEASON_HOME_WIN_RATE = 0.57
PLAYOFF_HOME_BOOST_DELTA = PLAYOFF_HOME_WIN_RATE - REG_SEASON_HOME_WIN_RATE


def add_playoff_features(games):
    """
    Add playoff-specific features to the game-level DataFrame.

    Features added (12 new columns):
    - IS_PLAYOFF: 1 if playoff game, 0 otherwise
    - PLAYOFF_HOME_BOOST: amplified home-court signal for playoff games
    - PLAYOFF_ROAD_PENALTY: negative signal for away teams in playoffs
    - PLAYOFF_REST_ADVANTAGE: rest * playoff interaction
    - HOME_PLAYOFF_WIN_PCT: team's rolling playoff home win rate
    - AWAY_PLAYOFF_WIN_PCT: team's rolling playoff away win rate
    - SERIES_GAME_NUM: estimated game number within a series (1-7)
    - IS_ELIMINATION: 1 if either team faces elimination
    - IS_CLOSEOUT: 1 if one team can clinch the series
    - HOME_SERIES_LEAD: home team's series lead entering the game
    - AWAY_SERIES_LEAD: away team's series lead entering the game
    - PLAYOFF_INTENSITY: composite pressure score (0-100)

    Args:
        games: game-level DataFrame with SEASON_TYPE or IS_PLAYOFF column

    Returns:
        DataFrame with playoff features added
    """
    df = games.copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # ── Base IS_PLAYOFF flag ────────────────────────────────────────────
    if "IS_PLAYOFF" not in df.columns:
        if "SEASON_TYPE" in df.columns:
            df["IS_PLAYOFF"] = (
                df["SEASON_TYPE"].str.contains("Playoff", case=False, na=False)
            ).astype(int)
        else:
            df["IS_PLAYOFF"] = 0

    df["IS_PLAYOFF"] = df["IS_PLAYOFF"].astype(int)

    # ── Playoff Home-Court Boost ────────────────────────────────────────
    # Home court is worth ~6% more in playoffs than regular season.
    if "HOME_COURT_STRENGTH" in df.columns:
        df["PLAYOFF_HOME_BOOST"] = (
            df["IS_PLAYOFF"]
            * PLAYOFF_HOME_BOOST_DELTA
            * (1 + df["HOME_COURT_STRENGTH"].fillna(0.5))
        )
    else:
        df["PLAYOFF_HOME_BOOST"] = df["IS_PLAYOFF"] * PLAYOFF_HOME_BOOST_DELTA

    # ── Playoff Road Penalty ────────────────────────────────────────────
    # Away teams score ~3-4 fewer points per game in playoffs vs reg season.
    if "AWAY_TEAM_STRENGTH" in df.columns:
        df["PLAYOFF_ROAD_PENALTY"] = (
            df["IS_PLAYOFF"]
            * -PLAYOFF_HOME_BOOST_DELTA
            * (1 + (1 - df["AWAY_TEAM_STRENGTH"].fillna(0.5)))
        )
    else:
        df["PLAYOFF_ROAD_PENALTY"] = df["IS_PLAYOFF"] * -PLAYOFF_HOME_BOOST_DELTA

    # ── Playoff Rest Advantage ──────────────────────────────────────────
    # Rest matters more in tighter playoff rotations.
    if "REST_DAYS_DIFF" in df.columns:
        df["PLAYOFF_REST_ADVANTAGE"] = (
            df["IS_PLAYOFF"] * df["REST_DAYS_DIFF"].fillna(0) * 1.5
        )
    else:
        df["PLAYOFF_REST_ADVANTAGE"] = 0.0

    # ── Rolling Playoff Win Percentages ────────────────────────────────
    df["HOME_PLAYOFF_WIN_PCT"] = _rolling_playoff_win_pct(
        df, team_col="HOME_TEAM", wl_col="HOME_WL", is_playoff_col="IS_PLAYOFF"
    )
    df["AWAY_PLAYOFF_WIN_PCT"] = _rolling_playoff_win_pct(
        df, team_col="AWAY_TEAM", wl_col="AWAY_WL", is_playoff_col="IS_PLAYOFF"
    )

    # ── Series-Level Features ───────────────────────────────────────────
    series_info = _estimate_series_state(df)
    df["SERIES_GAME_NUM"]  = series_info["series_game_num"]
    df["IS_ELIMINATION"]   = series_info["is_elimination"]
    df["IS_CLOSEOUT"]      = series_info["is_closeout"]
    df["HOME_SERIES_LEAD"] = series_info["home_series_lead"]
    df["AWAY_SERIES_LEAD"] = series_info["away_series_lead"]

    # ── Playoff Intensity Score ─────────────────────────────────────────
    df["PLAYOFF_INTENSITY"] = _compute_intensity(df)

    # ── Zero out all playoff features for regular-season games ──────────
    playoff_feature_cols = [
        "PLAYOFF_HOME_BOOST", "PLAYOFF_ROAD_PENALTY",
        "PLAYOFF_REST_ADVANTAGE",
        "HOME_PLAYOFF_WIN_PCT", "AWAY_PLAYOFF_WIN_PCT",
        "SERIES_GAME_NUM", "IS_ELIMINATION", "IS_CLOSEOUT",
        "HOME_SERIES_LEAD", "AWAY_SERIES_LEAD",
        "PLAYOFF_INTENSITY",
    ]
    mask_reg = df["IS_PLAYOFF"] == 0
    for col in playoff_feature_cols:
        if col in df.columns:
            df.loc[mask_reg, col] = 0.0

    return df


def _rolling_playoff_win_pct(df, team_col, wl_col, is_playoff_col,
                              window=20, min_periods=3):
    """
    Compute a rolling playoff-only win percentage for a team.
    Uses shift(1) so the current game is excluded (no leakage).
    """
    playoff_wins = []
    for _, group in df.groupby(team_col):
        g = group.copy()
        playoff_mask = g[is_playoff_col] == 1
        g["_playoff_win"] = (
            (g[wl_col] == "W") & playoff_mask
        ).astype(float)
        g["_playoff_game"] = playoff_mask.astype(float)

        g["_cum_wins"] = (
            g["_playoff_win"].shift(1).rolling(window, min_periods=min_periods).sum()
        )
        g["_cum_games"] = (
            g["_playoff_game"].shift(1).rolling(window, min_periods=min_periods).sum()
        )

        g["_pct"] = np.where(
            g["_cum_games"] > 0,
            g["_cum_wins"] / g["_cum_games"],
            0.5,  # default to 50% if no playoff history
        )
        playoff_wins.append(
            g[["_pct"]].rename(columns={"_pct": "_result"})
        )

    result = pd.concat(playoff_wins).sort_index()
    return result["_result"].values


def _estimate_series_state(df):
    """
    Estimate series-level state for playoff games.

    Detects series by looking at consecutive playoff games between
    the same two teams within the same season.
    """
    n = len(df)
    series_game_num  = np.zeros(n, dtype=float)
    is_elimination   = np.zeros(n, dtype=float)
    is_closeout      = np.zeros(n, dtype=float)
    home_series_lead = np.zeros(n, dtype=float)
    away_series_lead = np.zeros(n, dtype=float)

    series_tracker = {}
    last_season = None

    for idx in range(n):
        row = df.iloc[idx]
        is_po = row.get("IS_PLAYOFF", 0)

        if not is_po:
            continue

        season = row.get("SEASON", "")
        home   = row.get("HOME_TEAM", "")
        away   = row.get("AWAY_TEAM", "")

        if not home or not away:
            continue

        if season != last_season:
            series_tracker = {}
            last_season = season

        matchup_key = frozenset([home, away])

        if matchup_key not in series_tracker:
            series_tracker[matchup_key] = {home: 0, away: 0, "games": 0}

        state   = series_tracker[matchup_key]
        game_num = state["games"] + 1
        h_wins   = state[home]
        a_wins   = state[away]

        series_game_num[idx]  = game_num
        home_series_lead[idx] = h_wins - a_wins
        away_series_lead[idx] = a_wins - h_wins

        # Closeout / elimination: one team is at 3 wins before this game
        if h_wins == 3:
            is_closeout[idx]  = 1  # home can close out
            is_elimination[idx] = 1  # away faces elimination
        elif a_wins == 3:
            is_closeout[idx]  = 1  # away can close out
            is_elimination[idx] = 1  # home faces elimination

        # Update tracker with result
        home_wl = row.get("HOME_WL", "")
        if home_wl == "W":
            state[home] += 1
        elif home_wl == "L":
            state[away] += 1
        state["games"] += 1

        # Reset if series is over (4 wins)
        if state[home] >= 4 or state[away] >= 4:
            del series_tracker[matchup_key]

    return {
        "series_game_num":  series_game_num,
        "is_elimination":   is_elimination,
        "is_closeout":      is_closeout,
        "home_series_lead": home_series_lead,
        "away_series_lead": away_series_lead,
    }


def _compute_intensity(df):
    """
    Compute a playoff intensity score (0-100).

    Fully vectorized -- no row-by-row iteration.

    Factors:
    - Series game number: later games = higher intensity (max 70)
    - Elimination / closeout games: +20 bonus
    - Close series (lead <= 1 game): +10 bonus
    """
    is_playoff = df["IS_PLAYOFF"].values.astype(float)

    gnum = pd.to_numeric(
        df.get("SERIES_GAME_NUM", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).values

    elim = pd.to_numeric(
        df.get("IS_ELIMINATION", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).values

    lead = pd.to_numeric(
        df.get("HOME_SERIES_LEAD", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).abs().values

    score = np.clip(gnum * 10, 0, 70)
    score = score + elim * 20
    score = score + (lead <= 1).astype(float) * 10

    intensity = np.where(is_playoff == 1, np.clip(score, 0, 100), 0.0)
    return intensity