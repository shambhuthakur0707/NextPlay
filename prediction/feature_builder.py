# -*- coding: utf-8 -*-
"""
NextPlay -- Live Feature Builder
================================
Builds a single matchup feature row from historical game data.
The base row comes from the latest available history, while the
rolling/state features are refreshed from unified team history.
"""

from __future__ import annotations

import os

import pandas as pd

from config import (
    FEATURE_COLS_FINAL,
    ROLLING_WINDOW,
    GAMELOGS_ALL_PATH,
    GAMELOGS_RAW_PATH,
)
from features.rolling import build_team_rolling_snapshot


def _normalize_history_frame(model_df=None, raw_gamelogs=None):
    """Return a game-level historical frame sorted by game date."""
    history = None

    if raw_gamelogs is not None:
        history = raw_gamelogs.copy()
        if "GAME_DATE" in history.columns:
            history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    elif os.path.exists(GAMELOGS_ALL_PATH):
        history = pd.read_csv(GAMELOGS_ALL_PATH)
        history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    elif os.path.exists(GAMELOGS_RAW_PATH):
        history = pd.read_csv(GAMELOGS_RAW_PATH)
        history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])

    if history is not None and "TEAM_ABBR" in history.columns and "HOME_TEAM" not in history.columns:
        from ingestion.gamelogs import build_game_level_dataset

        history = build_game_level_dataset(history)

    if history is None:
        return None

    if "GAME_DATE" in history.columns:
        history = history.sort_values("GAME_DATE").reset_index(drop=True)

    return history


def _normalize_base_frame(model_df=None):
    """Return the engineered game-level frame used for base features."""
    if model_df is None:
        return None

    base = model_df.copy()
    if "GAME_DATE" in base.columns:
        base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
        base = base.sort_values("GAME_DATE").reset_index(drop=True)
    return base


def _latest_team_row(history_df, team, as_of_date=None):
    """Get the most recent historical row for a team."""
    if history_df is None or len(history_df) == 0:
        return None

    frame = history_df.copy()
    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        frame = frame[frame["GAME_DATE"] < as_of_date]

    candidates = frame[
        (frame["HOME_TEAM"] == team) | (frame["AWAY_TEAM"] == team)
    ].copy()
    if len(candidates) == 0:
        return None

    return candidates.sort_values(["GAME_DATE", "GAME_ID"]).iloc[-1]


def _apply_snapshot(prefix, row, snapshot):
    """Write snapshot fields back into the feature row using team prefixes."""
    if snapshot is None:
        return

    mapping = {
        f"{prefix}_ROLL10_PTS": snapshot.get("ROLL10_PTS"),
        f"{prefix}_ROLL10_FG_PCT": snapshot.get("ROLL10_FG_PCT"),
        f"{prefix}_ROLL10_FG3_PCT": snapshot.get("ROLL10_FG3_PCT"),
        f"{prefix}_ROLL10_FT_PCT": snapshot.get("ROLL10_FT_PCT"),
        f"{prefix}_ROLL10_FG3A": snapshot.get("ROLL10_FG3A"),
        f"{prefix}_ROLL10_FTA": snapshot.get("ROLL10_FTA"),
        f"{prefix}_ROLL10_AST": snapshot.get("ROLL10_AST"),
        f"{prefix}_ROLL10_TOV": snapshot.get("ROLL10_TOV"),
        f"{prefix}_ROLL10_OREB": snapshot.get("ROLL10_OREB"),
        f"{prefix}_ROLL10_REB": snapshot.get("ROLL10_REB"),
        f"{prefix}_ROLL10_PF": snapshot.get("ROLL10_PF"),
        f"{prefix}_ROLL10_TOTAL_PTS": snapshot.get("ROLL10_TOTAL_PTS"),
        f"{prefix}_ROLL10_POSS_EST": snapshot.get("ROLL10_POSS_EST"),
        f"{prefix}_DEF_ROLL10": snapshot.get("DEF_ROLL10"),
        f"{prefix}_EFG_ROLL10": snapshot.get("EFG_ROLL10"),
        f"{prefix}_TS_ROLL10": snapshot.get("TS_ROLL10"),
        f"{prefix}_ROLL5_PTS": snapshot.get("ROLL5_PTS"),
        f"{prefix}_ROLL15_PTS": snapshot.get("ROLL15_PTS"),
        f"{prefix}_FORM_TREND": snapshot.get("FORM_TREND"),
        f"{prefix}_EFG": snapshot.get("EFG"),
        f"{prefix}_TS": snapshot.get("TS"),
        f"{prefix}_POSS_EST": snapshot.get("POSS_EST"),
        f"{prefix}_PTS": snapshot.get("PTS"),
    }

    row.update(mapping)


def _apply_derived_features(row):
    """Recompute combined features that depend on the refreshed snapshots."""
    derived_pairs = [
        ("COMBINED_PTS_ROLL10", row.get("HOME_ROLL10_PTS"), row.get("AWAY_ROLL10_PTS"), "+"),
        ("COMBINED_FG3A_ROLL10", row.get("HOME_ROLL10_FG3A"), row.get("AWAY_ROLL10_FG3A"), "+"),
        ("COMBINED_FTA_ROLL10", row.get("HOME_ROLL10_FTA"), row.get("AWAY_ROLL10_FTA"), "+"),
        ("COMBINED_POSS_ROLL10", row.get("HOME_ROLL10_POSS_EST"), row.get("AWAY_ROLL10_POSS_EST"), "+"),
        ("COMBINED_DEF_ROLL10", row.get("HOME_DEF_ROLL10"), row.get("AWAY_DEF_ROLL10"), "+"),
        ("TOV_DIFF_ROLL10", row.get("HOME_ROLL10_TOV"), row.get("AWAY_ROLL10_TOV"), "-"),
        ("OREB_DIFF_ROLL10", row.get("HOME_ROLL10_OREB"), row.get("AWAY_ROLL10_OREB"), "-"),
    ]

    for key, left, right, op in derived_pairs:
        if key in row and pd.notna(left) and pd.notna(right):
            row[key] = left + right if op == "+" else left - right

    if "HOME_FORM_TREND" in row and pd.notna(row.get("HOME_ROLL5_PTS")) and pd.notna(row.get("HOME_ROLL15_PTS")):
        row["HOME_FORM_TREND"] = row["HOME_ROLL5_PTS"] - row["HOME_ROLL15_PTS"]
    if "AWAY_FORM_TREND" in row and pd.notna(row.get("AWAY_ROLL5_PTS")) and pd.notna(row.get("AWAY_ROLL15_PTS")):
        row["AWAY_FORM_TREND"] = row["AWAY_ROLL5_PTS"] - row["AWAY_ROLL15_PTS"]

    if "HOME_MOMENTUM" in row and pd.notna(row.get("HOME_EWM_PTS")) and pd.notna(row.get("HOME_ROLL10_PTS")):
        row["HOME_MOMENTUM"] = row["HOME_EWM_PTS"] - row["HOME_ROLL10_PTS"]
    if "AWAY_MOMENTUM" in row and pd.notna(row.get("AWAY_EWM_PTS")) and pd.notna(row.get("AWAY_ROLL10_PTS")):
        row["AWAY_MOMENTUM"] = row["AWAY_EWM_PTS"] - row["AWAY_ROLL10_PTS"]
    if "COMBINED_MOMENTUM" in row and pd.notna(row.get("HOME_MOMENTUM")) and pd.notna(row.get("AWAY_MOMENTUM")):
        row["COMBINED_MOMENTUM"] = row["HOME_MOMENTUM"] + row["AWAY_MOMENTUM"]

    if "EXPECTED_TOTAL" in row and pd.notna(row.get("HOME_ROLL10_PTS")) and pd.notna(row.get("AWAY_ROLL10_PTS")):
        if pd.notna(row.get("HOME_DEF_ROLL10")) and pd.notna(row.get("AWAY_DEF_ROLL10")):
            row["EXPECTED_TOTAL"] = (
                row["HOME_ROLL10_PTS"]
                + row["AWAY_ROLL10_PTS"]
                + row["HOME_DEF_ROLL10"]
                + row["AWAY_DEF_ROLL10"]
            ) / 2

    if "EWM_EXPECTED_TOTAL" in row:
        home_ewm = row.get("HOME_EWM_PTS")
        away_ewm = row.get("AWAY_EWM_PTS")
        home_def = row.get("HOME_EWM_DEF")
        away_def = row.get("AWAY_EWM_DEF")
        if pd.notna(home_ewm) and pd.notna(away_ewm) and pd.notna(home_def) and pd.notna(away_def):
            row["EWM_EXPECTED_TOTAL"] = (home_ewm + away_ewm + home_def + away_def) / 2


def _apply_playoff_features(row, is_playoff=False):
    """Populate playoff features for live inference.

    When ``is_playoff`` is True the function computes playoff-specific
    signals from already-populated context features.  When False every
    playoff column is zeroed out so the model sees a clean regular-season
    signal.
    """
    from config import PLAYOFF_FEATURES

    if not is_playoff:
        for col in PLAYOFF_FEATURES:
            row[col] = 0.0
        return

    row["IS_PLAYOFF"] = 1

    # Home-court boost (amplified in playoffs)
    home_strength = row.get("HOME_COURT_STRENGTH", 0.5)
    delta = 0.06  # historical playoff vs reg-season home-win-rate gap
    row["PLAYOFF_HOME_BOOST"] = delta * (1 + (home_strength if pd.notna(home_strength) else 0.5))

    # Road penalty
    away_strength = row.get("AWAY_TEAM_STRENGTH", 0.5)
    row["PLAYOFF_ROAD_PENALTY"] = -delta * (1 + (1 - (away_strength if pd.notna(away_strength) else 0.5)))

    # Rest advantage interaction
    rest_diff = row.get("REST_DAYS_DIFF", 0)
    row["PLAYOFF_REST_ADVANTAGE"] = (rest_diff if pd.notna(rest_diff) else 0) * 1.5

    # Default playoff win pcts (use historical base row values if present)
    for col in ["HOME_PLAYOFF_WIN_PCT", "AWAY_PLAYOFF_WIN_PCT"]:
        if col not in row or pd.isna(row.get(col)):
            row[col] = 0.5

    # Series-state defaults for live prediction (caller can override)
    for col in ["SERIES_GAME_NUM", "IS_ELIMINATION", "IS_CLOSEOUT",
                "HOME_SERIES_LEAD", "AWAY_SERIES_LEAD"]:
        if col not in row or pd.isna(row.get(col)):
            row[col] = 0.0

    # Intensity (will be low if series state isn't provided)
    gnum = row.get("SERIES_GAME_NUM", 0) or 0
    elim = row.get("IS_ELIMINATION", 0) or 0
    lead = abs(row.get("HOME_SERIES_LEAD", 0) or 0)
    intensity = min(gnum * 10, 70) + (20 if elim else 0) + (10 if lead <= 1 else 0)
    row["PLAYOFF_INTENSITY"] = min(intensity, 100)


def build_game_features(home_team, away_team, model_df=None, raw_gamelogs=None,
                        game_date=None, feature_cols=None, window=ROLLING_WINDOW,
                        is_playoff=None, series_game_num=None,
                        home_series_wins=None, away_series_wins=None):
    """Build a single matchup feature row for live inference.

    New playoff args (optional):
        is_playoff: force playoff mode (auto-detected from date if None)
        series_game_num: game number within the series (1-7)
        home_series_wins: home team's series wins entering this game
        away_series_wins: away team's series wins entering this game
    """
    from config import PLAYOFF_SEASON_START_MONTH, PLAYOFF_SEASON_END_MONTH

    base_history_df = _normalize_base_frame(model_df=model_df)
    rolling_history_df = _normalize_history_frame(model_df=model_df, raw_gamelogs=raw_gamelogs)

    if base_history_df is None or len(base_history_df) == 0:
        return None
    if rolling_history_df is None or len(rolling_history_df) == 0:
        rolling_history_df = base_history_df

    if game_date is None:
        game_date = base_history_df["GAME_DATE"].max() + pd.Timedelta(days=1)
    game_date = pd.Timestamp(game_date)

    base_cols = feature_cols or [c for c in FEATURE_COLS_FINAL if c in base_history_df.columns]
    home_row = _latest_team_row(base_history_df, home_team, as_of_date=game_date)
    away_row = _latest_team_row(base_history_df, away_team, as_of_date=game_date)
    if home_row is None or away_row is None:
        return None

    feature_row = {}
    for col in base_cols:
        if col in home_row.index:
            feature_row[col] = home_row[col]
        elif col in away_row.index:
            feature_row[col] = away_row[col]
        else:
            feature_row[col] = 0.0

    feature_row["GAME_DATE"] = game_date
    feature_row["HOME_TEAM"] = home_team
    feature_row["AWAY_TEAM"] = away_team

    # Refresh rolling snapshots from unified team history.
    home_snap = build_team_rolling_snapshot(rolling_history_df, home_team, as_of_date=game_date, window=window)
    away_snap = build_team_rolling_snapshot(rolling_history_df, away_team, as_of_date=game_date, window=window)
    _apply_snapshot("HOME", feature_row, home_snap)
    _apply_snapshot("AWAY", feature_row, away_snap)
    _apply_derived_features(feature_row)

    # ── Playoff features ────────────────────────────────────
    if is_playoff is None:
        month = game_date.month
        if PLAYOFF_SEASON_START_MONTH <= PLAYOFF_SEASON_END_MONTH:
            is_playoff = PLAYOFF_SEASON_START_MONTH <= month <= PLAYOFF_SEASON_END_MONTH
        else:
            is_playoff = month >= PLAYOFF_SEASON_START_MONTH or month <= PLAYOFF_SEASON_END_MONTH

    _apply_playoff_features(feature_row, is_playoff=is_playoff)

    # Override series-state if caller provided explicit values
    if series_game_num is not None:
        feature_row["SERIES_GAME_NUM"] = series_game_num
    if home_series_wins is not None and away_series_wins is not None:
        feature_row["HOME_SERIES_LEAD"] = home_series_wins - away_series_wins
        feature_row["AWAY_SERIES_LEAD"] = away_series_wins - home_series_wins
        feature_row["IS_CLOSEOUT"] = 1.0 if max(home_series_wins, away_series_wins) == 3 else 0.0
        feature_row["IS_ELIMINATION"] = feature_row["IS_CLOSEOUT"]
        # Recompute intensity with updated series state
        gnum = feature_row.get("SERIES_GAME_NUM", 0) or 0
        elim = feature_row.get("IS_ELIMINATION", 0) or 0
        lead = abs(feature_row.get("HOME_SERIES_LEAD", 0) or 0)
        feature_row["PLAYOFF_INTENSITY"] = min(
            min(gnum * 10, 70) + (20 if elim else 0) + (10 if lead <= 1 else 0), 100
        )

    ordered_cols = [col for col in base_cols if col in feature_row]
    feature_frame = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in ordered_cols}])
    return feature_frame
