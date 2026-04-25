# -*- coding: utf-8 -*-
"""
NextPlay -- ELO Rating Features
=================================
Adds pre-game ELO ratings to game-level data as model features.
For historical rows, ratings are updated game-by-game in chronological order.
For rows without game results, the module falls back to the static ratings table.
"""
import pandas as pd

from config import ELO_RATINGS_PATH


def _load_elo_lookup(path=ELO_RATINGS_PATH):
    """Load ELO ratings CSV and return a team -> elo dict."""
    elo_df = pd.read_csv(path)
    return dict(zip(elo_df["team"], elo_df["elo"]))


def _sort_games_for_elo(games):
    """Sort games chronologically while preserving a stable tie-break order."""
    sort_cols = [c for c in ["SEASON", "GAME_DATE", "GAME_ID"] if c in games.columns]
    if not sort_cols:
        return games.copy()

    return games.sort_values(sort_cols).reset_index(drop=True)


def _elo_expected_score(home_elo, away_elo, home_advantage=100):
    """
    Classic ELO expected score formula.

    The home team gets a small ELO bonus (default 100) to account
    for home-court advantage built into the rating.

    Returns a value between 0 and 1 representing the home team's
    expected win probability.
    """
    diff = (home_elo + home_advantage) - away_elo
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _get_actual_home_result(row):
    """Return the actual home-game result when results are available."""
    if "HOME_WIN" in row.index and pd.notna(row["HOME_WIN"]):
        return float(row["HOME_WIN"])

    if {
        "HOME_PTS",
        "AWAY_PTS",
    }.issubset(row.index) and pd.notna(row["HOME_PTS"]) and pd.notna(row["AWAY_PTS"]):
        return float(row["HOME_PTS"] > row["AWAY_PTS"])

    return None


def _add_rolling_elo_features(
    games,
    k_factor=20.0,
    home_advantage=100.0,
    league_avg_elo=1500.0,
):
    """Compute rolling ELO features using only prior game information."""
    df = _sort_games_for_elo(games)

    current_ratings = {}
    home_elos = []
    away_elos = []
    elo_diffs = []
    elo_expected = []

    for _, row in df.iterrows():
        home_team = row["HOME_TEAM"]
        away_team = row["AWAY_TEAM"]

        home_elo = current_ratings.get(home_team, league_avg_elo)
        away_elo = current_ratings.get(away_team, league_avg_elo)
        expected_home = _elo_expected_score(
            home_elo,
            away_elo,
            home_advantage=home_advantage,
        )

        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)
        elo_expected.append(expected_home)

        actual_home = _get_actual_home_result(row)
        if actual_home is None:
            continue

        rating_delta = k_factor * (actual_home - expected_home)
        current_ratings[home_team] = home_elo + rating_delta
        current_ratings[away_team] = away_elo - rating_delta

    df["HOME_ELO"] = home_elos
    df["AWAY_ELO"] = away_elos
    df["ELO_DIFF"] = elo_diffs
    df["ELO_EXPECTED"] = elo_expected

    return df


def _add_static_elo_features(games, elo_path=ELO_RATINGS_PATH, league_avg_elo=1500.0):
    """Fallback for rows without outcomes; uses the current static ratings table."""
    df = games.copy()
    elo_lookup = _load_elo_lookup(elo_path)

    df["HOME_ELO"] = df["HOME_TEAM"].map(elo_lookup).fillna(league_avg_elo)
    df["AWAY_ELO"] = df["AWAY_TEAM"].map(elo_lookup).fillna(league_avg_elo)
    df["ELO_DIFF"] = df["HOME_ELO"] - df["AWAY_ELO"]
    df["ELO_EXPECTED"] = df.apply(
        lambda r: _elo_expected_score(r["HOME_ELO"], r["AWAY_ELO"]),
        axis=1,
    )

    return df


def add_elo_features(games, elo_path=ELO_RATINGS_PATH):
    """
    Add ELO ratings as model features.

    Features added:
    - HOME_ELO: raw ELO rating of home team
    - AWAY_ELO: raw ELO rating of away team
    - ELO_DIFF: home ELO minus away ELO
    - ELO_EXPECTED: expected home win probability (0-1) from ELO formula

    Args:
        games: game-level DataFrame with HOME_TEAM and AWAY_TEAM columns
        elo_path: path to elo_ratings.csv

    Returns:
        DataFrame with ELO features added
    """
    df = games.copy()

    if {"HOME_TEAM", "AWAY_TEAM"}.issubset(df.columns) and (
        "HOME_WIN" in df.columns or {"HOME_PTS", "AWAY_PTS"}.issubset(df.columns)
    ):
        return _add_rolling_elo_features(df)

    return _add_static_elo_features(df, elo_path=elo_path)
