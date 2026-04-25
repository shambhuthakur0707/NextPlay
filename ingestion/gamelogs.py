# -*- coding: utf-8 -*-
"""
NextPlay -- Game Log Ingestion
==============================
Pulls team-level game logs from the NBA API and builds
a game-level dataset (one row per game with home + away).
"""

import time
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams as nba_teams_static

from config import API_DELAY


def pull_team_gamelogs(season="2024-25"):
    """
    Pull game logs for all 30 NBA teams for a given season.

    Args:
        season: NBA season string e.g. '2024-25'

    Returns:
        DataFrame with all team game logs for the season
    """
    all_teams = nba_teams_static.get_teams()
    all_logs = []

    print(f"Pulling game logs for {len(all_teams)} teams -- Season {season}...")

    for team in all_teams:
        try:
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team["id"],
                season_nullable=season,
                season_type_nullable="Regular Season",
            ).get_data_frames()[0]

            logs["TEAM_ABBR"] = team["abbreviation"]
            logs["SEASON"] = season
            all_logs.append(logs)

            print(f"  [OK] {team['abbreviation']} -- {len(logs)} games")

        except Exception as e:
            print(f"  [FAIL] {team['abbreviation']} -- failed: {e}")

        time.sleep(API_DELAY)

    df = pd.concat(all_logs, ignore_index=True)
    return df


def pull_multi_season(seasons):
    """
    Pull game logs for multiple seasons and combine.

    Args:
        seasons: list of season strings e.g. ['2023-24', '2024-25']

    Returns:
        Combined DataFrame sorted by GAME_DATE
    """
    all_dfs = []

    for i, season in enumerate(seasons):
        df = pull_team_gamelogs(season)
        all_dfs.append(df)
        print(f"\n[OK] {season} done -- {len(df)} rows\n")

        if i < len(seasons) - 1:
            time.sleep(5)  # pause between seasons

    gamelogs = pd.concat(all_dfs, ignore_index=True)
    gamelogs["GAME_DATE"] = pd.to_datetime(gamelogs["GAME_DATE"])
    gamelogs = gamelogs.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"{'=' * 50}")
    print(f"[OK] Total rows: {len(gamelogs)}")
    print(f"   Seasons: {gamelogs['SEASON'].unique()}")
    print(f"   Date range: {gamelogs['GAME_DATE'].min().date()} -> "
          f"{gamelogs['GAME_DATE'].max().date()}")

    return gamelogs


def build_game_level_dataset(gamelogs):
    """
    Convert team-level game logs (2 rows per game) into
    game-level dataset (1 row per game with home + away columns).

    Home team = team whose MATCHUP contains 'vs.'
    Away team = team whose MATCHUP contains '@'
    """
    home = gamelogs[gamelogs["MATCHUP"].str.contains("vs.")].copy()
    away = gamelogs[gamelogs["MATCHUP"].str.contains("@")].copy()

    rename_home = {
        "TEAM_ABBR": "HOME_TEAM", "PTS": "HOME_PTS",
        "FGM": "HOME_FGM", "FGA": "HOME_FGA", "FG_PCT": "HOME_FG_PCT",
        "FG3M": "HOME_FG3M", "FG3A": "HOME_FG3A", "FG3_PCT": "HOME_FG3_PCT",
        "FTM": "HOME_FTM", "FTA": "HOME_FTA", "FT_PCT": "HOME_FT_PCT",
        "AST": "HOME_AST", "TOV": "HOME_TOV",
        "OREB": "HOME_OREB", "DREB": "HOME_DREB", "REB": "HOME_REB",
        "STL": "HOME_STL", "BLK": "HOME_BLK", "PF": "HOME_PF",
        "PLUS_MINUS": "HOME_PLUS_MINUS", "WL": "HOME_WL",
    }

    rename_away = {
        "TEAM_ABBR": "AWAY_TEAM", "PTS": "AWAY_PTS",
        "FGM": "AWAY_FGM", "FGA": "AWAY_FGA", "FG_PCT": "AWAY_FG_PCT",
        "FG3M": "AWAY_FG3M", "FG3A": "AWAY_FG3A", "FG3_PCT": "AWAY_FG3_PCT",
        "FTM": "AWAY_FTM", "FTA": "AWAY_FTA", "FT_PCT": "AWAY_FT_PCT",
        "AST": "AWAY_AST", "TOV": "AWAY_TOV",
        "OREB": "AWAY_OREB", "DREB": "AWAY_DREB", "REB": "AWAY_REB",
        "STL": "AWAY_STL", "BLK": "AWAY_BLK", "PF": "AWAY_PF",
        "PLUS_MINUS": "AWAY_PLUS_MINUS", "WL": "AWAY_WL",
    }

    home = home.rename(columns=rename_home)
    away = away.rename(columns=rename_away)

    home_cols = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM", "HOME_PTS", "HOME_FGM", "HOME_FGA",
        "HOME_FG_PCT", "HOME_FG3M", "HOME_FG3A", "HOME_FG3_PCT",
        "HOME_FTM", "HOME_FTA", "HOME_FT_PCT",
        "HOME_AST", "HOME_TOV", "HOME_OREB", "HOME_DREB",
        "HOME_REB", "HOME_STL", "HOME_BLK", "HOME_PF",
        "HOME_PLUS_MINUS", "HOME_WL",
    ]

    away_cols = [
        "GAME_ID",
        "AWAY_TEAM", "AWAY_PTS", "AWAY_FGM", "AWAY_FGA",
        "AWAY_FG_PCT", "AWAY_FG3M", "AWAY_FG3A", "AWAY_FG3_PCT",
        "AWAY_FTM", "AWAY_FTA", "AWAY_FT_PCT",
        "AWAY_AST", "AWAY_TOV", "AWAY_OREB", "AWAY_DREB",
        "AWAY_REB", "AWAY_STL", "AWAY_BLK", "AWAY_PF",
        "AWAY_PLUS_MINUS", "AWAY_WL",
    ]

    games = pd.merge(home[home_cols], away[away_cols], on="GAME_ID")
    games["TOTAL_PTS"] = games["HOME_PTS"] + games["AWAY_PTS"]
    games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
    games["PTS_MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"[OK] Game-level dataset built")
    print(f"   Total games : {len(games)}")
    print(f"   Date range  : {games['GAME_DATE'].min().date()} -> "
          f"{games['GAME_DATE'].max().date()}")

    return games
