# -*- coding: utf-8 -*-
"""
NextPlay -- Nightly Update Pipeline
=====================================
Run every morning to:
1. Pull last night's completed games
2. Compare actual scores vs predictions
3. Predict tonight's games
"""
import pandas as pd
from datetime import datetime, timedelta, timezone
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv3

from ingestion.odds import pull_closing_market_lines
from prediction.predict import predict_game


def get_last_night_games():
    """Pull completed games from last night."""
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Pulling games from {yesterday}...")

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=yesterday,
            date_to_nullable=yesterday,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]

        if len(finder) == 0:
            print(f"  No games found for {yesterday}")
            return None

        home = finder[finder["MATCHUP"].str.contains("vs.")].copy()
        away = finder[finder["MATCHUP"].str.contains("@")].copy()

        if len(home) == 0:
            return None

        home = home.rename(columns={
            "TEAM_ABBREVIATION": "HOME_TEAM", "PTS": "HOME_PTS"})
        away = away.rename(columns={
            "TEAM_ABBREVIATION": "AWAY_TEAM", "PTS": "AWAY_PTS"})

        games = pd.merge(
            home[["GAME_ID", "GAME_DATE", "HOME_TEAM", "HOME_PTS"]],
            away[["GAME_ID", "AWAY_TEAM", "AWAY_PTS"]],
            on="GAME_ID",
        )
        games["TOTAL_PTS"] = games["HOME_PTS"] + games["AWAY_PTS"]

        print(f"  [OK] {len(games)} games found")
        return games

    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return None


def get_tonight_schedule():
    """Pull tonight's scheduled games."""
    today = datetime.today().strftime("%Y-%m-%d")

    try:
        sb = scoreboardv3.ScoreboardV3(game_date=today, league_id="00")
        frames = sb.get_data_frames()
        if len(frames) < 3:
            return []

        game_header = frames[1].copy()
        team_lines = frames[2][["gameId", "teamId", "teamTricode"]].copy()

        upcoming = game_header.copy()
        upcoming["gameTimeUTC"] = pd.to_datetime(upcoming["gameTimeUTC"], utc=True)
        now_utc = pd.Timestamp.now(tz=timezone.utc)
        upcoming = upcoming[upcoming["gameTimeUTC"] > now_utc]

        if len(upcoming) == 0:
            return []

        schedule = []
        for _, row in upcoming.iterrows():
            gid = row["gameId"]
            game_teams = team_lines[team_lines["gameId"] == gid].reset_index(drop=True)
            if len(game_teams) < 2:
                continue

            home = game_teams.iloc[0]["teamTricode"]
            away = game_teams.iloc[1]["teamTricode"]
            schedule.append((home, away))

        return schedule

    except Exception as e:
        print(f"  [FAIL] Schedule pull failed: {e}")
        return []


def nightly_update(model_df, models, shot_df=None,
                   player_impact_df=None, feature_cols=None,
                   home_injuries=None, away_injuries=None):
    """
    Full nightly pipeline.

    Args:
        model_df: current feature dataset
        models: trained model dict
        home_injuries: dict of {team: [player_names]}
        away_injuries: dict of {team: [player_names]}
    """
    home_injuries = home_injuries or {}
    away_injuries = away_injuries or {}

    print("=" * 55)
    print(f"NIGHTLY UPDATE -- {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 55)

    # Step 1: Last night's results
    print("\n[PULL] STEP 1 -- Last night's results")
    last_night = get_last_night_games()

    if last_night is not None and len(last_night) > 0:
        print("\n  Results:")
        for _, g in last_night.iterrows():
            winner = (g["HOME_TEAM"] if g["HOME_PTS"] > g["AWAY_PTS"]
                      else g["AWAY_TEAM"])
            print(f"  {g['HOME_TEAM']} {g['HOME_PTS']:.0f} - "
                  f"{g['AWAY_PTS']:.0f} {g['AWAY_TEAM']}  "
                  f"| Winner: {winner} "
                  f"| Total: {g['TOTAL_PTS']:.0f}")

    # Step 2: Tonight's predictions
    print(f"\n[PRED] STEP 2 -- Tonight's predictions")
    schedule = get_tonight_schedule()

    market_lines = None
    try:
        today = datetime.today().strftime("%Y-%m-%d")
        market_lines = pull_closing_market_lines(date_from=today, date_to=today)
    except Exception as exc:
        print(f"  [WARN] Could not load market lines: {exc}")

    if len(schedule) == 0:
        print("  No games scheduled tonight")
    else:
        print(f"  {len(schedule)} games tonight\n")
        for home, away in schedule:
            market_total_line = None
            if market_lines is not None and len(market_lines) > 0:
                match = market_lines[
                    (market_lines["HOME_TEAM"] == home) &
                    (market_lines["AWAY_TEAM"] == away)
                ]
                if len(match) > 0:
                    market_total_line = match.iloc[0].get("CLOSE_TOTAL")

            h_out = home_injuries.get(home, [])
            a_out = away_injuries.get(away, [])

            predict_game(
                home, away, model_df, models,
                shot_df=shot_df,
                player_impact_df=player_impact_df,
                feature_cols=feature_cols,
                home_out=h_out, away_out=a_out,
                market_total_line=market_total_line,
                verbose=True,
            )

    print(f"\n[OK] Nightly update complete")
