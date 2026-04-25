# -*- coding: utf-8 -*-
"""
NextPlay -- Player Impact Ingestion
====================================
Computes true on/off impact for each player by comparing
team performance with and without each player in the lineup.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams as nba_teams_static

from config import API_DELAY


def build_player_impact(gamelogs, seasons):
    """
    Build true player on/off impact data for all teams.

    For each player with 10+ minutes avg:
    - Compare team scoring with player vs without
    - Calculate composite impact score

    Args:
        gamelogs: raw team-level gamelogs DataFrame
        seasons: list of season strings to process

    Returns:
        DataFrame with player impact metrics
    """
    print("Building true player on/off impact system...")
    print("This takes ~8 mins per season\n")

    all_teams = nba_teams_static.get_teams()
    all_impacts = []

    for season in seasons:
        season_games = gamelogs[gamelogs["SEASON"] == season].copy()

        # Normalize GAME_ID for matching
        season_games["GAME_ID_STR"] = (
            season_games["GAME_ID"].astype(str).str.lstrip("0")
        )

        print(f"\n{'=' * 50}")
        print(f"Season: {season}")
        print(f"{'=' * 50}")

        for team in all_teams:
            team_id = team["id"]
            team_abbr = team["abbreviation"]

            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id, season=season
                ).get_data_frames()[0]
            except Exception as e:
                print(f"  [FAIL] {team_abbr} roster failed: {e}")
                time.sleep(1)
                continue

            time.sleep(API_DELAY)

            # Team game log for this season
            team_log = season_games[
                season_games["TEAM_ABBR"] == team_abbr
            ][["GAME_ID_STR", "GAME_DATE", "PTS"]].copy()

            team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])
            team_log = team_log.sort_values("GAME_DATE")

            if len(team_log) == 0:
                continue

            players_tracked = 0

            for _, player in roster.iterrows():
                player_id = player["PLAYER_ID"]
                player_name = player["PLAYER"]

                try:
                    p_log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        season_type_all_star="Regular Season",
                    ).get_data_frames()[0]

                    time.sleep(0.5)

                    if len(p_log) == 0:
                        continue

                    # Filter 10+ min avg players
                    p_log["MIN"] = pd.to_numeric(p_log["MIN"], errors="coerce")
                    avg_min = p_log["MIN"].mean()
                    if avg_min < 10:
                        continue

                    # Normalize player log Game_ID
                    played_game_ids = set(
                        p_log["Game_ID"].astype(str).str.lstrip("0")
                    )

                    # Match against team log
                    team_log["PLAYED"] = team_log["GAME_ID_STR"].isin(
                        played_game_ids
                    )

                    with_games = team_log[team_log["PLAYED"]]
                    without_games = team_log[~team_log["PLAYED"]]

                    if len(with_games) < 5:
                        continue

                    pts_with = with_games["PTS"].mean()
                    pts_without = (
                        without_games["PTS"].mean()
                        if len(without_games) >= 3
                        else None
                    )

                    avg_pts = p_log["PTS"].mean()
                    avg_ast = p_log["AST"].mean()
                    avg_pm = p_log["PLUS_MINUS"].mean()

                    pts_impact = (
                        (pts_with - pts_without)
                        if pts_without is not None
                        else avg_pts * 0.75
                    )

                    composite_impact = (
                        pts_impact * 0.60
                        + avg_ast * 2.0 * 0.25
                        + avg_pm * 0.15
                    )

                    all_impacts.append({
                        "PLAYER_ID": player_id,
                        "PLAYER_NAME": player_name,
                        "TEAM_ABBR": team_abbr,
                        "SEASON": season,
                        "AVG_MIN": round(avg_min, 1),
                        "AVG_PTS": round(avg_pts, 1),
                        "AVG_AST": round(avg_ast, 1),
                        "AVG_PM": round(avg_pm, 1),
                        "GAMES_PLAYED": len(with_games),
                        "GAMES_MISSED": len(without_games),
                        "TEAM_PTS_WITH": round(pts_with, 1),
                        "TEAM_PTS_WITHOUT": (
                            round(pts_without, 1) if pts_without else None
                        ),
                        "PTS_IMPACT": round(pts_impact, 2),
                        "COMPOSITE_IMPACT": round(composite_impact, 2),
                    })
                    players_tracked += 1

                except Exception:
                    continue

            print(f"  [OK] {team_abbr} -- {players_tracked} players tracked")

    player_impact_df = pd.DataFrame(all_impacts)
    print(f"\n[OK] Total: {len(player_impact_df)} player impacts built")

    return player_impact_df
