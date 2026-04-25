# -*- coding: utf-8 -*-
"""
NextPlay -- Shot Chart Ingestion
=================================
Pulls team-level shot chart data from the NBA API.
Computes zone-based shooting rates and efficiency.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams as nba_teams_static

from config import API_DELAY


def get_team_shot_profile(team_id, team_abbr, season):
    """
    Pull shot chart for one team for one season.
    Returns a dict of shot zone percentages or None on failure.
    """
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=0,  # 0 = all players (team total)
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
        ).get_data_frames()[0]

        if len(shot_data) == 0:
            return None

        total_shots = len(shot_data)
        zone_counts = shot_data["SHOT_ZONE_BASIC"].value_counts()

        # Count shots by zone
        paint_shots = (
            zone_counts.get("Restricted Area", 0)
            + zone_counts.get("In The Paint (Non-RA)", 0)
        )
        threept_shots = (
            zone_counts.get("Left Corner 3", 0)
            + zone_counts.get("Right Corner 3", 0)
            + zone_counts.get("Above the Break 3", 0)
        )
        midrange_shots = zone_counts.get("Mid-Range", 0)

        # Shot accuracy by zone
        made = shot_data[shot_data["SHOT_MADE_FLAG"] == 1]
        made_zones = made["SHOT_ZONE_BASIC"].value_counts()

        paint_made = (
            made_zones.get("Restricted Area", 0)
            + made_zones.get("In The Paint (Non-RA)", 0)
        )
        threept_made = (
            made_zones.get("Left Corner 3", 0)
            + made_zones.get("Right Corner 3", 0)
            + made_zones.get("Above the Break 3", 0)
        )

        paint_pct = paint_made / paint_shots if paint_shots > 0 else 0
        threept_pct = threept_made / threept_shots if threept_shots > 0 else 0

        pts_per_shot = (
            (threept_made * 3) + (paint_made * 2)
            + (made_zones.get("Mid-Range", 0) * 2)
        ) / total_shots

        return {
            "TEAM_ABBR": team_abbr,
            "SEASON": season,
            "TOTAL_SHOTS": total_shots,
            "PAINT_RATE": round(paint_shots / total_shots, 4),
            "THREEPT_RATE": round(threept_shots / total_shots, 4),
            "MIDRANGE_RATE": round(midrange_shots / total_shots, 4),
            "PAINT_PCT": round(paint_pct, 4),
            "THREEPT_PCT": round(threept_pct, 4),
            "PTS_PER_SHOT": round(pts_per_shot, 4),
        }

    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        return None


def pull_all_shot_profiles(seasons):
    """
    Pull shot profiles for all teams across given seasons.

    Args:
        seasons: list of season strings

    Returns:
        DataFrame with shot profiles for all team-seasons
    """
    all_teams = nba_teams_static.get_teams()
    shot_profiles = []

    for season in seasons:
        print(f"\nPulling shot profiles -- {season}")
        print("-" * 40)

        for team in all_teams:
            profile = get_team_shot_profile(
                team["id"], team["abbreviation"], season
            )

            if profile:
                shot_profiles.append(profile)
                print(
                    f"  [OK] {team['abbreviation']} -- "
                    f"3PT: {profile['THREEPT_RATE']:.1%}  "
                    f"Paint: {profile['PAINT_RATE']:.1%}  "
                    f"Mid: {profile['MIDRANGE_RATE']:.1%}"
                )
            else:
                print(f"  [FAIL] {team['abbreviation']} -- no data")

            time.sleep(0.8)

    shot_df = pd.DataFrame(shot_profiles)
    print(f"\n[OK] Shot profiles pulled: {len(shot_df)} team-seasons")

    return shot_df
