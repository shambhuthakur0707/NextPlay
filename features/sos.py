# -*- coding: utf-8 -*-
"""
NextPlay -- Strength of Schedule Features
==========================================
Normalizes team performance based on opponent quality.
"""
import pandas as pd


def add_sos_features(games):
    """Add Strength of Schedule normalization features."""
    df = games.copy()

    league_avg_pts = df.groupby("SEASON")["HOME_PTS"].mean().to_dict()
    df["LEAGUE_AVG_PTS"] = df["SEASON"].map(league_avg_pts)

    all_app = pd.concat([
        df[["GAME_DATE", "GAME_ID", "HOME_TEAM", "AWAY_DEF_ROLL10"]].rename(
            columns={"HOME_TEAM": "TEAM", "AWAY_DEF_ROLL10": "OPP_DEF"}),
        df[["GAME_DATE", "GAME_ID", "AWAY_TEAM", "HOME_DEF_ROLL10"]].rename(
            columns={"AWAY_TEAM": "TEAM", "HOME_DEF_ROLL10": "OPP_DEF"}),
    ], ignore_index=True).sort_values(["TEAM", "GAME_DATE"])

    all_app["OPP_DEF_ROLL10"] = all_app.groupby("TEAM")[
        "OPP_DEF"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())

    sos_lkp = all_app[["GAME_ID", "TEAM", "OPP_DEF_ROLL10"]].drop_duplicates()

    df = df.merge(sos_lkp.rename(columns={
        "TEAM": "HOME_TEAM", "OPP_DEF_ROLL10": "HOME_SOS_DEF"
    }), on=["GAME_ID", "HOME_TEAM"], how="left")

    df = df.merge(sos_lkp.rename(columns={
        "TEAM": "AWAY_TEAM", "OPP_DEF_ROLL10": "AWAY_SOS_DEF"
    }), on=["GAME_ID", "AWAY_TEAM"], how="left")

    # Pace-adjusted offensive/defensive ratings (points per 100 possessions)
    df["HOME_OFF_RTG"] = 100 * df["HOME_ROLL10_PTS"] / df["HOME_ROLL10_POSS_EST"].clip(lower=70)
    df["AWAY_OFF_RTG"] = 100 * df["AWAY_ROLL10_PTS"] / df["AWAY_ROLL10_POSS_EST"].clip(lower=70)
    df["HOME_DEF_RTG"] = 100 * df["HOME_DEF_ROLL10"] / df["HOME_ROLL10_POSS_EST"].clip(lower=70)
    df["AWAY_DEF_RTG"] = 100 * df["AWAY_DEF_ROLL10"] / df["AWAY_ROLL10_POSS_EST"].clip(lower=70)
    df["HOME_NET_RTG"] = df["HOME_OFF_RTG"] - df["HOME_DEF_RTG"]
    df["AWAY_NET_RTG"] = df["AWAY_OFF_RTG"] - df["AWAY_DEF_RTG"]
    df["PACE_EST"] = (df["HOME_ROLL10_POSS_EST"] + df["AWAY_ROLL10_POSS_EST"]) / 2

    league_avg_off_rtg = pd.concat([
        df["HOME_OFF_RTG"],
        df["AWAY_OFF_RTG"],
    ], ignore_index=True).mean()

    # Opponent-quality normalization (proxy using opponent defensive rolling).
    opp_quality_home = (100 * df["HOME_SOS_DEF"] / df["HOME_ROLL10_POSS_EST"].clip(lower=70)).clip(lower=95)
    opp_quality_away = (100 * df["AWAY_SOS_DEF"] / df["AWAY_ROLL10_POSS_EST"].clip(lower=70)).clip(lower=95)

    df["HOME_ADJ_OFF_RTG"] = df["HOME_OFF_RTG"] * (league_avg_off_rtg / opp_quality_home)
    df["AWAY_ADJ_OFF_RTG"] = df["AWAY_OFF_RTG"] * (league_avg_off_rtg / opp_quality_away)
    df["HOME_ADJ_DEF_RTG"] = df["HOME_DEF_RTG"] * (league_avg_off_rtg / opp_quality_home)
    df["AWAY_ADJ_DEF_RTG"] = df["AWAY_DEF_RTG"] * (league_avg_off_rtg / opp_quality_away)
    df["HOME_ADJ_NET_RTG"] = df["HOME_ADJ_OFF_RTG"] - df["HOME_ADJ_DEF_RTG"]
    df["AWAY_ADJ_NET_RTG"] = df["AWAY_ADJ_OFF_RTG"] - df["AWAY_ADJ_DEF_RTG"]

    df["HOME_ADJ_OFF"] = df["HOME_ROLL10_PTS"] * (
        df["LEAGUE_AVG_PTS"] / df["HOME_SOS_DEF"].clip(lower=90))
    df["AWAY_ADJ_OFF"] = df["AWAY_ROLL10_PTS"] * (
        df["LEAGUE_AVG_PTS"] / df["AWAY_SOS_DEF"].clip(lower=90))
    df["HOME_ADJ_DEF"] = df["HOME_DEF_ROLL10"] * (
        df["LEAGUE_AVG_PTS"] / df["HOME_SOS_DEF"].clip(lower=90))
    df["AWAY_ADJ_DEF"] = df["AWAY_DEF_ROLL10"] * (
        df["LEAGUE_AVG_PTS"] / df["AWAY_SOS_DEF"].clip(lower=90))

    df["SOS_EXPECTED_HOME"] = (df["HOME_ADJ_OFF"] + df["AWAY_ADJ_DEF"]) / 2
    df["SOS_EXPECTED_AWAY"] = (df["AWAY_ADJ_OFF"] + df["HOME_ADJ_DEF"]) / 2
    df["SOS_EXPECTED_TOTAL"] = df["SOS_EXPECTED_HOME"] + df["SOS_EXPECTED_AWAY"]
    df["SOS_DIFF"] = df["HOME_SOS_DEF"] - df["AWAY_SOS_DEF"]

    return df
