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
