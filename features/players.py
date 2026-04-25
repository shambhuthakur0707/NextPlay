# -*- coding: utf-8 -*-
"""
NextPlay -- Player Impact Features
===================================
Aggregates player-level impact data into team-level features.
"""

import pandas as pd


def build_team_player_features(player_impact_df, min_games=15):
    """
    Build team-level aggregated player features from individual impacts.

    Returns DataFrame with team-season level features like
    TOP_IMPACT, STAR_DEPENDENCY, BENCH_STRENGTH, etc.
    """
    reliable = player_impact_df[
        player_impact_df["GAMES_PLAYED"] >= min_games
    ].copy()

    reliable = reliable.copy()
    reliable["TOTAL_GAMES"] = (
        reliable["GAMES_PLAYED"] + reliable["GAMES_MISSED"]
    )
    reliable["MISS_RATE"] = reliable.apply(
        lambda r: r["GAMES_MISSED"] / r["TOTAL_GAMES"]
        if r["TOTAL_GAMES"] > 0 else 0.0,
        axis=1,
    )

    team_feat = reliable.groupby(["TEAM_ABBR", "SEASON"]).agg(
        TOP_IMPACT=("COMPOSITE_IMPACT", "max"),
        TOP3_IMPACT_SUM=("COMPOSITE_IMPACT", lambda x: x.nlargest(3).sum()),
        AVG_PLAYER_IMPACT=("COMPOSITE_IMPACT", "mean"),
        DEPTH_COUNT=("PLAYER_ID", "count"),
        STAR_DEPENDENCY=(
            "COMPOSITE_IMPACT",
            lambda x: x.max() / x.sum() if x.sum() > 0 else 0,
        ),
        BENCH_STRENGTH=(
            "COMPOSITE_IMPACT",
            lambda x: x.nsmallest(max(len(x) - 3, 1)).mean(),
        ),
        AVG_MISS_RATE=("MISS_RATE", "mean"),
        TOP_MISS_RATE=("MISS_RATE", "max"),
        TOP3_MISS_RATE=(
            "MISS_RATE",
            lambda x: x.nlargest(3).mean(),
        ),
        IMPACT_AVAIL=(
            "COMPOSITE_IMPACT",
            lambda x: x.sum(),
        ),
    ).reset_index()

    impact_avail = reliable.groupby(["TEAM_ABBR", "SEASON"]).apply(
        lambda g: (
            (g["COMPOSITE_IMPACT"] * (1 - g["MISS_RATE"]).fillna(0)).sum()
        ),
        include_groups=False,
    ).reset_index(name="IMPACT_AVAIL")

    team_feat = team_feat.drop(columns=["IMPACT_AVAIL"]).merge(
        impact_avail,
        on=["TEAM_ABBR", "SEASON"],
        how="left",
    )

    return team_feat


def add_player_features(games, player_impact_df, extend_season=None):
    """
    Merge team-level player features onto the game dataset.

    Args:
        games: game-level DataFrame
        player_impact_df: player impact DataFrame
        extend_season: if set, copies the previous season's features
                       to this season (e.g. '2025-26' copies from '2024-25')
    """
    team_feat = build_team_player_features(player_impact_df)

    # Optionally extend to a new season using previous season as proxy
    if extend_season:
        prev_season = team_feat["SEASON"].unique()
        prev_season = sorted(prev_season)[-1]
        extended = team_feat[team_feat["SEASON"] == prev_season].copy()
        extended["SEASON"] = extend_season
        team_feat = pd.concat(
            [team_feat, extended], ignore_index=True
        ).drop_duplicates(subset=["TEAM_ABBR", "SEASON"])

    # Merge home features
    home_feat = team_feat.rename(columns={
        "TEAM_ABBR": "HOME_TEAM",
        "TOP_IMPACT": "HOME_TOP_IMPACT",
        "TOP3_IMPACT_SUM": "HOME_TOP3_IMPACT",
        "AVG_PLAYER_IMPACT": "HOME_AVG_IMPACT",
        "DEPTH_COUNT": "HOME_DEPTH",
        "STAR_DEPENDENCY": "HOME_STAR_DEP",
        "BENCH_STRENGTH": "HOME_BENCH",
        "AVG_MISS_RATE": "HOME_AVG_MISS",
        "TOP_MISS_RATE": "HOME_TOP_MISS",
        "TOP3_MISS_RATE": "HOME_TOP3_MISS",
        "IMPACT_AVAIL": "HOME_IMPACT_AVAIL",
    })

    away_feat = team_feat.rename(columns={
        "TEAM_ABBR": "AWAY_TEAM",
        "TOP_IMPACT": "AWAY_TOP_IMPACT",
        "TOP3_IMPACT_SUM": "AWAY_TOP3_IMPACT",
        "AVG_PLAYER_IMPACT": "AWAY_AVG_IMPACT",
        "DEPTH_COUNT": "AWAY_DEPTH",
        "STAR_DEPENDENCY": "AWAY_STAR_DEP",
        "BENCH_STRENGTH": "AWAY_BENCH",
        "AVG_MISS_RATE": "AWAY_AVG_MISS",
        "TOP_MISS_RATE": "AWAY_TOP_MISS",
        "TOP3_MISS_RATE": "AWAY_TOP3_MISS",
        "IMPACT_AVAIL": "AWAY_IMPACT_AVAIL",
    })

    home_cols = [
        "HOME_TEAM", "SEASON", "HOME_TOP_IMPACT", "HOME_TOP3_IMPACT",
        "HOME_AVG_IMPACT", "HOME_DEPTH", "HOME_STAR_DEP", "HOME_BENCH",
        "HOME_AVG_MISS", "HOME_TOP_MISS", "HOME_TOP3_MISS",
        "HOME_IMPACT_AVAIL",
    ]
    away_cols = [
        "AWAY_TEAM", "SEASON", "AWAY_TOP_IMPACT", "AWAY_TOP3_IMPACT",
        "AWAY_AVG_IMPACT", "AWAY_DEPTH", "AWAY_STAR_DEP", "AWAY_BENCH",
        "AWAY_AVG_MISS", "AWAY_TOP_MISS", "AWAY_TOP3_MISS",
        "AWAY_IMPACT_AVAIL",
    ]

    result = games.merge(
        home_feat[home_cols], on=["HOME_TEAM", "SEASON"], how="left"
    ).merge(
        away_feat[away_cols], on=["AWAY_TEAM", "SEASON"], how="left"
    )

    return result
