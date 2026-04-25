# -*- coding: utf-8 -*-
"""
NextPlay -- Full Feature Pipeline
==================================
Orchestrates all feature engineering steps in the correct order.
"""
import pandas as pd
from features.rolling import add_rolling_features
from features.rest_streak import add_rest_and_streak
from features.context import add_context_features
from features.defensive import add_defensive_features
from features.ewm import add_ewm_features
from features.sos import add_sos_features
from features.matchup import add_matchup_features
from features.shots import merge_shot_profiles
from features.players import add_player_features
from features.elo import add_elo_features
from config import FEATURE_COLS_FINAL, META_COLS, ELO_RATINGS_PATH


def build_full_features(games, shot_df=None, player_impact_df=None,
                        extend_player_season=None, elo_path=None):
    """
    Run the complete feature engineering pipeline.

    Args:
        games: game-level DataFrame (output of build_game_level_dataset)
        shot_df: shot profile DataFrame (optional)
        player_impact_df: player impact DataFrame (optional)
        extend_player_season: season to extend player features to
        elo_path: path to ELO ratings CSV (default: config.ELO_RATINGS_PATH)

    Returns:
        model_df: DataFrame ready for model training/prediction
    """
    print("Building full feature pipeline...")
    elo_path = elo_path or ELO_RATINGS_PATH

    # Step 1: Rolling averages
    print("  1/10 Rolling features...")
    df = add_rolling_features(games)

    # Drop rows without enough rolling history
    df = df.dropna(subset=["HOME_ROLL10_PTS", "AWAY_ROLL10_PTS"]).copy()
    df = df.reset_index(drop=True)

    # Step 2: Matchup-specific history
    print("  2/10 Matchup features...")
    df = add_matchup_features(df)

    # Step 3: Rest days + win streaks
    print("  3/10 Rest + streak features...")
    df = add_rest_and_streak(df)

    # Step 4: Season context + home court
    print("  4/10 Context features...")
    df = add_context_features(df)

    # Step 5: Defensive rolling
    print("  5/10 Defensive features...")
    df = add_defensive_features(df)

    # Step 6: EWM features
    print("  6/10 EWM features...")
    df = add_ewm_features(df)

    # Step 7: SoS normalization
    print("  7/10 Strength of Schedule features...")
    df = add_sos_features(df)

    # Step 8: Shot profiles (if provided)
    if shot_df is not None:
        print("  8/10 Shot profile features...")
        df = merge_shot_profiles(df, shot_df)
    else:
        print("  8/10 Shot profiles -- SKIPPED (no data)")

    # Step 9: Player features (if provided)
    if player_impact_df is not None:
        print("  9/10 Player impact features...")
        df = add_player_features(
            df, player_impact_df,
            extend_season=extend_player_season
        )
    else:
        print("  9/10 Player features -- SKIPPED (no data)")

    # Step 10: ELO ratings
    print("  10/10 ELO rating features...")
    df = add_elo_features(df, elo_path=elo_path)

    # Add margin rolling (V7 feature)
    df["HOME_ROLL10_MARGIN"] = df.groupby("HOME_TEAM")[
        "PTS_MARGIN"
    ].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    df["AWAY_ROLL10_MARGIN"] = df.groupby("AWAY_TEAM")[
        "PTS_MARGIN"
    ].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())

    # Drop rows with NaN in required features
    available_features = [c for c in FEATURE_COLS_FINAL if c in df.columns]
    available_meta = [c for c in META_COLS if c in df.columns]

    model_df = df[available_features + available_meta].dropna().reset_index(
        drop=True
    )

    print(f"\n[OK] Feature pipeline complete")
    print(f"   Games: {len(model_df)}")
    print(f"   Features: {len(available_features)}")
    if "SEASON" in model_df.columns:
        print(f"   Seasons: {model_df['SEASON'].value_counts().to_dict()}")

    return model_df
