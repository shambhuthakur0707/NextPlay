# -*- coding: utf-8 -*-
"""
NextPlay -- Full Rebuild Pipeline
===================================
Rebuilds everything from scratch: API pulls -> features -> training.
Run this when you need to start fresh or add a new season.
"""
import os
import pandas as pd

from config import (
    SEASONS, DATA_DIR, GAMELOGS_ALL_PATH, GAMES_ALL_PATH,
    SHOT_PROFILES_PATH, PLAYER_IMPACT_PATH, MODEL_READY_PATH,
    MARKET_LINES_PATH,
)
from ingestion.gamelogs import pull_multi_season, build_game_level_dataset
from ingestion.shots import pull_all_shot_profiles
from ingestion.players import build_player_impact
from ingestion.odds import pull_closing_market_lines
from features.pipeline import build_full_features
from models.train import train_models, save_models
from optimize import run_optimization


def full_rebuild(seasons=None, skip_api=False,
                 run_model_optimization=True,
                 optimize_walk_forward=True,
                 train_window=800, step=50,
                 include_market_lines=True):
    """
    Full pipeline rebuild from API to saved models.

    Args:
        seasons: list of seasons to process (default: config.SEASONS)
        skip_api: if True, load from saved CSVs instead of API
        run_model_optimization: run optimize.py over current dataset
        optimize_walk_forward: use walk-forward evaluation for optimization
        train_window: walk-forward training window size
        step: walk-forward step size
    """
    seasons = seasons or SEASONS
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 55)
    print("FULL PIPELINE REBUILD")
    print("=" * 55)

    # Step 1: Game logs
    if skip_api and os.path.exists(GAMELOGS_ALL_PATH):
        print("\n[LOAD] Loading gamelogs from disk...")
        gamelogs = pd.read_csv(GAMELOGS_ALL_PATH)
        gamelogs["GAME_DATE"] = pd.to_datetime(gamelogs["GAME_DATE"])
    else:
        print("\n[STEP 1] Pulling game logs from NBA API...")
        gamelogs = pull_multi_season(seasons)
        gamelogs.to_csv(GAMELOGS_ALL_PATH, index=False)

    # Step 2: Build game-level dataset
    print("\n[STEP 2] Building game-level dataset...")
    games = build_game_level_dataset(gamelogs)
    games.to_csv(GAMES_ALL_PATH, index=False)

    # Step 3: Shot profiles
    if skip_api and os.path.exists(SHOT_PROFILES_PATH):
        print("\n[LOAD] Loading shot profiles from disk...")
        shot_df = pd.read_csv(SHOT_PROFILES_PATH)
    else:
        print("\n[STEP 3] Pulling shot profiles...")
        shot_df = pull_all_shot_profiles(seasons)
        shot_df.to_csv(SHOT_PROFILES_PATH, index=False)

    # Step 4: Player impact
    if skip_api and os.path.exists(PLAYER_IMPACT_PATH):
        print("\n[LOAD] Loading player impact from disk...")
        player_impact_df = pd.read_csv(PLAYER_IMPACT_PATH)
    else:
        print("\n[STEP 4] Building player impact data...")
        player_impact_df = build_player_impact(gamelogs, seasons)
        player_impact_df.to_csv(PLAYER_IMPACT_PATH, index=False)

    # Step 5: Market lines
    market_df = None
    if include_market_lines:
        if skip_api and os.path.exists(MARKET_LINES_PATH):
            print("\n[LOAD] Loading market lines from disk...")
            market_df = pd.read_csv(MARKET_LINES_PATH)
        else:
            print("\n[STEP 5] Pulling market lines from Odds API...")
            try:
                market_df = pull_closing_market_lines(
                    date_from=games["GAME_DATE"].min(),
                    date_to=games["GAME_DATE"].max(),
                )
                if len(market_df) > 0:
                    market_df.to_csv(MARKET_LINES_PATH, index=False)
                else:
                    print("  [WARN] No market lines returned; continuing without market features")
            except Exception as e:
                print(f"  [WARN] Market pull failed: {e}")
                market_df = None

    # Step 6: Feature engineering
    print("\n[STEP 6] Feature engineering...")
    model_df = build_full_features(
        games, shot_df=shot_df,
        player_impact_df=player_impact_df,
        market_df=market_df,
        extend_player_season=seasons[-1],
    )
    model_df.to_csv(MODEL_READY_PATH, index=False)

    # Step 7: Train/optimize models
    print("\n[STEP 7] Training models...")
    train_seasons = seasons[:-1] if len(seasons) > 1 else seasons
    test_season = seasons[-1]

    if run_model_optimization:
        print("\n[OPTIMIZE] Running model optimization...")
        result = run_optimization(
            model_df,
            use_walk_forward=optimize_walk_forward,
            train_window=train_window,
            step=step,
        )
    else:
        result = train_models(
            model_df,
            train_seasons=train_seasons,
            test_season=test_season,
        )
        save_models(result)

    print(f"\n{'=' * 55}")
    print(f"[OK] FULL REBUILD COMPLETE")
    print(f"   Games: {len(model_df)}")
    if "feature_cols" in result:
        print(f"   Features: {len(result['feature_cols'])}")
    if "mae_total" in result:
        print(f"   Total MAE: {result['mae_total']:.2f} pts")
    print(f"{'=' * 55}")

    return model_df, result


if __name__ == "__main__":
    full_rebuild(skip_api=True)
