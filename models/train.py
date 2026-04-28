# -*- coding: utf-8 -*-
"""
NextPlay -- Model Training
===========================
Trains RandomForest models A (home), B (away), C (total).
Includes garbage time and OT filtering for clean training data.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import (
    RF_PARAMS, FEATURE_COLS_FINAL,
    STACKED_TOTAL_FEATURES,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
    MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH,
    MODEL_A_PLAYOFF_PATH, MODEL_B_PLAYOFF_PATH, MODEL_C_PLAYOFF_PATH,
    WEIGHT_PLAYOFF_ROWS,
)
from models.stacking import build_total_meta_features


def filter_training_data(train_df):
    """
    Remove blowout and overtime games from training data.
    These outliers teach the model wrong patterns.
    """
    before = len(train_df)

    clean = train_df[
        (train_df["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
        ~(
            (train_df["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
            (train_df["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD)
        )
    ].copy()

    after = len(clean)
    print(f"  Garbage time filter: {before} -> {after} "
          f"(removed {before - after} games, "
          f"{(before - after) / before:.1%})")

    return clean


def train_models(model_df, train_seasons=None, test_season=None,
                 apply_filter=True, weight_playoff_rows=None):
    """
    Train RF models A/B/C on the provided dataset.

    Args:
        model_df: feature-engineered DataFrame
        train_seasons: list of seasons for training
        test_season: season for testing
        apply_filter: whether to apply garbage time filter

    Returns:
        dict with models, MAE scores, and feature columns used
    """
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    meta_feature_cols = STACKED_TOTAL_FEATURES

    if train_seasons and test_season:
        train = model_df[model_df["SEASON"].isin(train_seasons)].copy()
        test = model_df[model_df["SEASON"] == test_season].copy()
    else:
        # 80/20 time-based split
        model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)
        split_idx = int(len(model_df) * 0.80)
        train = model_df.iloc[:split_idx].copy()
        test = model_df.iloc[split_idx:].copy()

    if apply_filter and "PTS_MARGIN" in train.columns:
        train = filter_training_data(train)

    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    # Build sample weights to downweight playoff rows if requested
    if weight_playoff_rows is None:
        weight_playoff_rows = WEIGHT_PLAYOFF_ROWS

    # default weight = 1.0 for regular-season rows
    train_weights = np.ones(len(train), dtype=float)
    if "IS_PLAYOFF" in train.columns and weight_playoff_rows is not None:
        train_weights = np.where(train["IS_PLAYOFF"], weight_playoff_rows, 1.0)

    print(f"\n  Training: {len(train)} games")
    print(f"  Testing:  {len(test)} games")
    print(f"  Features: {len(feature_cols)}")
    print(f"\n  Training models...")

    # Model A: Home score
    rf_A = RandomForestRegressor(**RF_PARAMS)
    rf_A.fit(X_train, train["HOME_PTS"], sample_weight=train_weights)
    mae_home = mean_absolute_error(test["HOME_PTS"], rf_A.predict(X_test))

    # Model B: Away score
    rf_B = RandomForestRegressor(**RF_PARAMS)
    rf_B.fit(X_train, train["AWAY_PTS"], sample_weight=train_weights)
    mae_away = mean_absolute_error(test["AWAY_PTS"], rf_B.predict(X_test))

    # Model C: stacked total points
    train_meta = build_total_meta_features(
        train[feature_cols],
        rf_A.predict(X_train),
        rf_B.predict(X_train),
        feature_cols=meta_feature_cols,
    )
    test_meta = build_total_meta_features(
        test[feature_cols],
        rf_A.predict(X_test),
        rf_B.predict(X_test),
        feature_cols=meta_feature_cols,
    )
    rf_C = RandomForestRegressor(**RF_PARAMS)
    rf_C.fit(train_meta, train["TOTAL_PTS"], sample_weight=train_weights)
    mae_total = mean_absolute_error(test["TOTAL_PTS"], rf_C.predict(test_meta))

    print(f"\n  {'=' * 45}")
    print(f"  RESULTS")
    print(f"  {'=' * 45}")
    print(f"  Home MAE  : {mae_home:.2f} pts")
    print(f"  Away MAE  : {mae_away:.2f} pts")
    print(f"  Total MAE : {mae_total:.2f} pts")

    return {
        "model_A": rf_A,
        "model_B": rf_B,
        "model_C": rf_C,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
        "feature_cols": feature_cols,
        "meta_feature_cols": meta_feature_cols,
    }


def save_models(models_dict, path_a=MODEL_A_PATH,
                path_b=MODEL_B_PATH, path_c=MODEL_C_PATH):
    """Save trained models to pickle files."""
    with open(path_a, "wb") as f:
        pickle.dump(models_dict["model_A"], f)
    with open(path_b, "wb") as f:
        pickle.dump(models_dict["model_B"], f)
    with open(path_c, "wb") as f:
        pickle.dump(models_dict["model_C"], f)
    print("[OK] Models saved")


def load_models(path_a=MODEL_A_PATH, path_b=MODEL_B_PATH,
                path_c=MODEL_C_PATH):
    """Load trained models from pickle files."""
    with open(path_a, "rb") as f:
        model_A = pickle.load(f)
    with open(path_b, "rb") as f:
        model_B = pickle.load(f)
    with open(path_c, "rb") as f:
        model_C = pickle.load(f)
    print("[OK] Models loaded")
    return {"model_A": model_A, "model_B": model_B, "model_C": model_C}


def train_playoff_models(model_df):
    """
    Train models on playoff games only (IS_PLAYOFF==True).
    
    Args:
        model_df: feature-engineered DataFrame with IS_PLAYOFF column
    
    Returns:
        dict with playoff models, MAE scores, and feature columns
    """
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    meta_feature_cols = STACKED_TOTAL_FEATURES
    
    # Filter to playoff games only
    playoff_df = model_df[model_df.get("IS_PLAYOFF", False)].copy()
    
    if len(playoff_df) == 0:
        print("[WARN] No playoff games found in dataset. Returning None.")
        return None
    
    playoff_df = playoff_df.dropna(subset=feature_cols)
    
    # Use 70/30 split since playoff dataset is small
    playoff_df = playoff_df.sort_values("GAME_DATE").reset_index(drop=True)
    split_idx = int(len(playoff_df) * 0.70)
    train = playoff_df.iloc[:split_idx].copy()
    test = playoff_df.iloc[split_idx:].copy()
    
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    
    print(f"\n  [PLAYOFF] Training: {len(train)} games")
    print(f"  [PLAYOFF] Testing:  {len(test)} games")
    print(f"  [PLAYOFF] Features: {len(feature_cols)}")
    
    # Model A: Home score
    rf_A = RandomForestRegressor(**RF_PARAMS)
    rf_A.fit(X_train, train["HOME_PTS"])
    mae_home = mean_absolute_error(test["HOME_PTS"], rf_A.predict(X_test))
    
    # Model B: Away score
    rf_B = RandomForestRegressor(**RF_PARAMS)
    rf_B.fit(X_train, train["AWAY_PTS"])
    mae_away = mean_absolute_error(test["AWAY_PTS"], rf_B.predict(X_test))
    
    # Model C: stacked total
    train_meta = build_total_meta_features(
        train[feature_cols],
        rf_A.predict(X_train),
        rf_B.predict(X_train),
        feature_cols=meta_feature_cols,
    )
    test_meta = build_total_meta_features(
        test[feature_cols],
        rf_A.predict(X_test),
        rf_B.predict(X_test),
        feature_cols=meta_feature_cols,
    )
    rf_C = RandomForestRegressor(**RF_PARAMS)
    rf_C.fit(train_meta, train["TOTAL_PTS"])
    mae_total = mean_absolute_error(test["TOTAL_PTS"], rf_C.predict(test_meta))
    
    print(f"\n  [PLAYOFF] {'=' * 45}")
    print(f"  [PLAYOFF] RESULTS")
    print(f"  [PLAYOFF] {'=' * 45}")
    print(f"  [PLAYOFF] Home MAE  : {mae_home:.2f} pts")
    print(f"  [PLAYOFF] Away MAE  : {mae_away:.2f} pts")
    print(f"  [PLAYOFF] Total MAE : {mae_total:.2f} pts")
    
    return {
        "model_A": rf_A,
        "model_B": rf_B,
        "model_C": rf_C,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
        "feature_cols": feature_cols,
        "meta_feature_cols": meta_feature_cols,
    }


def save_playoff_models(models_dict, path_a=MODEL_A_PLAYOFF_PATH,
                        path_b=MODEL_B_PLAYOFF_PATH, path_c=MODEL_C_PLAYOFF_PATH):
    """Save playoff models to pickle files."""
    with open(path_a, "wb") as f:
        pickle.dump(models_dict["model_A"], f)
    with open(path_b, "wb") as f:
        pickle.dump(models_dict["model_B"], f)
    with open(path_c, "wb") as f:
        pickle.dump(models_dict["model_C"], f)
    print("[OK] Playoff models saved")


def load_playoff_models(path_a=MODEL_A_PLAYOFF_PATH,
                        path_b=MODEL_B_PLAYOFF_PATH,
                        path_c=MODEL_C_PLAYOFF_PATH):
    """Load playoff models from pickle files."""
    try:
        with open(path_a, "rb") as f:
            model_A = pickle.load(f)
        with open(path_b, "rb") as f:
            model_B = pickle.load(f)
        with open(path_c, "rb") as f:
            model_C = pickle.load(f)
        print("[OK] Playoff models loaded")
        return {"model_A": model_A, "model_B": model_B, "model_C": model_C}
    except FileNotFoundError:
        return None

