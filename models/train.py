# -*- coding: utf-8 -*-
"""
NextPlay -- Model Training
===========================
Trains models A (home), B (away), C (total) using any sklearn-compatible
estimator. Defaults to RandomForest but the optimizer can inject XGBoost,
CatBoost, LightGBM, or VotingRegressor.
Includes garbage time and OT filtering for clean training data.

FIX (v9): Model C now uses 5-fold out-of-fold (OOF) predictions from
Models A and B instead of in-sample predictions. This is the correct
stacking procedure -- training Model C on A/B's predictions of their
own training data causes overfitting because those predictions are
near-perfect. OOF predictions are honest estimates of what A/B would
predict on unseen data, giving Model C a realistic signal to learn from.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from config import (
    RF_PARAMS, FEATURE_COLS_FINAL,
    STACKED_TOTAL_FEATURES,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
    MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH,
    MODEL_A_PLAYOFF_PATH, MODEL_B_PLAYOFF_PATH, MODEL_C_PLAYOFF_PATH,
    WEIGHT_PLAYOFF_ROWS,
)
from models.stacking import build_total_meta_features


def _default_model_factory():
    return RandomForestRegressor(**RF_PARAMS)


def filter_training_data(train_df):
    """Remove blowout and overtime games from training data."""
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


def _build_oof_predictions(X, y_home, y_away, model_factory, n_splits=5):
    """
    Build out-of-fold predictions for Models A and B.

    Instead of predicting on training data (which causes overfitting),
    we use k-fold cross-validation to generate honest predictions:
    - Split training data into N folds
    - For each fold: train on N-1 folds, predict on the held-out fold
    - Stitch predictions back together in original order

    This gives Model C realistic signal -- the predictions it sees
    are from models that never saw those rows during training.

    Args:
        X: feature matrix
        y_home: home points target
        y_away: away points target
        model_factory: callable returning a fresh model instance
        n_splits: number of CV folds (default 5)

    Returns:
        oof_home: array of OOF home predictions
        oof_away: array of OOF away predictions
    """
    oof_home = np.zeros(len(X))
    oof_away = np.zeros(len(X))

    kf = KFold(n_splits=n_splits, shuffle=False)  # no shuffle -- temporal order

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]

        y_home_train = y_home.iloc[train_idx]
        y_away_train = y_away.iloc[train_idx]

        model_a = model_factory()
        model_b = model_factory()

        model_a.fit(X_fold_train, y_home_train)
        model_b.fit(X_fold_train, y_away_train)

        oof_home[val_idx] = model_a.predict(X_fold_val)
        oof_away[val_idx] = model_b.predict(X_fold_val)

    return oof_home, oof_away


def train_models(model_df, train_seasons=None, test_season=None,
                 apply_filter=True, weight_playoff_rows=None,
                 n_oof_splits=5):
    """
    Train RF models A/B/C on the provided dataset.

    Model C uses OOF predictions from A/B to avoid stacking leakage.

    Args:
        model_df: feature-engineered DataFrame
        train_seasons: list of seasons for training
        test_season: season for testing
        apply_filter: whether to apply garbage time filter
        weight_playoff_rows: playoff row sample weight
        n_oof_splits: number of OOF folds for Model C training (default 5)

    Returns:
        dict with models, MAE scores, and feature columns used
    """
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    meta_feature_cols = STACKED_TOTAL_FEATURES

    if train_seasons and test_season:
        train = model_df[model_df["SEASON"].isin(train_seasons)].copy()
        test = model_df[model_df["SEASON"] == test_season].copy()
    else:
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

    if weight_playoff_rows is None:
        weight_playoff_rows = WEIGHT_PLAYOFF_ROWS

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

    # Model C: stacked total using OOF predictions
    # FIX: use OOF predictions instead of in-sample predictions
    print(f"  Building OOF predictions for Model C ({n_oof_splits}-fold)...")
    oof_home, oof_away = _build_oof_predictions(
        X_train,
        train["HOME_PTS"],
        train["AWAY_PTS"],
        _default_model_factory,
        n_splits=n_oof_splits,
    )

    train_meta = build_total_meta_features(
        X_train,
        oof_home,      # OOF predictions -- honest signal
        oof_away,      # OOF predictions -- honest signal
        feature_cols=meta_feature_cols,
    )
    test_meta = build_total_meta_features(
        X_test,
        rf_A.predict(X_test),   # test predictions from fully trained A
        rf_B.predict(X_test),   # test predictions from fully trained B
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
    with open(path_a, "wb") as f:
        pickle.dump(models_dict["model_A"], f)
    with open(path_b, "wb") as f:
        pickle.dump(models_dict["model_B"], f)
    with open(path_c, "wb") as f:
        pickle.dump(models_dict["model_C"], f)
    print("[OK] Models saved")


def load_models(path_a=MODEL_A_PATH, path_b=MODEL_B_PATH,
                path_c=MODEL_C_PATH):
    with open(path_a, "rb") as f:
        model_A = pickle.load(f)
    with open(path_b, "rb") as f:
        model_B = pickle.load(f)
    with open(path_c, "rb") as f:
        model_C = pickle.load(f)
    print("[OK] Models loaded")
    return {"model_A": model_A, "model_B": model_B, "model_C": model_C}


def train_models_v2(model_df, model_factory_a=None, model_factory_b=None,
                    model_factory_c=None, train_seasons=None,
                    test_season=None, apply_filter=True,
                    weight_playoff_rows=None, n_oof_splits=5):
    """
    Train models A/B/C using injectable model factories.
    Model C uses OOF predictions to avoid stacking leakage.
    """
    model_factory_a = model_factory_a or _default_model_factory
    model_factory_b = model_factory_b or _default_model_factory
    model_factory_c = model_factory_c or _default_model_factory

    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    meta_feature_cols = STACKED_TOTAL_FEATURES

    if train_seasons and test_season:
        train = model_df[model_df["SEASON"].isin(train_seasons)].copy()
        test = model_df[model_df["SEASON"] == test_season].copy()
    else:
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

    if weight_playoff_rows is None:
        weight_playoff_rows = WEIGHT_PLAYOFF_ROWS
    train_weights = np.ones(len(train), dtype=float)
    if "IS_PLAYOFF" in train.columns and weight_playoff_rows is not None:
        train_weights = np.where(train["IS_PLAYOFF"], weight_playoff_rows, 1.0)

    print(f"\n  Training: {len(train)} games")
    print(f"  Testing:  {len(test)} games")
    print(f"  Features: {len(feature_cols)}")

    # Model A
    rf_A = model_factory_a()
    model_a_name = type(rf_A).__name__
    print(f"\n  Training Model A ({model_a_name})...")
    try:
        rf_A.fit(X_train, train["HOME_PTS"], sample_weight=train_weights)
    except TypeError:
        rf_A.fit(X_train, train["HOME_PTS"])
    mae_home = mean_absolute_error(test["HOME_PTS"], rf_A.predict(X_test))

    # Model B
    rf_B = model_factory_b()
    model_b_name = type(rf_B).__name__
    print(f"  Training Model B ({model_b_name})...")
    try:
        rf_B.fit(X_train, train["AWAY_PTS"], sample_weight=train_weights)
    except TypeError:
        rf_B.fit(X_train, train["AWAY_PTS"])
    mae_away = mean_absolute_error(test["AWAY_PTS"], rf_B.predict(X_test))

    # Model C: OOF stacking
    print(f"  Building OOF predictions for Model C ({n_oof_splits}-fold)...")
    oof_home, oof_away = _build_oof_predictions(
        X_train,
        train["HOME_PTS"],
        train["AWAY_PTS"],
        model_factory_a,
        n_splits=n_oof_splits,
    )

    train_meta = build_total_meta_features(
        X_train,
        oof_home,
        oof_away,
        feature_cols=meta_feature_cols,
    )
    test_meta = build_total_meta_features(
        X_test,
        rf_A.predict(X_test),
        rf_B.predict(X_test),
        feature_cols=meta_feature_cols,
    )

    rf_C = model_factory_c()
    model_c_name = type(rf_C).__name__
    print(f"  Training Model C ({model_c_name})...")
    try:
        rf_C.fit(train_meta, train["TOTAL_PTS"], sample_weight=train_weights)
    except TypeError:
        rf_C.fit(train_meta, train["TOTAL_PTS"])
    mae_total = mean_absolute_error(test["TOTAL_PTS"], rf_C.predict(test_meta))

    print(f"\n  {'=' * 45}")
    print(f"  RESULTS  (A={model_a_name}, B={model_b_name}, C={model_c_name})")
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


def train_playoff_models(model_df, upweight_factor=None):
    """
    Train playoff-optimized models using ALL games with playoff upweighting.
    Uses OOF predictions for Model C stacking.
    """
    from config import PLAYOFF_UPWEIGHT_FACTOR

    if upweight_factor is None:
        upweight_factor = PLAYOFF_UPWEIGHT_FACTOR

    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    meta_feature_cols = STACKED_TOTAL_FEATURES

    has_playoff = "IS_PLAYOFF" in model_df.columns
    n_playoff = int(model_df["IS_PLAYOFF"].sum()) if has_playoff else 0

    if n_playoff == 0:
        print("[WARN] No playoff games found in dataset. Returning None.")
        return None

    model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)
    model_df = model_df.dropna(subset=feature_cols)

    split_idx = int(len(model_df) * 0.80)
    train = model_df.iloc[:split_idx].copy()
    test_all = model_df.iloc[split_idx:].copy()

    test_playoff = test_all[test_all["IS_PLAYOFF"] == 1].copy()
    test = test_playoff if len(test_playoff) >= 10 else test_all

    train_weights = np.where(
        train["IS_PLAYOFF"] == 1, upweight_factor, 1.0
    )

    n_train_playoff = int(train["IS_PLAYOFF"].sum())
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    print(f"\n  [PLAYOFF] Training: {len(train)} games "
          f"({n_train_playoff} playoff, upweight={upweight_factor}x)")
    print(f"  [PLAYOFF] Testing:  {len(test)} games "
          f"({int(test['IS_PLAYOFF'].sum())} playoff)")
    print(f"  [PLAYOFF] Features: {len(feature_cols)}")

    rf_A = RandomForestRegressor(**RF_PARAMS)
    rf_A.fit(X_train, train["HOME_PTS"], sample_weight=train_weights)
    mae_home = mean_absolute_error(test["HOME_PTS"], rf_A.predict(X_test))

    rf_B = RandomForestRegressor(**RF_PARAMS)
    rf_B.fit(X_train, train["AWAY_PTS"], sample_weight=train_weights)
    mae_away = mean_absolute_error(test["AWAY_PTS"], rf_B.predict(X_test))

    # OOF stacking for playoff Model C
    print(f"  [PLAYOFF] Building OOF predictions for Model C (5-fold)...")
    oof_home, oof_away = _build_oof_predictions(
        X_train,
        train["HOME_PTS"],
        train["AWAY_PTS"],
        _default_model_factory,
        n_splits=5,
    )

    train_meta = build_total_meta_features(
        X_train,
        oof_home,
        oof_away,
        feature_cols=meta_feature_cols,
    )
    test_meta = build_total_meta_features(
        X_test,
        rf_A.predict(X_test),
        rf_B.predict(X_test),
        feature_cols=meta_feature_cols,
    )

    rf_C = RandomForestRegressor(**RF_PARAMS)
    rf_C.fit(train_meta, train["TOTAL_PTS"], sample_weight=train_weights)
    mae_total = mean_absolute_error(test["TOTAL_PTS"], rf_C.predict(test_meta))

    print(f"\n  [PLAYOFF] {'=' * 45}")
    print(f"  [PLAYOFF] RESULTS (blended + upweighted)")
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
                        path_b=MODEL_B_PLAYOFF_PATH,
                        path_c=MODEL_C_PLAYOFF_PATH):
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