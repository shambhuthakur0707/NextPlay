"""Model optimization - try multiple approaches to minimize MAE."""
import warnings
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from config import (
    MODEL_READY_PATH, FEATURE_COLS_FINAL,
    MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
)

warnings.filterwarnings("ignore")


def _build_model_configs():
    configs = {
        "RF-400": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=9, min_samples_leaf=15,
            random_state=42, n_jobs=-1),
        "RF-600-d12": lambda: RandomForestRegressor(
            n_estimators=600, max_depth=12, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
        "RF-800-d15": lambda: RandomForestRegressor(
            n_estimators=800, max_depth=15, min_samples_leaf=8,
            random_state=42, n_jobs=-1),
        "GB-500": lambda: GradientBoostingRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42),
        "GB-800": lambda: GradientBoostingRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            min_samples_leaf=10, subsample=0.8, random_state=42),
    }

    try:
        import lightgbm as lgb
        configs["LGB-500"] = lambda: lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1)
        configs["LGB-800"] = lambda: lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=45,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1)
        configs["LGB-1000"] = lambda: lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.02, num_leaves=63,
            min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)
    except ImportError:
        print("LightGBM not available")

    return configs


def _apply_garbage_filter(train_df):
    if "PTS_MARGIN" not in train_df.columns:
        return train_df

    return train_df[
        (train_df["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
        ~((train_df["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
          (train_df["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD))
    ]


def _time_split_mae(model_df, model_fn, target, feature_cols, apply_filter=True):
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_idx = int(len(df) * 0.80)
    train_all = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    if apply_filter:
        train_all = _apply_garbage_filter(train_all)

    train = train_all.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    if len(train) < 50 or len(test) == 0:
        return np.nan

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    model = model_fn()
    model.fit(X_train, train[target])
    pred = model.predict(X_test)
    return mean_absolute_error(test[target], pred)


def _walk_forward_mae(
    model_df,
    model_fn,
    target,
    feature_cols,
    train_window=800,
    step=50,
    apply_filter=True,
):
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    total_games = len(df)
    n_batches = (total_games - train_window) // step
    errors = []

    for i in range(n_batches):
        train_end = train_window + (i * step)
        test_end = min(train_end + step, total_games)

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()
        if len(test) == 0:
            break

        if apply_filter:
            train = _apply_garbage_filter(train)

        train = train.dropna(subset=feature_cols)
        test = test.dropna(subset=feature_cols)
        if len(train) < 50 or len(test) == 0:
            continue

        X_train = train[feature_cols]
        X_test = test[feature_cols]

        model = model_fn()
        model.fit(X_train, train[target])
        pred = model.predict(X_test)
        errors.extend(np.abs(pred - test[target].values))

    if len(errors) == 0:
        return np.nan

    return float(np.mean(errors))


def run_optimization(
    model_df,
    use_walk_forward=True,
    train_window=800,
    step=50,
    apply_filter=True,
    verbose=True,
):
    configs = _build_model_configs()
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    targets = {
        "HOME_PTS": ("Home", MODEL_A_PATH),
        "AWAY_PTS": ("Away", MODEL_B_PATH),
        "TOTAL_PTS": ("Total", MODEL_C_PATH),
    }

    if verbose:
        print("=" * 55)
        print("MODEL OPTIMIZATION")
        print("=" * 55)
        print(f"Games: {len(model_df)}, Features: {len(feature_cols)}")
        if use_walk_forward:
            print(f"Walk-forward: window={train_window}, step={step}")
        else:
            print("Evaluation: 80/20 time split")

    evaluator = _walk_forward_mae if use_walk_forward else _time_split_mae

    results = {}
    for name, model_fn in configs.items():
        maes = {}
        for target in targets:
            maes[target] = evaluator(
                model_df,
                model_fn,
                target,
                feature_cols,
                train_window=train_window,
                step=step,
                apply_filter=apply_filter,
            ) if use_walk_forward else evaluator(
                model_df,
                model_fn,
                target,
                feature_cols,
                apply_filter=apply_filter,
            )

        results[name] = maes
        if verbose:
            print(
                f"  {name:12s} | Home: {maes['HOME_PTS']:.2f} | "
                f"Away: {maes['AWAY_PTS']:.2f} | Total: {maes['TOTAL_PTS']:.2f}"
            )

    if verbose:
        print("\n" + "=" * 55)
        print("BEST MODELS:")

    best_models = {}
    for target, (label, path) in targets.items():
        best_name = min(results, key=lambda n: results[n][target])
        best_mae = results[best_name][target]
        if verbose:
            print(f"  {label:5s}: {best_name} (MAE={best_mae:.2f})")
        best_models[target] = (best_name, path)

    if verbose:
        print("\n[SAVING] Training best models on full data...")

    saved = {}
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df = _apply_garbage_filter(df) if apply_filter else df
    df = df.dropna(subset=feature_cols)
    X_full = df[feature_cols]

    for target, (best_name, path) in best_models.items():
        model = configs[best_name]()
        model.fit(X_full, df[target])
        with open(path, "wb") as f:
            pickle.dump(model, f)
        saved[target] = best_name

    return {
        "results": results,
        "best_models": best_models,
        "saved": saved,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    df = pd.read_csv(MODEL_READY_PATH)
    run_optimization(df, use_walk_forward=True)
