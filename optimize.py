"""Model optimization - try multiple approaches to minimize MAE.

Candidates: RandomForest, GradientBoosting, LightGBM, XGBoost, CatBoost.
After finding the best individual model per target, optionally builds a
VotingRegressor ensemble from the top-N performers.
"""
import warnings
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error

from config import (
    MODEL_READY_PATH, FEATURE_COLS_FINAL,
    STACKED_TOTAL_FEATURES,
    MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
    RF_PARAMS,
    VOTING_ENSEMBLE, VOTING_TOP_N,
)
from models.stacking import build_total_meta_features

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
# MODEL CANDIDATE CONFIGS
# ════════════════════════════════════════════════════════════

def _build_model_configs():
    """Build dict of model factory lambdas for benchmarking."""
    configs = {
        # ── RandomForest variants ──
        "RF-400": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=9, min_samples_leaf=15,
            random_state=42, n_jobs=-1),
        "RF-600-d12": lambda: RandomForestRegressor(
            n_estimators=600, max_depth=12, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
        "RF-800-d15": lambda: RandomForestRegressor(
            n_estimators=800, max_depth=15, min_samples_leaf=8,
            random_state=42, n_jobs=-1),
        # ── GradientBoosting variants ──
        "GB-500": lambda: GradientBoostingRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42),
        "GB-800": lambda: GradientBoostingRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            min_samples_leaf=10, subsample=0.8, random_state=42),
    }

    # ── LightGBM variants ──
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
        print("[WARN] LightGBM not available -- skipping LGB configs")

    # ── XGBoost variants ──
    try:
        from xgboost import XGBRegressor
        configs["XGB-500"] = lambda: XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
        configs["XGB-800"] = lambda: XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=0.8,
            random_state=42, n_jobs=-1, verbosity=0)
        configs["XGB-1000"] = lambda: XGBRegressor(
            n_estimators=1000, learning_rate=0.02, max_depth=8,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        print("[WARN] XGBoost not available -- skipping XGB configs")

    # ── CatBoost variants ──
    try:
        from catboost import CatBoostRegressor
        configs["CAT-500"] = lambda: CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0)
        configs["CAT-800"] = lambda: CatBoostRegressor(
            iterations=800, learning_rate=0.03, depth=7,
            l2_leaf_reg=5, random_seed=42, verbose=0)
    except ImportError:
        print("[WARN] CatBoost not available -- skipping CAT configs")

    return configs


# ════════════════════════════════════════════════════════════
# VOTING ENSEMBLE BUILDER
# ════════════════════════════════════════════════════════════

def _build_voting_ensemble(configs, results, target, top_n=VOTING_TOP_N):
    """Build a VotingRegressor from the top-N models for a given target.

    Args:
        configs: dict of name -> model factory lambda
        results: dict of name -> {target: mae}
        target: target column name (e.g. 'HOME_PTS')
        top_n: number of models to include in the ensemble

    Returns:
        (VotingRegressor factory lambda, list of model names used)
    """
    ranked = sorted(results.keys(), key=lambda n: results[n][target])
    top_names = ranked[:top_n]

    def _factory():
        estimators = [(name, configs[name]()) for name in top_names]
        return VotingRegressor(estimators=estimators, n_jobs=-1)

    return _factory, top_names


# ════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ════════════════════════════════════════════════════════════

def _apply_garbage_filter(train_df):
    if "PTS_MARGIN" not in train_df.columns:
        return train_df

    return train_df[
        (train_df["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
        ~((train_df["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
          (train_df["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD))
    ]


def _time_split_mae(model_df, model_fn, target, feature_cols, apply_filter=True,
                     **_kwargs):
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


# ════════════════════════════════════════════════════════════
# STACKED TOTAL EVALUATION
# ════════════════════════════════════════════════════════════

def _stacked_total_time_split_mae(model_df, home_model_fn, away_model_fn,
                                  meta_model_fn, feature_cols,
                                  apply_filter=True, **_kwargs):
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_idx = int(len(df) * 0.80)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    if apply_filter:
        train = _apply_garbage_filter(train)

    train = train.dropna(subset=feature_cols)
    test = test.dropna(subset=feature_cols)
    if len(train) < 50 or len(test) == 0:
        return np.nan

    home_model = home_model_fn()
    away_model = away_model_fn()
    home_model.fit(train[feature_cols], train["HOME_PTS"])
    away_model.fit(train[feature_cols], train["AWAY_PTS"])

    meta_cols = STACKED_TOTAL_FEATURES
    train_meta = build_total_meta_features(
        train[feature_cols],
        home_model.predict(train[feature_cols]),
        away_model.predict(train[feature_cols]),
        feature_cols=meta_cols,
    )
    test_meta = build_total_meta_features(
        test[feature_cols],
        home_model.predict(test[feature_cols]),
        away_model.predict(test[feature_cols]),
        feature_cols=meta_cols,
    )

    meta_model = meta_model_fn()
    meta_model.fit(train_meta, train["TOTAL_PTS"])
    pred = meta_model.predict(test_meta)
    return mean_absolute_error(test["TOTAL_PTS"], pred)


def _stacked_total_walk_forward_mae(model_df, home_model_fn, away_model_fn,
                                    meta_model_fn, feature_cols,
                                    train_window=800, step=50,
                                    apply_filter=True):
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

        home_model = home_model_fn()
        away_model = away_model_fn()
        home_model.fit(train[feature_cols], train["HOME_PTS"])
        away_model.fit(train[feature_cols], train["AWAY_PTS"])

        meta_cols = STACKED_TOTAL_FEATURES
        train_meta = build_total_meta_features(
            train[feature_cols],
            home_model.predict(train[feature_cols]),
            away_model.predict(train[feature_cols]),
            feature_cols=meta_cols,
        )
        test_meta = build_total_meta_features(
            test[feature_cols],
            home_model.predict(test[feature_cols]),
            away_model.predict(test[feature_cols]),
            feature_cols=meta_cols,
        )

        meta_model = meta_model_fn()
        meta_model.fit(train_meta, train["TOTAL_PTS"])
        pred = meta_model.predict(test_meta)
        errors.extend(np.abs(pred - test["TOTAL_PTS"].values))

    if len(errors) == 0:
        return np.nan

    return float(np.mean(errors))


# ════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION ENTRY POINT
# ════════════════════════════════════════════════════════════

def run_optimization(
    model_df,
    use_walk_forward=True,
    train_window=800,
    step=50,
    apply_filter=True,
    use_voting=None,
    verbose=True,
):
    """Run full model optimization across all candidate algorithms.

    Steps:
        1. Evaluate every candidate config for HOME_PTS and AWAY_PTS.
        2. Pick the best single model per target.
        3. Optionally build a VotingRegressor from the top-N models.
        4. Evaluate the stacked total model using the best base configs.
        5. Train final models on full data and save to disk.

    Args:
        model_df: feature-engineered DataFrame
        use_walk_forward: walk-forward evaluation (True) or 80/20 split (False)
        train_window: walk-forward training window
        step: walk-forward step size
        apply_filter: apply garbage-time filter
        use_voting: override config.VOTING_ENSEMBLE (None = use config)
        verbose: print progress

    Returns:
        dict with results, best_models, saved configs, mae_total
    """
    if use_voting is None:
        use_voting = VOTING_ENSEMBLE

    configs = _build_model_configs()
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    targets = {
        "HOME_PTS": ("Home", MODEL_A_PATH),
        "AWAY_PTS": ("Away", MODEL_B_PATH),
    }

    if verbose:
        print("=" * 60)
        print("MODEL OPTIMIZATION")
        print("=" * 60)
        print(f"Games: {len(model_df)}, Features: {len(feature_cols)}")
        print(f"Candidates: {len(configs)} ({', '.join(configs.keys())})")
        if use_walk_forward:
            print(f"Walk-forward: window={train_window}, step={step}")
        else:
            print("Evaluation: 80/20 time split")
        if use_voting:
            print(f"VotingRegressor: enabled (top-{VOTING_TOP_N})")

    evaluator = _walk_forward_mae if use_walk_forward else _time_split_mae

    # -- Phase 1: evaluate all individual models --
    if verbose:
        print(f"\n{'-' * 60}")
        print("PHASE 1 -- Individual model evaluation")
        print(f"{'-' * 60}")

    results = {}
    for name, model_fn in configs.items():
        maes = {}
        for target in targets:
            if use_walk_forward:
                maes[target] = evaluator(
                    model_df, model_fn, target, feature_cols,
                    train_window=train_window, step=step,
                    apply_filter=apply_filter,
                )
            else:
                maes[target] = evaluator(
                    model_df, model_fn, target, feature_cols,
                    apply_filter=apply_filter,
                )

        results[name] = maes
        if verbose:
            print(
                f"  {name:14s} | Home: {maes['HOME_PTS']:.2f} | "
                f"Away: {maes['AWAY_PTS']:.2f}"
            )

    # -- Phase 2: pick best single and optional voting ensemble --
    if verbose:
        print(f"\n{'-' * 60}")
        print("PHASE 2 -- Best model selection")
        print(f"{'-' * 60}")

    best_models = {}
    for target, (label, path) in targets.items():
        best_name = min(results, key=lambda n: results[n][target])
        best_mae = results[best_name][target]
        if verbose:
            print(f"  Best {label:5s}: {best_name} (MAE={best_mae:.2f})")
        best_models[target] = (best_name, path)

    # Voting ensemble evaluation
    voting_used = {}
    if use_voting and len(configs) >= VOTING_TOP_N:
        if verbose:
            print(f"\n{'-' * 60}")
            print(f"PHASE 2b -- VotingRegressor (top-{VOTING_TOP_N}) evaluation")
            print(f"{'-' * 60}")

        for target, (label, path) in targets.items():
            voting_fn, voting_names = _build_voting_ensemble(
                configs, results, target, top_n=VOTING_TOP_N)

            if use_walk_forward:
                voting_mae = evaluator(
                    model_df, voting_fn, target, feature_cols,
                    train_window=train_window, step=step,
                    apply_filter=apply_filter,
                )
            else:
                voting_mae = evaluator(
                    model_df, voting_fn, target, feature_cols,
                    apply_filter=apply_filter,
                )

            best_single_name, _ = best_models[target]
            best_single_mae = results[best_single_name][target]

            if verbose:
                print(f"  {label:5s} Voting ({'+'.join(voting_names)}): "
                      f"MAE={voting_mae:.2f}  "
                      f"(best single: {best_single_mae:.2f})")

            if voting_mae < best_single_mae:
                if verbose:
                    print(f"    >> VotingRegressor WINS for {label} "
                          f"(delta={best_single_mae - voting_mae:+.3f})")
                results[f"VOTE-{label}"] = {target: voting_mae}
                configs[f"VOTE-{label}"] = voting_fn
                best_models[target] = (f"VOTE-{label}", path)
                voting_used[target] = voting_names
            else:
                if verbose:
                    print(f"    >> Best single model wins for {label}")

    best_home_name, _ = best_models["HOME_PTS"]
    best_away_name, _ = best_models["AWAY_PTS"]

    # -- Phase 3: stacked total model --
    if verbose:
        print(f"\n{'-' * 60}")
        print("PHASE 3 -- Stacked total model evaluation")
        print(f"{'-' * 60}")

    # Use RF for the meta-model (stacked total) by default
    stacked_meta_fn = lambda: RandomForestRegressor(**RF_PARAMS)

    if use_walk_forward:
        total_mae = _stacked_total_walk_forward_mae(
            model_df,
            configs[best_home_name],
            configs[best_away_name],
            stacked_meta_fn,
            feature_cols,
            train_window=train_window,
            step=step,
            apply_filter=apply_filter,
        )
    else:
        total_mae = _stacked_total_time_split_mae(
            model_df,
            configs[best_home_name],
            configs[best_away_name],
            stacked_meta_fn,
            feature_cols,
            apply_filter=apply_filter,
        )

    if verbose:
        print(f"  Stacked Total MAE: {total_mae:.2f}")

    # -- Phase 4: train best models on full data and save --
    if verbose:
        print(f"\n{'-' * 60}")
        print("PHASE 4 -- Training final models on full data")
        print(f"{'-' * 60}")

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
        if verbose:
            label = "Home" if target == "HOME_PTS" else "Away"
            print(f"  [SAVED] {label}: {best_name} -> {path}")

    # Train stacked total with best base models
    home_model = configs[best_home_name]()
    away_model = configs[best_away_name]()
    home_model.fit(X_full, df["HOME_PTS"])
    away_model.fit(X_full, df["AWAY_PTS"])

    meta_cols = STACKED_TOTAL_FEATURES
    meta_full = build_total_meta_features(
        X_full,
        home_model.predict(X_full),
        away_model.predict(X_full),
        feature_cols=meta_cols,
    )
    total_model = stacked_meta_fn()
    total_model.fit(meta_full, df["TOTAL_PTS"])
    with open(MODEL_C_PATH, "wb") as f:
        pickle.dump(total_model, f)
    saved["TOTAL_PTS"] = "STACKED"
    if verbose:
        print(f"  [SAVED] Total: STACKED -> {MODEL_C_PATH}")

    # -- Summary --
    if verbose:
        print(f"\n{'=' * 60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        for target, name in saved.items():
            label = {"HOME_PTS": "Home (A)", "AWAY_PTS": "Away (B)",
                     "TOTAL_PTS": "Total (C)"}[target]
            print(f"  {label:12s} -> {name}")
        print(f"  Total MAE    : {total_mae:.2f}")
        if voting_used:
            print(f"  Voting used  : {voting_used}")
        print(f"{'=' * 60}")

    return {
        "results": results,
        "best_models": best_models,
        "saved": saved,
        "feature_cols": feature_cols,
        "mae_total": total_mae,
        "voting_used": voting_used,
    }


if __name__ == "__main__":
    df = pd.read_csv(MODEL_READY_PATH)
    run_optimization(df, use_walk_forward=True)
