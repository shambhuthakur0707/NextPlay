"""Fast total-model optimization with all available algorithms."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import MODEL_READY_PATH, FEATURE_COLS_FINAL, RF_PARAMS


def walk_forward_total_mae(model_df, feature_cols, model_fn, train_window=800, step=50):
    errors = []
    total_games = len(model_df)
    n_batches = (total_games - train_window) // step

    for i in range(n_batches):
        train_end = train_window + i * step
        test_end = min(train_end + step, total_games)

        train = model_df.iloc[:train_end].copy().dropna(subset=feature_cols)
        test = model_df.iloc[train_end:test_end].copy().dropna(subset=feature_cols)

        if len(train) < 50 or len(test) == 0:
            continue

        model = model_fn()
        model.fit(train[feature_cols], train["TOTAL_PTS"])
        pred = model.predict(test[feature_cols])

        errors.extend(np.abs(pred - test["TOTAL_PTS"].values))

    return float(np.mean(errors)) if errors else np.nan


def main():
    model_df = pd.read_csv(MODEL_READY_PATH)
    model_df["GAME_DATE"] = pd.to_datetime(model_df["GAME_DATE"])
    model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)

    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]

    print("=" * 60)
    print("FAST TOTAL OPTIMIZATION (WALK-FORWARD)")
    print("=" * 60)
    print(f"Games: {len(model_df)}, Features: {len(feature_cols)}")

    configs = {
        "RF-400-base": lambda: RandomForestRegressor(**RF_PARAMS),
        "RF-600-d12": lambda: RandomForestRegressor(
            n_estimators=600, max_depth=12, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        ),
        "RF-800-d15": lambda: RandomForestRegressor(
            n_estimators=800, max_depth=15, min_samples_leaf=8,
            random_state=42, n_jobs=-1,
        ),
    }

    # LightGBM
    try:
        import lightgbm as lgb
        configs["LGB-1000"] = lambda: lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.02, num_leaves=63,
            min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        )
    except ImportError:
        print("[WARN] LightGBM not available; skipping LGB-1000")

    # XGBoost
    try:
        from xgboost import XGBRegressor
        configs["XGB-800"] = lambda: XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        configs["XGB-1000"] = lambda: XGBRegressor(
            n_estimators=1000, learning_rate=0.02, max_depth=8,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        print("[WARN] XGBoost not available; skipping XGB configs")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        configs["CAT-800"] = lambda: CatBoostRegressor(
            iterations=800, learning_rate=0.03, depth=7,
            l2_leaf_reg=5, random_seed=42, verbose=0,
        )
    except ImportError:
        print("[WARN] CatBoost not available; skipping CAT configs")

    print(f"Candidates: {len(configs)} ({', '.join(configs.keys())})")
    print("-" * 60)

    results = {}
    for name, model_fn in configs.items():
        mae = walk_forward_total_mae(model_df, feature_cols, model_fn)
        results[name] = mae
        print(f"  {name:14s} | TOTAL MAE: {mae:.3f}")

    best_name = min(results, key=results.get)
    best_mae = results[best_name]

    print("-" * 60)
    print(f"BEST TOTAL MODEL: {best_name} (MAE={best_mae:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
