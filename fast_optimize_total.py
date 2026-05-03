"""
Fast total-model optimization -- LightGBM only, lean walk-forward.
Designed to run in under 10 minutes on 8GB RAM.

Changes from original:
- Removed RF-600, RF-800, GB-500, GB-800 (slow, LGB beats them anyway)
- Kept one RF-400 as a baseline comparison only
- Reduced walk-forward window: 800 -> 400
- Increased step size: 50 -> 100 (fewer folds, same signal)
- LGB-500 added alongside LGB-1000 for comparison
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import MODEL_READY_PATH, FEATURE_COLS_FINAL, RF_PARAMS


def walk_forward_total_mae(model_df, feature_cols, model_fn,
                           train_window=400, step=100):
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
    print("FAST TOTAL OPTIMIZATION (LightGBM, lean walk-forward)")
    print("=" * 60)
    print(f"Games: {len(model_df)}, Features: {len(feature_cols)}")
    print(f"Walk-forward: window=400, step=100")
    print()

    # RF-400 kept only as a baseline so you can see the LGB improvement
    configs = {
        "RF-400 (baseline)": lambda: RandomForestRegressor(**RF_PARAMS),
    }

    try:
        import lightgbm as lgb

        configs["LGB-500"] = lambda: lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        configs["LGB-800"] = lambda: lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=47,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        configs["LGB-1000"] = lambda: lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )

    except ImportError:
        print("[WARN] LightGBM not available -- only RF-400 will run.")
        print("       Install with: pip install lightgbm --break-system-packages")

    print(f"Candidates: {list(configs.keys())}")
    print("-" * 60)

    results = {}
    for name, model_fn in configs.items():
        print(f"  Running {name}...", end=" ", flush=True)
        mae = walk_forward_total_mae(model_df, feature_cols, model_fn)
        results[name] = mae
        print(f"TOTAL MAE: {mae:.3f}")

    best_name = min(results, key=results.get)
    best_mae = results[best_name]

    print("-" * 60)
    print(f"BEST MODEL : {best_name}")
    print(f"BEST MAE   : {best_mae:.3f}")
    if "RF-400 (baseline)" in results:
        baseline = results["RF-400 (baseline)"]
        improvement = baseline - best_mae
        print(f"IMPROVEMENT: {improvement:+.3f} pts vs RF baseline")
    print("=" * 60)
    print()
    print("Next step: update config.py LGB_PARAMS with the winning")
    print("config and run: python -m pipelines.full_rebuild")


if __name__ == "__main__":
    main()