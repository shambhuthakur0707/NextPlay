# -*- coding: utf-8 -*-
"""
NextPlay -- Historical Backtesting (OPTIMIZED)
==============================================
Walk-forward backtesting with sliding window, LightGBM, and progress logging.
"""
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Use LightGBM instead of RandomForest (10-50x faster)
import lightgbm as lgb

from config import (
    FEATURE_COLS_FINAL, BACKTEST_RESULTS_PATH,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
)
from utils.helpers import confidence_tier

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback if tqdm not installed


# Lightweight, fast LGBM params for backtesting
LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    n_jobs=-1,
    verbose=-1,
    objective="regression",
)


def walk_forward_backtest(
    model_df,
    train_window=800,        # SLIDING window size (not expanding)
    step=50,
    retrain_every=5,         # Retrain only every N batches (5x speedup)
    apply_filter=True,
    verbose=True,
    use_sliding_window=True, # NEW: True = O(N), False = O(N²) old behavior
):
    """
    Walk-forward backtest with sliding window + LightGBM.

    Speedups vs original:
      - LightGBM instead of RandomForest:  ~20x
      - Sliding window (capped train size): ~5x on later batches
      - Retrain every 5 batches:           ~5x
      - Pre-filter once, not per batch:    minor
      ----------------------------------------------
      Combined: ~100-300x faster (4 hrs -> ~2-5 min)
    """
    t0 = time.time()
    model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]

    if verbose:
        print(f"[Backtest] Using {len(feature_cols)}/{len(FEATURE_COLS_FINAL)} features")

    # Pre-apply garbage-time filter ONCE (was running every batch)
    if apply_filter and "PTS_MARGIN" in model_df.columns:
        before = len(model_df)
        clean_mask = (
            (model_df["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
            ~((model_df["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
              (model_df["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD))
        )
        # Keep filtered version for TRAIN; full version for TEST
        train_pool = model_df[clean_mask].reset_index(drop=True)
        if verbose:
            print(f"[Backtest] Filtered {before} -> {len(train_pool)} train pool "
                  f"(removed {before - len(train_pool)} blowouts/OT)")
    else:
        train_pool = model_df.copy()

    total_games = len(model_df)
    n_batches = max(0, (total_games - train_window) // step)
    if verbose:
        print(f"[Backtest] {total_games} games, {n_batches} batches, "
              f"window={train_window}, step={step}, retrain_every={retrain_every}")

    all_results = []
    rf_h = rf_a = rf_t = None  # cached models
    last_train_end = -1

    iterator = range(n_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Backtest", unit="batch")

    for i in iterator:
        train_end = train_window + (i * step)
        test_end = min(train_end + step, total_games)

        # SLIDING WINDOW: only last `train_window` games
        if use_sliding_window:
            train_start = max(0, train_end - train_window)
        else:
            train_start = 0  # expanding (old behavior)

        # Apply same date cutoff to filtered train pool
        cutoff_date = model_df.iloc[train_end - 1]["GAME_DATE"]
        start_date = model_df.iloc[train_start]["GAME_DATE"]
        train = train_pool[
            (train_pool["GAME_DATE"] >= start_date) &
            (train_pool["GAME_DATE"] <= cutoff_date)
        ]
        test = model_df.iloc[train_end:test_end]

        if len(test) == 0:
            break

        train = train.dropna(subset=feature_cols)
        test = test.dropna(subset=feature_cols)
        if len(train) < 50 or len(test) == 0:
            continue

        X_tr = train[feature_cols].values
        X_te = test[feature_cols].values

        # RETRAIN CADENCE: only refit every N batches
        need_retrain = (rf_h is None) or ((i - last_train_end) >= retrain_every)
        if need_retrain:
            rf_h = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_tr, train["HOME_PTS"])
            rf_a = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_tr, train["AWAY_PTS"])
            rf_t = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_tr, train["TOTAL_PTS"])
            last_train_end = i

        ph = rf_h.predict(X_te)
        pa = rf_a.predict(X_te)
        pt = rf_t.predict(X_te)

        batch = test[["GAME_ID", "GAME_DATE", "SEASON",
                      "HOME_TEAM", "AWAY_TEAM",
                      "HOME_PTS", "AWAY_PTS", "TOTAL_PTS"]].copy()
        batch["pred_home"] = ph
        batch["pred_away"] = pa
        batch["pred_total"] = pt
        batch["abs_error_total"] = np.abs(pt - batch["TOTAL_PTS"].values)
        batch["pred_winner"] = np.where(ph > pa, "HOME", "AWAY")
        batch["actual_winner"] = np.where(
            batch["HOME_PTS"] > batch["AWAY_PTS"], "HOME", "AWAY")
        batch["winner_correct"] = batch["pred_winner"] == batch["actual_winner"]
        batch["confidence"] = [confidence_tier(m) for m in (ph - pa)]
        batch["batch"] = i
        batch["was_retrained"] = need_retrain

        all_results.append(batch)

    results = pd.concat(all_results, ignore_index=True)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n[Backtest] DONE in {elapsed/60:.1f} min")
        print(f"  Games: {len(results)}")
        print(f"  MAE total: {results['abs_error_total'].mean():.2f}")
        print(f"  Winner acc: {results['winner_correct'].mean():.1%}")

    return results