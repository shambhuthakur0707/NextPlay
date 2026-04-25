# -*- coding: utf-8 -*-
"""
NextPlay -- Historical Backtesting
====================================
Walk-forward backtesting to evaluate model on historical data.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import (
    RF_PARAMS, FEATURE_COLS_FINAL, BACKTEST_RESULTS_PATH,
    BLOWOUT_MARGIN_THRESHOLD, OT_TOTAL_THRESHOLD, OT_MARGIN_THRESHOLD,
)
from utils.helpers import win_probability, confidence_tier


def walk_forward_backtest(model_df, train_window=800, step=50,
                          apply_filter=True, verbose=True):
    """
    Walk-forward backtest: train on past N games, predict next batch.

    Returns:
        DataFrame with predictions, actuals, and errors
    """
    model_df = model_df.sort_values("GAME_DATE").reset_index(drop=True)
    feature_cols = [c for c in FEATURE_COLS_FINAL if c in model_df.columns]
    all_results = []
    total_games = len(model_df)
    n_batches = (total_games - train_window) // step

    if verbose:
        print(f"Walk-forward backtest: {total_games} games, "
              f"{n_batches} batches")

    for i in range(n_batches):
        train_end = train_window + (i * step)
        test_end = min(train_end + step, total_games)

        train = model_df.iloc[:train_end].copy()
        test = model_df.iloc[train_end:test_end].copy()
        if len(test) == 0:
            break

        # Garbage time filter
        if apply_filter and "PTS_MARGIN" in train.columns:
            clean = train[
                (train["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
                ~((train["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
                  (train["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD))
            ]
            if len(clean) > 100:
                train = clean

        train = train.dropna(subset=feature_cols)
        test = test.dropna(subset=feature_cols)
        if len(train) < 50 or len(test) == 0:
            continue

        X_tr, X_te = train[feature_cols], test[feature_cols]

        rf_h = RandomForestRegressor(**RF_PARAMS)
        rf_h.fit(X_tr, train["HOME_PTS"])
        rf_a = RandomForestRegressor(**RF_PARAMS)
        rf_a.fit(X_tr, train["AWAY_PTS"])
        rf_t = RandomForestRegressor(**RF_PARAMS)
        rf_t.fit(X_tr, train["TOTAL_PTS"])

        ph, pa, pt = rf_h.predict(X_te), rf_a.predict(X_te), rf_t.predict(X_te)

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

        all_results.append(batch)
        if verbose and (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{n_batches}: "
                  f"MAE={batch['abs_error_total'].mean():.1f}")

    results = pd.concat(all_results, ignore_index=True)
    if verbose:
        print(f"\n  BACKTEST: {len(results)} games, "
              f"MAE={results['abs_error_total'].mean():.2f}, "
              f"Win%={results['winner_correct'].mean():.1%}")
    return results


def analyze_backtest(results):
    """Analyze backtest results with breakdowns."""
    return {
        "total_games": len(results),
        "mae_total": results["abs_error_total"].mean(),
        "winner_accuracy": results["winner_correct"].mean(),
        "within_10": (results["abs_error_total"] <= 10).mean(),
        "within_15": (results["abs_error_total"] <= 15).mean(),
        "by_confidence": results.groupby("confidence").agg(
            games=("GAME_ID", "count"),
            mae=("abs_error_total", "mean"),
            win_acc=("winner_correct", "mean"),
        ).round(3),
    }


def save_backtest_results(results, path=BACKTEST_RESULTS_PATH):
    """Save backtest results to CSV."""
    results.to_csv(path, index=False)
    print(f"[OK] Backtest saved to {path}")


if __name__ == "__main__":
    from config import MODEL_READY_PATH
    df = pd.read_csv(MODEL_READY_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    results = walk_forward_backtest(df)
    save_backtest_results(results)
