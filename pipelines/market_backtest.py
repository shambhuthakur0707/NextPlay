# -*- coding: utf-8 -*-
"""
NextPlay -- Market Line Backtest
================================
Walk-forward evaluation against closing lines (spread/total).
This is a diagnostic tool, not betting advice.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import (
    MODEL_READY_PATH,
    MARKET_LINES_PATH,
    FEATURE_COLS_FINAL,
    RF_PARAMS,
    BLOWOUT_MARGIN_THRESHOLD,
    OT_TOTAL_THRESHOLD,
    OT_MARGIN_THRESHOLD,
)


def _apply_garbage_filter(train_df):
    if "PTS_MARGIN" not in train_df.columns:
        return train_df

    return train_df[
        (train_df["PTS_MARGIN"].abs() <= BLOWOUT_MARGIN_THRESHOLD) &
        ~((train_df["TOTAL_PTS"] > OT_TOTAL_THRESHOLD) &
          (train_df["PTS_MARGIN"].abs() < OT_MARGIN_THRESHOLD))
    ]


def _normalize_lines(lines_df):
    df = lines_df.copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    rename_map = {
        "HOME_SPREAD": "HOME_LINE",
        "SPREAD_HOME": "HOME_LINE",
        "CLOSE_SPREAD": "HOME_LINE",
        "TOTAL": "TOTAL_LINE",
        "CLOSE_TOTAL": "TOTAL_LINE",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = {"HOME_TEAM", "AWAY_TEAM", "TOTAL_LINE", "HOME_LINE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing line columns: {sorted(missing)}")

    return df


def walk_forward_market_backtest(
    model_df,
    lines_df,
    train_window=800,
    step=50,
    apply_filter=True,
):
    """
    Walk-forward evaluation of model predictions vs market lines.

    Returns a DataFrame with predictions, lines, and errors.
    """
    df = model_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    lines_df = _normalize_lines(lines_df)

    feature_cols = [c for c in FEATURE_COLS_FINAL if c in df.columns]
    total_games = len(df)
    n_batches = (total_games - train_window) // step
    all_results = []

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

        X_tr = train[feature_cols]
        X_te = test[feature_cols]

        rf_h = RandomForestRegressor(**RF_PARAMS)
        rf_a = RandomForestRegressor(**RF_PARAMS)
        rf_t = RandomForestRegressor(**RF_PARAMS)

        rf_h.fit(X_tr, train["HOME_PTS"])
        rf_a.fit(X_tr, train["AWAY_PTS"])
        rf_t.fit(X_tr, train["TOTAL_PTS"])

        pred_home = rf_h.predict(X_te)
        pred_away = rf_a.predict(X_te)
        pred_total = rf_t.predict(X_te)

        batch = test[[
            "GAME_ID", "GAME_DATE", "SEASON",
            "HOME_TEAM", "AWAY_TEAM",
            "HOME_PTS", "AWAY_PTS", "TOTAL_PTS",
        ]].copy()

        batch["PRED_HOME"] = pred_home
        batch["PRED_AWAY"] = pred_away
        batch["PRED_TOTAL"] = pred_total
        batch["PRED_MARGIN"] = pred_home - pred_away
        batch["ACTUAL_MARGIN"] = batch["HOME_PTS"] - batch["AWAY_PTS"]

        batch = batch.merge(
            lines_df[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_LINE", "TOTAL_LINE"]],
            on=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"],
            how="left",
        )

        batch["TOTAL_EDGE"] = batch["PRED_TOTAL"] - batch["TOTAL_LINE"]
        batch["SPREAD_EDGE"] = batch["PRED_MARGIN"] - batch["HOME_LINE"]
        batch["ABS_TOTAL_ERR"] = (batch["PRED_TOTAL"] - batch["TOTAL_PTS"]).abs()

        batch["OVER_HIT"] = (
            (batch["PRED_TOTAL"] > batch["TOTAL_LINE"]) &
            (batch["TOTAL_PTS"] > batch["TOTAL_LINE"])
        ) | (
            (batch["PRED_TOTAL"] < batch["TOTAL_LINE"]) &
            (batch["TOTAL_PTS"] < batch["TOTAL_LINE"])
        )

        batch["ATS_HIT"] = (
            (batch["PRED_MARGIN"] > batch["HOME_LINE"]) &
            (batch["ACTUAL_MARGIN"] > batch["HOME_LINE"])
        ) | (
            (batch["PRED_MARGIN"] < batch["HOME_LINE"]) &
            (batch["ACTUAL_MARGIN"] < batch["HOME_LINE"])
        )

        all_results.append(batch)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def summarize_market_backtest(results):
    """Aggregate summary metrics for market backtest."""
    if len(results) == 0:
        return {}

    summary = {
        "games": int(len(results)),
        "line_coverage": float(results[["HOME_LINE", "TOTAL_LINE"]].notna().all(axis=1).mean()),
        "mae_total": float(results["ABS_TOTAL_ERR"].mean()),
        "over_hit_rate": float(results["OVER_HIT"].mean()),
        "ats_hit_rate": float(results["ATS_HIT"].mean()),
        "avg_total_edge": float(results["TOTAL_EDGE"].mean()),
        "avg_spread_edge": float(results["SPREAD_EDGE"].mean()),
    }

    return summary


def run_market_backtest(
    model_path=MODEL_READY_PATH,
    lines_path=MARKET_LINES_PATH,
    train_window=800,
    step=50,
    apply_filter=True,
    verbose=True,
):
    model_df = pd.read_csv(model_path)
    lines_df = pd.read_csv(lines_path)

    results = walk_forward_market_backtest(
        model_df,
        lines_df,
        train_window=train_window,
        step=step,
        apply_filter=apply_filter,
    )

    summary = summarize_market_backtest(results)

    if verbose:
        print("=" * 60)
        print("MARKET BACKTEST SUMMARY")
        print("=" * 60)
        if summary:
            print(f"Games evaluated: {summary['games']}")
            print(f"Line coverage : {summary['line_coverage']:.1%}")
            print(f"MAE (total)   : {summary['mae_total']:.2f}")
            print(f"Over hit rate : {summary['over_hit_rate']:.1%}")
            print(f"ATS hit rate  : {summary['ats_hit_rate']:.1%}")
            print(f"Avg total edge: {summary['avg_total_edge']:+.2f}")
            print(f"Avg spread edge: {summary['avg_spread_edge']:+.2f}")
        else:
            print("No results to summarize.")
        print("=" * 60)

    return results, summary


if __name__ == "__main__":
    run_market_backtest()
