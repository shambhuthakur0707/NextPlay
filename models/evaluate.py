# -*- coding: utf-8 -*-
"""
NextPlay -- Model Evaluation
=============================
MAE evaluation, error analysis, and model comparison tools.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluate a single model against test data.

    Args:
        model: trained sklearn model
        X_test: test features DataFrame
        y_test: test target Series
        name: model name for display

    Returns:
        dict with evaluation metrics
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    errors = preds - y_test.values
    within_5 = (np.abs(errors) <= 5).mean()
    within_10 = (np.abs(errors) <= 10).mean()
    within_15 = (np.abs(errors) <= 15).mean()
    bias = errors.mean()

    result = {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_5": within_5,
        "within_10": within_10,
        "within_15": within_15,
        "bias": bias,
        "n_test": len(y_test),
    }

    return result


def evaluate_all_models(models, X_test, test_df):
    """
    Evaluate all three models (A/B/C) together.

    Args:
        models: dict with model_A, model_B, model_C
        X_test: test features DataFrame
        test_df: test DataFrame (for target columns)

    Returns:
        dict with results for each model
    """
    results = {}

    results["home"] = evaluate_model(
        models["model_A"], X_test, test_df["HOME_PTS"], "Home Score (A)"
    )
    results["away"] = evaluate_model(
        models["model_B"], X_test, test_df["AWAY_PTS"], "Away Score (B)"
    )
    results["total"] = evaluate_model(
        models["model_C"], X_test, test_df["TOTAL_PTS"], "Total Points (C)"
    )

    return results


def print_comparison_table(results):
    """
    Print a formatted comparison table for model results.

    Args:
        results: dict from evaluate_all_models()
    """
    print(f"\n{'?' * 60}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'?' * 60}")
    print(f"  {'Metric':<20} {'Home (A)':>10} {'Away (B)':>10} {'Total (C)':>10}")
    print(f"  {'?' * 50}")

    for metric, label, fmt in [
        ("mae", "MAE", ".2f"),
        ("rmse", "RMSE", ".2f"),
        ("r2", "R²", ".3f"),
        ("within_5", "Within 5 pts", ".1%"),
        ("within_10", "Within 10 pts", ".1%"),
        ("within_15", "Within 15 pts", ".1%"),
        ("bias", "Bias", "+.2f"),
    ]:
        home_val = format(results["home"][metric], fmt)
        away_val = format(results["away"][metric], fmt)
        total_val = format(results["total"][metric], fmt)
        print(f"  {label:<20} {home_val:>10} {away_val:>10} {total_val:>10}")

    print(f"  {'?' * 50}")
    print(f"  Test games: {results['total']['n_test']}")
    print(f"{'?' * 60}")


def error_breakdown_by_team(model, X_test, y_test, test_df, team_col="HOME_TEAM"):
    """
    Break down MAE by team to find which teams are hardest to predict.

    Args:
        model: trained model
        X_test: test features
        y_test: test target
        test_df: test DataFrame with team columns
        team_col: column with team abbreviation

    Returns:
        DataFrame with per-team MAE sorted worst to best
    """
    preds = model.predict(X_test)
    errors = np.abs(preds - y_test.values)

    breakdown = pd.DataFrame({
        "team": test_df[team_col].values,
        "abs_error": errors,
    })

    team_mae = breakdown.groupby("team")["abs_error"].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "MAE", "count": "Games"}).sort_values(
        "MAE", ascending=False
    ).round(2)

    return team_mae


def error_breakdown_by_total_range(model_C, X_test, y_test):
    """
    Break down total points model error by score range.
    Helps identify if model struggles with high/low scoring games.

    Returns:
        DataFrame with per-range MAE
    """
    preds = model_C.predict(X_test)
    actuals = y_test.values
    errors = np.abs(preds - actuals)

    ranges = pd.cut(actuals, bins=[160, 190, 210, 230, 250, 300],
                    labels=["160-190", "190-210", "210-230", "230-250", "250+"])

    breakdown = pd.DataFrame({
        "range": ranges,
        "abs_error": errors,
    })

    return breakdown.groupby("range")["abs_error"].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "MAE", "count": "Games"}).round(2)
