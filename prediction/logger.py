# -*- coding: utf-8 -*-
"""
NextPlay -- Prediction Logger & Error Analysis
================================================
Logs every prediction, tracks errors, and analyzes patterns.
"""
import pandas as pd
from datetime import datetime
from prediction.predict import predict_game
from config import PREDICTION_LOG_PATH, MODEL_VERSION


def load_prediction_log():
    """Load existing prediction log or create empty one."""
    try:
        log = pd.read_csv(PREDICTION_LOG_PATH)
        print(f"[OK] Loaded prediction log -- {len(log)} entries")
        return log
    except FileNotFoundError:
        columns = [
            "PRED_DATE", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
            "PRED_HOME", "PRED_AWAY", "PRED_TOTAL",
            "ACTUAL_HOME", "ACTUAL_AWAY", "ACTUAL_TOTAL",
            "HOME_INJURED", "AWAY_INJURED",
            "ERROR_HOME", "ERROR_AWAY", "ERROR_TOTAL",
            "ABS_ERROR_HOME", "ABS_ERROR_AWAY", "ABS_ERROR_TOTAL",
            "CONFIDENCE", "MODEL_VERSION",
        ]
        print("[OK] New prediction log created")
        return pd.DataFrame(columns=columns)


def predict_and_log(home_team, away_team, model_df, models,
                    game_date=None, home_out=None, away_out=None,
                    actual_home=None, actual_away=None,
                    shot_df=None, player_impact_df=None,
                    feature_cols=None, verbose=True):
    """
    Make a prediction AND log it for tracking.

    If actual scores provided, calculates error immediately.
    """
    home_out = home_out or []
    away_out = away_out or []
    game_date = game_date or datetime.today().strftime("%Y-%m-%d")

    result = predict_game(
        home_team, away_team, model_df, models,
        shot_df=shot_df, player_impact_df=player_impact_df,
        feature_cols=feature_cols,
        home_out=home_out, away_out=away_out, verbose=verbose,
    )

    if result is None:
        return None

    # Calculate errors if actuals provided
    err_h = err_a = err_t = abs_h = abs_a = abs_t = None
    if actual_home is not None and actual_away is not None:
        actual_total = actual_home + actual_away
        err_h = result["pred_home"] - actual_home
        err_a = result["pred_away"] - actual_away
        err_t = result["pred_total"] - actual_total
        abs_h, abs_a, abs_t = abs(err_h), abs(err_a), abs(err_t)

        if verbose:
            print(f"\n  ACTUAL RESULT")
            print(f"  {home_team}: {actual_home} (error: {err_h:+.1f})")
            print(f"  {away_team}: {actual_away} (error: {err_a:+.1f})")
            print(f"  Total: {actual_total} (error: {err_t:+.1f})")

    new_row = {
        "PRED_DATE": datetime.today().strftime("%Y-%m-%d"),
        "GAME_DATE": game_date,
        "HOME_TEAM": home_team,
        "AWAY_TEAM": away_team,
        "PRED_HOME": result["pred_home"],
        "PRED_AWAY": result["pred_away"],
        "PRED_TOTAL": result["pred_total"],
        "ACTUAL_HOME": actual_home,
        "ACTUAL_AWAY": actual_away,
        "ACTUAL_TOTAL": (actual_home + actual_away) if actual_home else None,
        "HOME_INJURED": ",".join(home_out) if home_out else "",
        "AWAY_INJURED": ",".join(away_out) if away_out else "",
        "ERROR_HOME": err_h, "ERROR_AWAY": err_a, "ERROR_TOTAL": err_t,
        "ABS_ERROR_HOME": abs_h, "ABS_ERROR_AWAY": abs_a,
        "ABS_ERROR_TOTAL": abs_t,
        "CONFIDENCE": result["confidence"],
        "MODEL_VERSION": MODEL_VERSION,
    }

    return result, new_row


def analyze_errors(pred_log):
    """Analyze prediction log to find patterns in errors."""
    logged = pred_log.dropna(subset=["ABS_ERROR_TOTAL"]).copy()

    if len(logged) < 3:
        print("[WARN]  Need at least 3 logged games with actuals")
        return

    print(f"\n{'=' * 55}")
    print(f"ERROR ANALYSIS -- {len(logged)} games logged")
    print(f"{'=' * 55}")

    print(f"\nOverall performance:")
    print(f"  Avg total error  : {logged['ABS_ERROR_TOTAL'].mean():.1f} pts")
    print(f"  Avg home error   : {logged['ABS_ERROR_HOME'].mean():.1f} pts")
    print(f"  Within 10 pts    : "
          f"{(logged['ABS_ERROR_TOTAL'] <= 10).mean():.1%}")
    print(f"  Within 15 pts    : "
          f"{(logged['ABS_ERROR_TOTAL'] <= 15).mean():.1%}")

    print(f"\nBias check (positive = we over-predict):")
    print(f"  Home bias : {logged['ERROR_HOME'].mean():+.2f} pts")
    print(f"  Away bias : {logged['ERROR_AWAY'].mean():+.2f} pts")
    print(f"  Total bias: {logged['ERROR_TOTAL'].mean():+.2f} pts")
