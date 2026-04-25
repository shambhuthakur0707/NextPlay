# -*- coding: utf-8 -*-
"""
NextPlay -- Game Prediction Engine
====================================
Injury-aware score prediction for any NBA matchup.
Uses V8 models with 100 features (including ELO ratings).
"""
import pandas as pd
import numpy as np
from utils.helpers import win_probability, confidence_tier, style_label


def predict_game(home_team, away_team, model_df, models, shot_df=None,
                 player_impact_df=None, feature_cols=None,
                 home_out=None, away_out=None, verbose=True):
    """
    Predict the score of any NBA matchup.

    Args:
        home_team: team abbreviation e.g. 'BOS'
        away_team: team abbreviation e.g. 'LAL'
        model_df: feature-engineered dataset
        models: dict with model_A, model_B, model_C
        shot_df: shot profile data (for style display)
        player_impact_df: player impact data (for injury adjustment)
        feature_cols: list of feature column names
        home_out: list of injured home players e.g. ['Tatum']
        away_out: list of injured away players e.g. ['LeBron']
        verbose: print full report

    Returns:
        dict with all prediction values
    """
    home_out = home_out or []
    away_out = away_out or []

    if feature_cols is None:
        feature_cols = list(models["model_C"].feature_names_in_)

    # Get most recent season
    latest_season = model_df["SEASON"].max()

    # Get most recent form for each team
    home_games = model_df[
        (model_df["HOME_TEAM"] == home_team) &
        (model_df["SEASON"] == latest_season)
    ].sort_values("GAME_DATE")

    away_games = model_df[
        (model_df["AWAY_TEAM"] == away_team) &
        (model_df["SEASON"] == latest_season)
    ].sort_values("GAME_DATE")

    # Fallback: check opposite appearances
    if len(home_games) == 0:
        home_games = model_df[
            (model_df["AWAY_TEAM"] == home_team) &
            (model_df["SEASON"] == latest_season)
        ].sort_values("GAME_DATE")

    if len(away_games) == 0:
        away_games = model_df[
            (model_df["HOME_TEAM"] == away_team) &
            (model_df["SEASON"] == latest_season)
        ].sort_values("GAME_DATE")

    if len(home_games) == 0 or len(away_games) == 0:
        print(f"[FAIL] No data for {home_team} or {away_team}")
        print(f"   Valid: {sorted(model_df['HOME_TEAM'].unique())}")
        return None

    home_row = home_games.iloc[-1]
    away_row = away_games.iloc[-1]

    # Build feature vector
    feature_dict = {}
    for col in feature_cols:
        if col.startswith("AWAY_") and col in away_row.index:
            feature_dict[col] = away_row[col]
        elif col in home_row.index:
            feature_dict[col] = home_row[col]
        else:
            feature_dict[col] = 0

    X_pred = pd.DataFrame([feature_dict])[feature_cols]

    # Base predictions
    pred_home = models["model_A"].predict(X_pred)[0]
    pred_away = models["model_B"].predict(X_pred)[0]
    pred_total = models["model_C"].predict(X_pred)[0]

    # Injury adjustments
    injury_log = []
    if player_impact_df is not None and (home_out or away_out):
        s_latest = player_impact_df[
            (player_impact_df["SEASON"] == latest_season) &
            (player_impact_df["GAMES_PLAYED"] >= 15)
        ].copy()

        for player_name in home_out:
            matches = s_latest[
                (s_latest["TEAM_ABBR"] == home_team) &
                (s_latest["PLAYER_NAME"].str.contains(
                    player_name, case=False, na=False))
            ]
            if len(matches) > 0:
                impact = matches.iloc[0]["PTS_IMPACT"]
                pred_home -= impact
                pred_total -= impact
                injury_log.append(
                    f"  {home_team} OUT: "
                    f"{matches.iloc[0]['PLAYER_NAME']} -> "
                    f"-{impact:.1f} pts")
            else:
                injury_log.append(
                    f"  [WARN]  {player_name} not found in "
                    f"{home_team} roster")

        for player_name in away_out:
            matches = s_latest[
                (s_latest["TEAM_ABBR"] == away_team) &
                (s_latest["PLAYER_NAME"].str.contains(
                    player_name, case=False, na=False))
            ]
            if len(matches) > 0:
                impact = matches.iloc[0]["PTS_IMPACT"]
                pred_away -= impact
                pred_total -= impact
                injury_log.append(
                    f"  {away_team} OUT: "
                    f"{matches.iloc[0]['PLAYER_NAME']} -> "
                    f"-{impact:.1f} pts")
            else:
                injury_log.append(
                    f"  [WARN]  {player_name} not found in "
                    f"{away_team} roster")

    # Win probability and confidence
    margin = pred_home - pred_away
    wp = win_probability(margin)
    conf = confidence_tier(margin)

    # Shot style info
    h_style = a_style = "N/A"
    h_3pt = h_paint = h_pps = a_3pt = a_paint = a_pps = 0
    if shot_df is not None:
        for team, prefix in [(home_team, "h"), (away_team, "a")]:
            row = shot_df[
                (shot_df["TEAM_ABBR"] == team) &
                (shot_df["SEASON"] == latest_season)
            ]
            if len(row) > 0:
                r = row.iloc[0]
                t, p, pps = r["THREEPT_RATE"], r["PAINT_RATE"], r["PTS_PER_SHOT"]
                lbl = style_label(t, p)
                if prefix == "h":
                    h_3pt, h_paint, h_pps, h_style = t, p, pps, lbl
                else:
                    a_3pt, a_paint, a_pps, a_style = t, p, pps, lbl

    # Momentum
    hm = home_row.get("HOME_MOMENTUM", 0)
    am = away_row.get("AWAY_MOMENTUM", 0)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home": round(pred_home, 1),
        "pred_away": round(pred_away, 1),
        "pred_total": round(pred_total, 1),
        "home_range": (round(pred_home - 3.5), round(pred_home + 3.5)),
        "away_range": (round(pred_away - 3.5), round(pred_away + 3.5)),
        "total_range": (round(pred_total - 4.5), round(pred_total + 4.5)),
        "win_prob": round(wp * 100, 1),
        "confidence": conf,
        "home_style": h_style,
        "away_style": a_style,
        "home_momentum": round(hm, 1),
        "away_momentum": round(am, 1),
        "injury_log": injury_log,
    }

    if verbose:
        inj_section = ""
        if injury_log:
            inj_section = ("\n  INJURY ADJUSTMENTS\n"
                           + "\n".join(injury_log))

        h_scored = home_row.get("HOME_ROLL10_PTS", 0)
        h_allowed = home_row.get("HOME_DEF_ROLL10", 0)
        a_scored = away_row.get("AWAY_ROLL10_PTS", 0)
        a_allowed = away_row.get("AWAY_DEF_ROLL10", 0)

        print(f"""
????????????????????????????????????????????????????????
  NBA AI SCORE PREDICTOR  |  v7
  {home_team} (home)  vs  {away_team} (away)
????????????????????????????????????????????????????????
  PREDICTED FINAL SCORE
  {home_team}: {pred_home:.0f} pts   range: {result['home_range'][0]}-{result['home_range'][1]}
  {away_team}: {pred_away:.0f} pts   range: {result['away_range'][0]}-{result['away_range'][1]}
  Total  : {pred_total:.0f} pts   range: {result['total_range'][0]}-{result['total_range'][1]}

  WIN PROBABILITY
  {home_team}: {wp*100:.1f}%   {away_team}: {100-wp*100:.1f}%
  Confidence: {conf}
{inj_section}
  TEAM FORM -- last 10 games
  {home_team}  scored: {h_scored:.1f}  allowed: {h_allowed:.1f}  momentum: {hm:+.1f}
  {away_team}  scored: {a_scored:.1f}  allowed: {a_allowed:.1f}  momentum: {am:+.1f}

  SHOT STYLE
  {home_team}: {h_style:<16} 3PT:{h_3pt:.1%} Paint:{h_paint:.1%} PPS:{h_pps:.3f}
  {away_team}: {a_style:<16} 3PT:{a_3pt:.1%} Paint:{a_paint:.1%} PPS:{a_pps:.3f}
????????????????????????????????????????????????????????""")

    return result
