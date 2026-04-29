# -*- coding: utf-8 -*-
"""
NextPlay -- Game Prediction Engine
====================================
Injury-aware score prediction for any NBA matchup.
Uses V8 models with 100 features (including ELO ratings).
"""
import pandas as pd
import numpy as np

from config import MARKET_TOTAL_BLEND_WEIGHT
from prediction.feature_builder import build_game_features
from models.stacking import build_total_meta_features
from utils.helpers import win_probability, confidence_tier, style_label


def _blend_total_with_market(model_total, market_total_line=None, weight=MARKET_TOTAL_BLEND_WEIGHT):
    """Blend the raw model total with the sportsbook total when one is provided."""
    if market_total_line is None or pd.isna(market_total_line):
        return model_total

    market_weight = float(weight)
    model_weight = 1.0 - market_weight
    return (model_weight * model_total) + (market_weight * float(market_total_line))


def _infer_latest_season(*frames):
    """Find the most recent season label present in the supplied frames."""
    for frame in frames:
        if frame is None or "SEASON" not in getattr(frame, "columns", []):
            continue
        seasons = frame["SEASON"].dropna()
        if len(seasons) > 0:
            return seasons.astype(str).max()
    return None


def _find_player_impact_matches(player_impact_df, player_name, team_abbr=None, latest_season=None):
    """Return likely player-impact matches for a typed injury name."""
    if player_impact_df is None or len(player_impact_df) == 0:
        return pd.DataFrame()

    frame = player_impact_df.copy()
    if latest_season is not None and "SEASON" in frame.columns:
        frame = frame[frame["SEASON"].astype(str) == str(latest_season)]

    if "PLAYER_NAME" not in frame.columns:
        return pd.DataFrame()

    name_mask = frame["PLAYER_NAME"].astype(str).str.contains(player_name, case=False, na=False)
    matches = frame[name_mask].copy()

    if team_abbr is not None and "TEAM_ABBR" in matches.columns:
        return matches[matches["TEAM_ABBR"] == team_abbr].copy()

    return matches


def predict_game(home_team, away_team, model_df, models, shot_df=None,
                 player_impact_df=None, feature_cols=None,
                 home_out=None, away_out=None, raw_gamelogs=None,
                 game_date=None, market_total_line=None, verbose=True,
                 series_game_num=None, home_series_wins=None,
                 away_series_wins=None):
    """
    Predict the score of any NBA matchup.

    Args:
        home_team: team abbreviation e.g. 'BOS'
        away_team: team abbreviation e.g. 'LAL'
        model_df: feature-engineered dataset
        models: dict with model_A, model_B, model_C
        shot_df: shot profile data (for style display)
        player_impact_df: player impact data (for injury adjustment)
        feature_cols: list of feature column names (base feature fallback)
        home_out: list of injured home players e.g. ['Tatum']
        away_out: list of injured away players e.g. ['LeBron']
        raw_gamelogs: optional raw team-level gamelogs for fresh snapshots
        game_date: optional matchup date for the feature builder
        market_total_line: optional sportsbook total to calibrate the final total
        verbose: print full report
        series_game_num: game number within a playoff series (1-7)
        home_series_wins: home team's series wins entering this game (0-3)
        away_series_wins: away team's series wins entering this game (0-3)

    Returns:
        dict with all prediction values
    """
    home_out = home_out or []
    away_out = away_out or []

    base_feature_cols = list(getattr(models["model_A"], "feature_names_in_", []))
    if not base_feature_cols:
        base_feature_cols = feature_cols or []

    meta_feature_cols = list(getattr(models["model_C"], "feature_names_in_", []))
    if not meta_feature_cols:
        meta_feature_cols = feature_cols or []

    latest_season = _infer_latest_season(model_df, shot_df, player_impact_df)

    feature_frame = build_game_features(
        home_team,
        away_team,
        model_df=model_df,
        raw_gamelogs=raw_gamelogs,
        game_date=game_date,
        feature_cols=base_feature_cols,
        series_game_num=series_game_num,
        home_series_wins=home_series_wins,
        away_series_wins=away_series_wins,
    )

    if feature_frame is None or len(feature_frame) == 0:
        print(f"[FAIL] No data for {home_team} or {away_team}")
        print(f"   Valid: {sorted(model_df['HOME_TEAM'].unique())}")
        return None

    feature_row = feature_frame.iloc[0]
    X_pred = feature_frame[base_feature_cols]

    # Base predictions
    pred_home = models["model_A"].predict(X_pred)[0]
    pred_away = models["model_B"].predict(X_pred)[0]

    meta_X = build_total_meta_features(
        feature_frame,
        [pred_home],
        [pred_away],
        feature_cols=meta_feature_cols,
    )
    raw_pred_total = models["model_C"].predict(meta_X)[0]
    pred_total = _blend_total_with_market(raw_pred_total, market_total_line)

    # Injury adjustments
    injury_log = []
    if player_impact_df is not None and (home_out or away_out) and latest_season is not None:
        s_latest = player_impact_df[
            (player_impact_df["SEASON"] == latest_season) &
            (player_impact_df["GAMES_PLAYED"] >= 15)
        ].copy()

        for player_name in home_out:
            matches = _find_player_impact_matches(s_latest, player_name, home_team, latest_season)
            if len(matches) > 0:
                impact = matches.iloc[0]["PTS_IMPACT"]
                pred_home -= impact
                injury_log.append(
                    f"  {home_team} OUT: "
                    f"{matches.iloc[0]['PLAYER_NAME']} -> "
                    f"-{impact:.1f} pts")
            else:
                any_matches = _find_player_impact_matches(player_impact_df, player_name, latest_season=latest_season)
                if len(any_matches) > 0 and "TEAM_ABBR" in any_matches.columns:
                    teams = ", ".join(sorted(any_matches["TEAM_ABBR"].astype(str).unique().tolist()))
                    injury_log.append(
                        f"  [WARN]  {player_name} not on {home_team}; found on: {teams}")
                else:
                    injury_log.append(
                        f"  [WARN]  {player_name} not found in player-impact data")

        for player_name in away_out:
            matches = _find_player_impact_matches(s_latest, player_name, away_team, latest_season)
            if len(matches) > 0:
                impact = matches.iloc[0]["PTS_IMPACT"]
                pred_away -= impact
                injury_log.append(
                    f"  {away_team} OUT: "
                    f"{matches.iloc[0]['PLAYER_NAME']} -> "
                    f"-{impact:.1f} pts")
            else:
                any_matches = _find_player_impact_matches(player_impact_df, player_name, latest_season=latest_season)
                if len(any_matches) > 0 and "TEAM_ABBR" in any_matches.columns:
                    teams = ", ".join(sorted(any_matches["TEAM_ABBR"].astype(str).unique().tolist()))
                    injury_log.append(
                        f"  [WARN]  {player_name} not on {away_team}; found on: {teams}")
                else:
                    injury_log.append(
                        f"  [WARN]  {player_name} not found in player-impact data")

        meta_X = build_total_meta_features(
            feature_frame,
            [pred_home],
            [pred_away],
            feature_cols=meta_feature_cols,
        )
        raw_pred_total = models["model_C"].predict(meta_X)[0]
        pred_total = _blend_total_with_market(raw_pred_total, market_total_line)

    # Win probability and confidence
    margin = pred_home - pred_away
    wp = win_probability(margin)
    conf = confidence_tier(margin)

    # Shot style info
    h_style = a_style = "N/A"
    h_3pt = h_paint = h_pps = a_3pt = a_paint = a_pps = 0
    if shot_df is not None and latest_season is not None:
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
    hm = feature_row.get("HOME_MOMENTUM", 0)
    am = feature_row.get("AWAY_MOMENTUM", 0)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home": round(pred_home, 1),
        "pred_away": round(pred_away, 1),
        "pred_total_raw": round(raw_pred_total, 1),
        "pred_total": round(pred_total, 1),
        "market_total_line": None if market_total_line is None else round(float(market_total_line), 1),
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

        h_scored = feature_row.get("HOME_ROLL10_PTS", 0)
        h_allowed = feature_row.get("HOME_DEF_ROLL10", 0)
        a_scored = feature_row.get("AWAY_ROLL10_PTS", 0)
        a_allowed = feature_row.get("AWAY_DEF_ROLL10", 0)

        print(f"""
????????????????????????????????????????????????????????
  NBA AI SCORE PREDICTOR  |  v7
  {home_team} (home)  vs  {away_team} (away)
????????????????????????????????????????????????????????
  PREDICTED FINAL SCORE
  {home_team}: {pred_home:.0f} pts   range: {result['home_range'][0]}-{result['home_range'][1]}
  {away_team}: {pred_away:.0f} pts   range: {result['away_range'][0]}-{result['away_range'][1]}
    Total  : {pred_total:.0f} pts   range: {result['total_range'][0]}-{result['total_range'][1]}
    Raw total model: {raw_pred_total:.1f}{f' | Market total: {float(market_total_line):.1f}' if market_total_line is not None and not pd.isna(market_total_line) else ''}

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
