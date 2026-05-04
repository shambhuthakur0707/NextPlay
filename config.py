# -*- coding: utf-8 -*-
"""
NextPlay -- Configuration
========================
Central configuration for all constants, paths, and feature definitions.
Uses the V9 model feature set and stacked total model metadata.
"""

import os

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data files
GAMELOGS_RAW_PATH       = os.path.join(DATA_DIR, "gamelogs_raw.csv")
GAMELOGS_ALL_PATH       = os.path.join(DATA_DIR, "gamelogs_all.csv")
GAMES_ALL_PATH          = os.path.join(DATA_DIR, "games_all.csv")
GAMES_SOS_PATH          = os.path.join(DATA_DIR, "games_sos.csv")
MODEL_READY_PATH        = os.path.join(DATA_DIR, "model_ready_final.csv")
SHOT_PROFILES_PATH      = os.path.join(DATA_DIR, "shot_profiles.csv")
PLAYER_IMPACT_PATH      = os.path.join(DATA_DIR, "player_impact_true.csv")
PREDICTION_LOG_PATH     = os.path.join(DATA_DIR, "prediction_log.csv")
BACKTEST_RESULTS_PATH   = os.path.join(DATA_DIR, "backtest_results.csv")
ELO_RATINGS_PATH        = os.path.join(DATA_DIR, "elo_ratings.csv")
MARKET_LINES_PATH       = os.path.join(DATA_DIR, "market_lines.csv")
ODDS_API_KEY            = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL       = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
ODDS_SPORT_KEY          = os.getenv("ODDS_SPORT_KEY", "basketball_nba")

# Model files
MODEL_A_PATH = os.path.join(DATA_DIR, "model_A_home.pkl")
MODEL_B_PATH = os.path.join(DATA_DIR, "model_B_away.pkl")
MODEL_C_PATH = os.path.join(DATA_DIR, "model_C_total.pkl")

# Playoff model files
MODEL_A_PLAYOFF_PATH = os.path.join(DATA_DIR, "model_A_playoff_home.pkl")
MODEL_B_PLAYOFF_PATH = os.path.join(DATA_DIR, "model_B_playoff_away.pkl")
MODEL_C_PLAYOFF_PATH = os.path.join(DATA_DIR, "model_C_playoff_total.pkl")

# Playoff season month range (April-June)
PLAYOFF_SEASON_START_MONTH = 4
PLAYOFF_SEASON_END_MONTH = 6


# ─────────────────────────────────────────────────────────────
# NBA TEAMS
# ─────────────────────────────────────────────────────────────

NBA_TEAMS = sorted([
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
])

SEASONS = ["2023-24", "2024-25", "2025-26"]

# Whether to include playoff games when pulling data / rebuilding models
INCLUDE_PLAYOFFS = True

# Weight to apply to playoff rows during training (0 < weight <= 1).
WEIGHT_PLAYOFF_ROWS = 0.3

# Use separate playoff models during playoff season (April-June)
USE_PLAYOFF_MODELS = True

# Playoff model training blend ratio
PLAYOFF_BLEND_RATIO = 0.7

# Upweight factor for playoff rows in blended training
PLAYOFF_UPWEIGHT_FACTOR = 3.0


# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS -- V9 FINAL
# ─────────────────────────────────────────────────────────────

# Rolling form -- home team
ROLLING_HOME_FEATURES = [
    "HOME_ROLL10_PTS", "HOME_ROLL10_FG_PCT", "HOME_ROLL10_FG3_PCT",
    "HOME_ROLL10_FT_PCT", "HOME_ROLL10_FG3A", "HOME_ROLL10_FTA",
    "HOME_ROLL10_AST", "HOME_ROLL10_TOV", "HOME_ROLL10_OREB",
    "HOME_ROLL10_REB", "HOME_ROLL10_PF", "HOME_ROLL10_TOTAL_PTS",
    "HOME_ROLL10_POSS_EST",
    "HOME_EFG_ROLL10", "HOME_TS_ROLL10",
    "HOME_ROLL5_PTS", "HOME_ROLL15_PTS", "HOME_FORM_TREND",
]

# Rolling form -- away team
ROLLING_AWAY_FEATURES = [
    "AWAY_ROLL10_PTS", "AWAY_ROLL10_FG_PCT", "AWAY_ROLL10_FG3_PCT",
    "AWAY_ROLL10_FT_PCT", "AWAY_ROLL10_FG3A", "AWAY_ROLL10_FTA",
    "AWAY_ROLL10_AST", "AWAY_ROLL10_TOV", "AWAY_ROLL10_OREB",
    "AWAY_ROLL10_REB", "AWAY_ROLL10_PF",
    "AWAY_ROLL10_POSS_EST",
    "AWAY_EFG_ROLL10", "AWAY_TS_ROLL10",
    "AWAY_ROLL5_PTS", "AWAY_ROLL15_PTS", "AWAY_FORM_TREND",
]

# Combined rolling signals
COMBINED_FEATURES = [
    "COMBINED_PTS_ROLL10", "COMBINED_FG3A_ROLL10", "COMBINED_FTA_ROLL10",
    "COMBINED_POSS_ROLL10", "TOV_DIFF_ROLL10", "OREB_DIFF_ROLL10",
]

# Matchup-specific history
MATCHUP_FEATURES = [
    "MATCHUP_AVG_TOTAL", "MATCHUP_HOME_WIN", "MATCHUP_MEETINGS",
]

# Rest + momentum
# FIX: Removed HOME_TRAVEL_DIST -- it is hardcoded to 0.0 in rest_streak.py
#      (home team doesn't travel), making it a constant useless feature.
REST_STREAK_FEATURES = [
    "HOME_REST_DAYS", "AWAY_REST_DAYS", "REST_DAYS_DIFF",
    "AWAY_TRAVEL_DIST", "TRAVEL_DIFF",
    "HOME_STREAK", "AWAY_STREAK", "STREAK_DIFF",
    "HOME_B2B", "AWAY_B2B", "B2B_DIFF", "B2B_ADVANTAGE",
]

# Context
CONTEXT_FEATURES = [
    "SEASON_STAGE", "HOME_COURT_STRENGTH", "AWAY_TEAM_STRENGTH",
    "STRENGTH_DIFF", "HOME_ADVANTAGE_SCORE",
]

# Defensive
DEFENSIVE_FEATURES = [
    "HOME_DEF_ROLL10", "AWAY_DEF_ROLL10",
    "COMBINED_DEF_ROLL10", "EXPECTED_TOTAL",
]

# Shot profiles
SHOT_FEATURES = [
    "HOME_PAINT_RATE", "HOME_3PT_RATE", "HOME_MID_RATE",
    "HOME_PAINT_PCT", "HOME_3PT_PCT", "HOME_PTS_PER_SHOT",
    "AWAY_PAINT_RATE", "AWAY_3PT_RATE", "AWAY_MID_RATE",
    "AWAY_PAINT_PCT", "AWAY_3PT_PCT", "AWAY_PTS_PER_SHOT",
    "COMBINED_3PT_RATE", "COMBINED_PAINT_RATE",
    "COMBINED_PTS_PER_SHOT", "STYLE_MISMATCH", "EFFICIENCY_EDGE",
]

# Exponential weighted moving averages
EWM_FEATURES = [
    "HOME_EWM_PTS", "HOME_EWM_FG_PCT", "HOME_EWM_FG3_PCT", "HOME_EWM_DEF",
    "AWAY_EWM_PTS", "AWAY_EWM_FG_PCT", "AWAY_EWM_FG3_PCT", "AWAY_EWM_DEF",
    "EWM_EXPECTED_TOTAL", "HOME_MOMENTUM", "AWAY_MOMENTUM", "COMBINED_MOMENTUM",
]

# Strength of Schedule
SOS_FEATURES = [
    "HOME_SOS_DEF", "AWAY_SOS_DEF", "SOS_DIFF",
    "HOME_OFF_RTG", "AWAY_OFF_RTG",
    "HOME_DEF_RTG", "AWAY_DEF_RTG",
    "HOME_NET_RTG", "AWAY_NET_RTG",
    "PACE_EST",
    "HOME_ADJ_OFF_RTG", "AWAY_ADJ_OFF_RTG",
    "HOME_ADJ_DEF_RTG", "AWAY_ADJ_DEF_RTG",
    "HOME_ADJ_NET_RTG", "AWAY_ADJ_NET_RTG",
    "HOME_ADJ_OFF", "AWAY_ADJ_OFF",
    "HOME_ADJ_DEF", "AWAY_ADJ_DEF",
    "SOS_EXPECTED_HOME", "SOS_EXPECTED_AWAY", "SOS_EXPECTED_TOTAL",
]

# Market lines
# FIX: Disabled entirely. The Odds API free plan only returns current/upcoming
#      odds, not historical. Out of 3,642 games, only 8 had real market data.
#      The remaining 3,634 were filled with SOS_EXPECTED_TOTAL as fallback,
#      causing MARKET_TOTAL_LINE to dominate at 43% feature importance while
#      providing zero real predictive signal. Re-enable once real historical
#      odds are sourced (see README Step 1).
MARKET_FEATURES = []

# Live total calibration: weight applied to sportsbook total when available.
MARKET_TOTAL_BLEND_WEIGHT = 0.65

# Player impact
PLAYER_FEATURES = [
    "HOME_TOP_IMPACT", "HOME_TOP3_IMPACT", "HOME_AVG_IMPACT",
    "HOME_DEPTH", "HOME_STAR_DEP", "HOME_BENCH",
    "HOME_AVG_MISS", "HOME_TOP_MISS", "HOME_TOP3_MISS",
    "HOME_IMPACT_AVAIL",
    "AWAY_TOP_IMPACT", "AWAY_TOP3_IMPACT", "AWAY_AVG_IMPACT",
    "AWAY_DEPTH", "AWAY_STAR_DEP", "AWAY_BENCH",
    "AWAY_AVG_MISS", "AWAY_TOP_MISS", "AWAY_TOP3_MISS",
    "AWAY_IMPACT_AVAIL",
]

# Margin rolling features
MARGIN_FEATURES = [
    "HOME_ROLL10_MARGIN",
    "AWAY_ROLL10_MARGIN",
]

# ELO ratings
ELO_FEATURES = [
    "HOME_ELO", "AWAY_ELO", "ELO_DIFF", "ELO_EXPECTED",
]

# Playoff-specific features
PLAYOFF_FEATURES = [
    "IS_PLAYOFF",
    "PLAYOFF_HOME_BOOST", "PLAYOFF_ROAD_PENALTY",
    "PLAYOFF_REST_ADVANTAGE",
    "HOME_PLAYOFF_WIN_PCT", "AWAY_PLAYOFF_WIN_PCT",
    "SERIES_GAME_NUM", "IS_ELIMINATION", "IS_CLOSEOUT",
    "HOME_SERIES_LEAD", "AWAY_SERIES_LEAD",
    "PLAYOFF_INTENSITY",
]

# Stacked total model features
# FIX: Removed PLAYOFF_HOME_BOOST, IS_ELIMINATION, IS_CLOSEOUT from meta-features.
#      These were confirmed missing from the CSV per the leakage audit, causing
#      Model C to train on NaN meta-features. They remain in PLAYOFF_FEATURES
#      for the base models. Re-add here once confirmed present in rebuilt CSV.
# FIX: Removed MARKET_TOTAL_LINE and MARKET_HOME_LINE — fake data (see above).
STACKED_TOTAL_FEATURES = [
    "PRED_HOME", "PRED_AWAY", "PRED_SUM", "PRED_MARGIN",
    "HOME_ROLL10_PTS", "AWAY_ROLL10_PTS",
    "HOME_DEF_ROLL10", "AWAY_DEF_ROLL10",
    "HOME_ELO", "AWAY_ELO", "ELO_DIFF",
    "COMBINED_PTS_ROLL10",
    "IS_PLAYOFF",
    "PLAYOFF_INTENSITY",
    "COMBINED_POSS_ROLL10",
    "PACE_EST",
    "MARKET_TOTAL_LINE",
    "MARKET_HOME_LINE",
]

# ── Combined final feature list ───────────────────────────────
FEATURE_COLS_FINAL = (
    ROLLING_HOME_FEATURES
    + ROLLING_AWAY_FEATURES
    + COMBINED_FEATURES
    + MATCHUP_FEATURES
    + REST_STREAK_FEATURES
    + CONTEXT_FEATURES
    + DEFENSIVE_FEATURES
    + SHOT_FEATURES
    + EWM_FEATURES
    + SOS_FEATURES
    + MARKET_FEATURES
    + PLAYER_FEATURES
    + MARGIN_FEATURES
    + ELO_FEATURES
    + PLAYOFF_FEATURES
)

# Metadata columns (not features -- kept for reference/joins)
# FIX: Removed IS_PLAYOFF from META_COLS. It is already in PLAYOFF_FEATURES
#      above, and having it in both caused duplicate columns in the final
#      DataFrame selection in pipeline.py, silently breaking column ops.
# FIX: Added PTS_MARGIN so the garbage-time filter in train.py can run.
#      Without this column in the saved CSV, the filter silently skips.
META_COLS = [
    "GAME_ID", "GAME_DATE", "SEASON",
    "HOME_TEAM", "AWAY_TEAM",
    "HOME_PTS", "AWAY_PTS", "TOTAL_PTS", "HOME_WIN",
    "SEASON_TYPE", "PTS_MARGIN",
]

# Target columns
TARGET_HOME  = "HOME_PTS"
TARGET_AWAY  = "AWAY_PTS"
TARGET_TOTAL = "TOTAL_PTS"
TARGET_WIN   = "HOME_WIN"


# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 9,
    "min_samples_leaf": 15,
    "random_state": 42,
    "n_jobs": -1,
}

LGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
}

# Enable VotingRegressor ensemble: blend top-N models per target
VOTING_ENSEMBLE = True
VOTING_TOP_N = 3


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────

# Garbage time filter: remove blowouts > 25pt margin from training
BLOWOUT_MARGIN_THRESHOLD = 35

# OT detection: total > 240 AND margin < 10
OT_TOTAL_THRESHOLD  = 240
OT_MARGIN_THRESHOLD = 10

# Rolling window size
ROLLING_WINDOW = 10

# EWM span
EWM_SPAN = 10

# API rate limit delay (seconds between calls)
API_DELAY = 0.6

# Model version tag
MODEL_VERSION = "v9"
