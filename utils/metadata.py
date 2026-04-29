# -*- coding: utf-8 -*-
"""Helpers for version, feature counts, and metadata consistency."""

from config import (
    MODEL_VERSION,
    FEATURE_COLS_FINAL,
    STACKED_TOTAL_FEATURES,
    ROLLING_HOME_FEATURES,
    ROLLING_AWAY_FEATURES,
    COMBINED_FEATURES,
    MATCHUP_FEATURES,
    REST_STREAK_FEATURES,
    CONTEXT_FEATURES,
    DEFENSIVE_FEATURES,
    SHOT_FEATURES,
    EWM_FEATURES,
    SOS_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    MARGIN_FEATURES,
    ELO_FEATURES,
    PLAYOFF_FEATURES,
)


def model_version_display():
    """Return model version in dashboard/doc style (e.g., V8)."""
    return MODEL_VERSION.upper()


def feature_category_counts():
    """Return [(category, count, description), ...] for base features."""
    return [
        ("Rolling Home", len(ROLLING_HOME_FEATURES), "Home rolling box/efficiency form"),
        ("Rolling Away", len(ROLLING_AWAY_FEATURES), "Away rolling box/efficiency form"),
        ("Combined Rolling", len(COMBINED_FEATURES), "Pace/turnover/rebound combined signals"),
        ("Matchup", len(MATCHUP_FEATURES), "Head-to-head summary signals"),
        ("Rest/Travel/Streak", len(REST_STREAK_FEATURES), "Rest, travel load, streak signals"),
        ("Context", len(CONTEXT_FEATURES), "Season stage and strength context"),
        ("Defensive", len(DEFENSIVE_FEATURES), "Defensive trend and expected total"),
        ("Shot Profile", len(SHOT_FEATURES), "Shot distribution and efficiency"),
        ("EWM Momentum", len(EWM_FEATURES), "Recent weighted momentum signals"),
        ("SoS/Ratings", len(SOS_FEATURES), "Schedule-adjusted offense/defense ratings"),
        ("Market", len(MARKET_FEATURES), "Closing line and movement features"),
        ("Player Impact", len(PLAYER_FEATURES), "Availability and impact aggregates"),
        ("Margin", len(MARGIN_FEATURES), "Rolling margin stability signals"),
        ("ELO", len(ELO_FEATURES), "ELO and expected win strength"),
        ("Playoff", len(PLAYOFF_FEATURES), "Playoff-specific context and intensity signals"),
    ]


def project_metadata_snapshot():
    """Return lightweight metadata for UI/docs checks."""
    return {
        "model_version": model_version_display(),
        "base_feature_count": len(FEATURE_COLS_FINAL),
        "stacked_total_feature_count": len(STACKED_TOTAL_FEATURES),
    }
