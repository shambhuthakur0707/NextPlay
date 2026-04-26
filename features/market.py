# -*- coding: utf-8 -*-
"""
NextPlay -- Market Features
===========================
Merges closing line data and derives market-informed features.
"""

import numpy as np
import pandas as pd


def _to_naive_day(series):
    """Convert datetime-like series to timezone-naive normalized day."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    s = s.dt.tz_convert(None)
    return s.dt.normalize()


def _normalize_market_frame(market_df):
    """Normalize external market data to expected schema."""
    df = market_df.copy()

    rename_map = {
        "HOME_LINE": "CLOSE_HOME_LINE",
        "TOTAL_LINE": "CLOSE_TOTAL",
        "SPREAD_HOME": "CLOSE_HOME_LINE",
        "CLOSE_SPREAD": "CLOSE_HOME_LINE",
        "TOTAL": "CLOSE_TOTAL",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "OPEN_HOME_LINE" not in df.columns and "CLOSE_HOME_LINE" in df.columns:
        df["OPEN_HOME_LINE"] = df["CLOSE_HOME_LINE"]
    if "OPEN_TOTAL" not in df.columns and "CLOSE_TOTAL" in df.columns:
        df["OPEN_TOTAL"] = df["CLOSE_TOTAL"]

    if "MARKET_SPREAD_MOVE" not in df.columns:
        df["MARKET_SPREAD_MOVE"] = df["CLOSE_HOME_LINE"] - df["OPEN_HOME_LINE"]
    if "MARKET_TOTAL_MOVE" not in df.columns:
        df["MARKET_TOTAL_MOVE"] = df["CLOSE_TOTAL"] - df["OPEN_TOTAL"]

    required = ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "CLOSE_HOME_LINE", "CLOSE_TOTAL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    df["GAME_DATE"] = _to_naive_day(df["GAME_DATE"])
    return df


def merge_market_features(games, market_df):
    """Merge market lines and engineer line movement + model-vs-market features."""
    if market_df is None or len(market_df) == 0:
        return games

    df = games.copy()
    mkt = _normalize_market_frame(market_df)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = _to_naive_day(df["GAME_DATE"])

    keep_cols = [
        "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
        "OPEN_HOME_LINE", "CLOSE_HOME_LINE",
        "OPEN_TOTAL", "CLOSE_TOTAL",
        "MARKET_SPREAD_MOVE", "MARKET_TOTAL_MOVE",
    ]
    keep_cols = [c for c in keep_cols if c in mkt.columns]

    merged = df.merge(
        mkt[keep_cols],
        on=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"],
        how="left",
    )

    merged["MARKET_HOME_LINE"] = merged.get("CLOSE_HOME_LINE")
    merged["MARKET_TOTAL_LINE"] = merged.get("CLOSE_TOTAL")
    merged["MARKET_TOTAL_MOVE"] = merged.get("MARKET_TOTAL_MOVE")
    merged["MARKET_SPREAD_MOVE"] = merged.get("MARKET_SPREAD_MOVE")

    # Neutral defaults for games without line matches.
    merged["MARKET_HOME_LINE"] = merged["MARKET_HOME_LINE"].fillna(0.0)
    merged["MARKET_TOTAL_MOVE"] = merged["MARKET_TOTAL_MOVE"].fillna(0.0)
    merged["MARKET_SPREAD_MOVE"] = merged["MARKET_SPREAD_MOVE"].fillna(0.0)

    total_ref = None
    for col in ["SOS_EXPECTED_TOTAL", "EWM_EXPECTED_TOTAL", "EXPECTED_TOTAL"]:
        if col in merged.columns:
            total_ref = merged[col]
            break
    if total_ref is None:
        total_ref = 0.0
    merged["MARKET_TOTAL_LINE"] = merged["MARKET_TOTAL_LINE"].fillna(total_ref)

    # Convert home spread convention (negative means favored) to margin expectation.
    margin_exp = -merged["MARKET_HOME_LINE"]
    merged["MARKET_HOME_IMPLIED"] = (merged["MARKET_TOTAL_LINE"] + margin_exp) / 2
    merged["MARKET_AWAY_IMPLIED"] = merged["MARKET_TOTAL_LINE"] - merged["MARKET_HOME_IMPLIED"]

    model_total_ref = np.nan
    for col in ["SOS_EXPECTED_TOTAL", "EWM_EXPECTED_TOTAL", "EXPECTED_TOTAL"]:
        if col in merged.columns:
            model_total_ref = merged[col]
            break

    model_margin_ref = np.nan
    if "SOS_EXPECTED_HOME" in merged.columns and "SOS_EXPECTED_AWAY" in merged.columns:
        model_margin_ref = merged["SOS_EXPECTED_HOME"] - merged["SOS_EXPECTED_AWAY"]
    elif "HOME_ROLL10_PTS" in merged.columns and "AWAY_ROLL10_PTS" in merged.columns:
        model_margin_ref = merged["HOME_ROLL10_PTS"] - merged["AWAY_ROLL10_PTS"]

    merged["MODEL_TOTAL_VS_MARKET"] = model_total_ref - merged["MARKET_TOTAL_LINE"]
    merged["MODEL_MARGIN_VS_MARKET"] = model_margin_ref - margin_exp

    merged["MARKET_HOME_IMPLIED"] = merged["MARKET_HOME_IMPLIED"].fillna(0.0)
    merged["MARKET_AWAY_IMPLIED"] = merged["MARKET_AWAY_IMPLIED"].fillna(0.0)
    merged["MODEL_TOTAL_VS_MARKET"] = merged["MODEL_TOTAL_VS_MARKET"].fillna(0.0)
    merged["MODEL_MARGIN_VS_MARKET"] = merged["MODEL_MARGIN_VS_MARKET"].fillna(0.0)

    return merged
