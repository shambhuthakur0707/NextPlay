# -*- coding: utf-8 -*-
"""
NextPlay -- Historical Betting Odds Ingestion
==============================================
Converts the Kaggle NBA Betting Dataset (2007-2025) into the market
feature format used by features/market.py.

Source dataset columns used:
    season, date, away, home, total, spread, whos_favored,
    moneyline_away, moneyline_home

Output columns (matches existing market pipeline):
    GAME_DATE, HOME_TEAM, AWAY_TEAM,
    MARKET_TOTAL_LINE, MARKET_HOME_LINE,
    MARKET_HOME_IMPLIED, MARKET_AWAY_IMPLIED,
    MARKET_TOTAL_MOVE, MARKET_SPREAD_MOVE   (set to 0 -- not in source)

Usage:
    python ingestion/betting_odds.py

    This writes data/market_lines.csv which is automatically picked
    up by features/market.py during full_rebuild.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# TEAM NAME MAPPING
# Kaggle dataset uses lowercase short codes; NextPlay uses uppercase
# NBA standard abbreviations.
# ─────────────────────────────────────────────────────────────
TEAM_MAP = {
    "atl":  "ATL",
    "bkn":  "BKN",
    "bos":  "BOS",
    "cha":  "CHA",
    "chi":  "CHI",
    "cle":  "CLE",
    "dal":  "DAL",
    "den":  "DEN",
    "det":  "DET",
    "gs":   "GSW",   # golden state
    "hou":  "HOU",
    "ind":  "IND",
    "lac":  "LAC",
    "lal":  "LAL",
    "mem":  "MEM",
    "mia":  "MIA",
    "mil":  "MIL",
    "min":  "MIN",
    "no":   "NOP",   # new orleans
    "nop":  "NOP",
    "ny":   "NYK",   # new york
    "nyk":  "NYK",
    "okc":  "OKC",
    "orl":  "ORL",
    "phi":  "PHI",
    "phx":  "PHX",
    "por":  "POR",
    "sac":  "SAC",
    "sa":   "SAS",   # san antonio
    "sas":  "SAS",
    "tor":  "TOR",
    "utah": "UTA",   # utah
    "uta":  "UTA",
    "was":  "WAS",
    "wsh":  "WAS",   # washington
    # Historical teams that may appear
    "nj":   "BKN",   # new jersey -> brooklyn
    "sea":  "OKC",   # seattle -> okc
    "van":  "MEM",   # vancouver -> memphis
    "nor":  "NOP",   # new orleans
}


def moneyline_to_implied_prob(ml):
    """
    Convert American moneyline odds to implied probability.

    Positive moneyline (underdog): prob = 100 / (ml + 100)
    Negative moneyline (favorite): prob = |ml| / (|ml| + 100)
    """
    ml = pd.to_numeric(ml, errors="coerce")
    prob = np.where(
        ml >= 0,
        100 / (ml + 100),
        np.abs(ml) / (np.abs(ml) + 100),
    )
    return prob


def load_betting_odds(
    raw_path="data/nba_betting_raw.csv",
    output_path="data/market_lines.csv",
    seasons=(2024, 2025, 2026),   # kaggle encodes season by end year
    include_playoffs=True,
):
    """
    Load raw Kaggle betting data, filter to target seasons,
    map team names, compute implied probabilities, and save
    in the format expected by features/market.py.

    Args:
        raw_path: path to downloaded Kaggle CSV
        output_path: where to write the processed market lines
        seasons: tuple of season-end years to include
                 2024 = 2023-24, 2025 = 2024-25, 2026 = 2025-26
        include_playoffs: whether to include playoff games

    Returns:
        DataFrame of processed market lines
    """
    print(f"[ODDS] Loading raw betting data from {raw_path}...")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw betting data not found at {raw_path}.\n"
            f"Download from Kaggle and save as {raw_path}"
        )

    df = pd.read_csv(raw_path)
    print(f"[ODDS] Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"[ODDS] Seasons in dataset: {sorted(df['season'].unique())}")

    # ── Filter to target seasons ────────────────────────────────────────
    df = df[df["season"].isin(seasons)].copy()
    print(f"[ODDS] After season filter ({list(seasons)}): {len(df):,} rows")

    # ── Filter game types ───────────────────────────────────────────────
    if not include_playoffs:
        df = df[df["regular"] == True].copy()
        print(f"[ODDS] After regular-season filter: {len(df):,} rows")

    # ── Parse date ──────────────────────────────────────────────────────
    df["GAME_DATE"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])

    # ── Map team names ──────────────────────────────────────────────────
    unmapped_away = set(df["away"].str.lower().unique()) - set(TEAM_MAP.keys())
    unmapped_home = set(df["home"].str.lower().unique()) - set(TEAM_MAP.keys())
    unmapped = unmapped_away | unmapped_home
    if unmapped:
        print(f"[WARN] Unmapped team codes (will be dropped): {sorted(unmapped)}")

    df["HOME_TEAM"] = df["home"].str.lower().map(TEAM_MAP)
    df["AWAY_TEAM"] = df["away"].str.lower().map(TEAM_MAP)

    # Drop rows where team mapping failed
    before = len(df)
    df = df.dropna(subset=["HOME_TEAM", "AWAY_TEAM"])
    if len(df) < before:
        print(f"[ODDS] Dropped {before - len(df)} rows with unmapped teams")

    # ── Market total line ───────────────────────────────────────────────
    df["MARKET_TOTAL_LINE"] = pd.to_numeric(df["total"], errors="coerce")

    # ── Market home line (spread from home team perspective) ────────────
    # Kaggle: spread is always positive, whos_favored tells us direction.
    # Convention: negative = home is favored, positive = away is favored.
    df["spread_num"] = pd.to_numeric(df["spread"], errors="coerce")
    df["MARKET_HOME_LINE"] = np.where(
        df["whos_favored"] == "home",
        -df["spread_num"],   # home favored: negative spread
        df["spread_num"],    # away favored: positive spread
    )

    # ── Implied probabilities from moneylines ───────────────────────────
    df["MARKET_HOME_IMPLIED"] = moneyline_to_implied_prob(df["moneyline_home"])
    df["MARKET_AWAY_IMPLIED"] = moneyline_to_implied_prob(df["moneyline_away"])

    # Normalize to remove vig (sum to 1.0)
    total_implied = df["MARKET_HOME_IMPLIED"] + df["MARKET_AWAY_IMPLIED"]
    df["MARKET_HOME_IMPLIED"] = df["MARKET_HOME_IMPLIED"] / total_implied
    df["MARKET_AWAY_IMPLIED"] = df["MARKET_AWAY_IMPLIED"] / total_implied

    # ── Line movement (not available in this dataset -- set to 0) ───────
    # MARKET_TOTAL_MOVE and MARKET_SPREAD_MOVE require open + close lines.
    # This dataset only has one line per game. Set to 0 for compatibility.
    df["MARKET_TOTAL_MOVE"]  = 0.0
    df["MARKET_SPREAD_MOVE"] = 0.0

    # ── IS_PLAYOFF flag ─────────────────────────────────────────────────
    df["IS_PLAYOFF"] = df["playoffs"].astype(int)

    # ── Select and order output columns ─────────────────────────────────
    output_cols = [
        "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
        "MARKET_TOTAL_LINE", "MARKET_HOME_LINE",
        "MARKET_HOME_IMPLIED", "MARKET_AWAY_IMPLIED",
        "MARKET_TOTAL_MOVE", "MARKET_SPREAD_MOVE",
        "IS_PLAYOFF",
    ]
    out = df[output_cols].copy()

    # ── Drop rows with missing total line (core feature) ────────────────
    before = len(out)
    out = out.dropna(subset=["MARKET_TOTAL_LINE"])
    if len(out) < before:
        print(f"[ODDS] Dropped {before - len(out)} rows with missing total line")

    out = out.sort_values("GAME_DATE").reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────
    out.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(out):,} games to {output_path}")
    print(f"     Date range : {out['GAME_DATE'].min().date()} -> {out['GAME_DATE'].max().date()}")
    print(f"     Seasons    : {sorted(df['season'].unique())}")
    print(f"     Playoff    : {out['IS_PLAYOFF'].sum()} games")
    print(f"     Regular    : {(out['IS_PLAYOFF'] == 0).sum()} games")
    print(f"\n     MARKET_TOTAL_LINE stats:")
    print(out["MARKET_TOTAL_LINE"].describe().to_string())

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    out = load_betting_odds(
        raw_path="data/nba_betting_raw.csv",
        output_path="data/market_lines.csv",
        seasons=(2024, 2025, 2026),
        include_playoffs=True,
    )

    # Quick sanity check
    print("\n[SANITY] Sample rows:")
    print(out.sample(5).to_string())

    print("\n[SANITY] Null counts:")
    print(out.isnull().sum().to_string())

    print("\n[DONE] market_lines.csv is ready for full_rebuild.")
    print("       Next: re-enable MARKET_FEATURES in config.py")
    print("       Then: python -m pipelines.full_rebuild")