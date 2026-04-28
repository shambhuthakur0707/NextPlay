# -*- coding: utf-8 -*-
"""
NextPlay -- Rolling Average Features
=====================================
For each game, calculates each team's recent form from a unified
team-history table so the same team is not tracked separately by
home and away slot.

KEY RULE: shift(1) before rolling -- never include the current game
in its own features (prevents data leakage).
"""

import pandas as pd

from config import ROLLING_WINDOW


TEAM_STAT_COLUMNS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "FG3A", "FTA",
    "AST", "TOV", "OREB", "REB", "PF", "TOTAL_PTS", "POSS_EST",
]


def _build_team_history(games):
    """Convert one-row-per-game data into one-row-per-team-game data."""
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)

    home_cols = [
        "GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM",
        "HOME_PTS", "HOME_FGM", "HOME_FGA", "HOME_FG_PCT",
        "HOME_FG3M", "HOME_FG3A", "HOME_FG3_PCT",
        "HOME_FTM", "HOME_FTA", "HOME_FT_PCT",
        "HOME_AST", "HOME_TOV", "HOME_OREB", "HOME_DREB",
        "HOME_REB", "HOME_PF", "HOME_PLUS_MINUS",
        "TOTAL_PTS",
    ]
    # Preserve IS_PLAYOFF and SEASON_TYPE if present
    if "IS_PLAYOFF" in df.columns:
        home_cols.append("IS_PLAYOFF")
    if "SEASON_TYPE" in df.columns:
        home_cols.append("SEASON_TYPE")
    
    home = df[home_cols].copy()
    home["TEAM_ABBR"] = home["HOME_TEAM"]
    home["OPP_TEAM"] = home["AWAY_TEAM"]
    home["IS_HOME"] = True
    home = home.rename(
        columns={
            "HOME_PTS": "PTS",
            "HOME_FGM": "FGM",
            "HOME_FGA": "FGA",
            "HOME_FG_PCT": "FG_PCT",
            "HOME_FG3M": "FG3M",
            "HOME_FG3A": "FG3A",
            "HOME_FG3_PCT": "FG3_PCT",
            "HOME_FTM": "FTM",
            "HOME_FTA": "FTA",
            "HOME_FT_PCT": "FT_PCT",
            "HOME_AST": "AST",
            "HOME_TOV": "TOV",
            "HOME_OREB": "OREB",
            "HOME_DREB": "DREB",
            "HOME_REB": "REB",
            "HOME_PF": "PF",
            "HOME_PLUS_MINUS": "PLUS_MINUS",
        }
    )
    home["OPP_PTS"] = df["AWAY_PTS"].values

    away_cols = [
        "GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM",
        "AWAY_PTS", "AWAY_FGM", "AWAY_FGA", "AWAY_FG_PCT",
        "AWAY_FG3M", "AWAY_FG3A", "AWAY_FG3_PCT",
        "AWAY_FTM", "AWAY_FTA", "AWAY_FT_PCT",
        "AWAY_AST", "AWAY_TOV", "AWAY_OREB", "AWAY_DREB",
        "AWAY_REB", "AWAY_PF", "AWAY_PLUS_MINUS",
        "TOTAL_PTS",
    ]
    # Preserve IS_PLAYOFF and SEASON_TYPE if present
    if "IS_PLAYOFF" in df.columns:
        away_cols.append("IS_PLAYOFF")
    if "SEASON_TYPE" in df.columns:
        away_cols.append("SEASON_TYPE")
    
    away = df[away_cols].copy()
    away["TEAM_ABBR"] = away["AWAY_TEAM"]
    away["OPP_TEAM"] = away["HOME_TEAM"]
    away["IS_HOME"] = False
    away = away.rename(
        columns={
            "AWAY_PTS": "PTS",
            "AWAY_FGM": "FGM",
            "AWAY_FGA": "FGA",
            "AWAY_FG_PCT": "FG_PCT",
            "AWAY_FG3M": "FG3M",
            "AWAY_FG3A": "FG3A",
            "AWAY_FG3_PCT": "FG3_PCT",
            "AWAY_FTM": "FTM",
            "AWAY_FTA": "FTA",
            "AWAY_FT_PCT": "FT_PCT",
            "AWAY_AST": "AST",
            "AWAY_TOV": "TOV",
            "AWAY_OREB": "OREB",
            "AWAY_DREB": "DREB",
            "AWAY_REB": "REB",
            "AWAY_PF": "PF",
            "AWAY_PLUS_MINUS": "PLUS_MINUS",
        }
    )
    away["OPP_PTS"] = df["HOME_PTS"].values

    team_df = pd.concat([home, away], ignore_index=True)
    team_df["POSS_EST"] = (
        team_df["FGA"] + 0.44 * team_df["FTA"] - team_df["OREB"] + team_df["TOV"]
    )
    team_df["EFG"] = (team_df["FGM"] + 0.5 * team_df["FG3M"]) / team_df["FGA"]
    team_df["TS"] = team_df["PTS"] / (2 * (team_df["FGA"] + 0.44 * team_df["FTA"]))

    return team_df.sort_values(["TEAM_ABBR", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _add_team_rolls(team_df, window=ROLLING_WINDOW):
    """Add shifted rolling features to the team-history table."""
    df = team_df.copy()

    for stat in TEAM_STAT_COLUMNS:
        df[f"ROLL10_{stat}"] = df.groupby("TEAM_ABBR")[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    df["EFG_ROLL10"] = df.groupby("TEAM_ABBR")["EFG"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    df["TS_ROLL10"] = df.groupby("TEAM_ABBR")["TS"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    df["DEF_ROLL10"] = df.groupby("TEAM_ABBR")["OPP_PTS"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )

    for size, label in [(5, "ROLL5"), (15, "ROLL15")]:
        df[f"{label}_PTS"] = df.groupby("TEAM_ABBR")["PTS"].transform(
            lambda x: x.shift(1).rolling(size, min_periods=2).mean()
        )

    df["FORM_TREND"] = df["ROLL5_PTS"] - df["ROLL15_PTS"]
    return df


def _team_snapshot(team_df, team, as_of_date=None, window=ROLLING_WINDOW):
    """Return the latest rolling snapshot for one team before a date."""
    history = team_df[team_df["TEAM_ABBR"] == team].copy()
    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        history = history[history["GAME_DATE"] < as_of_date]

    if len(history) == 0:
        return None

    history = _add_team_rolls(history, window=window)
    row = history.sort_values(["GAME_DATE", "GAME_ID"]).iloc[-1]

    snapshot = {
        "ROLL10_PTS": row.get("ROLL10_PTS"),
        "ROLL10_FG_PCT": row.get("ROLL10_FG_PCT"),
        "ROLL10_FG3_PCT": row.get("ROLL10_FG3_PCT"),
        "ROLL10_FT_PCT": row.get("ROLL10_FT_PCT"),
        "ROLL10_FG3A": row.get("ROLL10_FG3A"),
        "ROLL10_FTA": row.get("ROLL10_FTA"),
        "ROLL10_AST": row.get("ROLL10_AST"),
        "ROLL10_TOV": row.get("ROLL10_TOV"),
        "ROLL10_OREB": row.get("ROLL10_OREB"),
        "ROLL10_REB": row.get("ROLL10_REB"),
        "ROLL10_PF": row.get("ROLL10_PF"),
        "ROLL10_TOTAL_PTS": row.get("ROLL10_TOTAL_PTS"),
        "ROLL10_POSS_EST": row.get("ROLL10_POSS_EST"),
        "DEF_ROLL10": row.get("DEF_ROLL10"),
        "EFG_ROLL10": row.get("EFG_ROLL10"),
        "TS_ROLL10": row.get("TS_ROLL10"),
        "ROLL5_PTS": row.get("ROLL5_PTS"),
        "ROLL15_PTS": row.get("ROLL15_PTS"),
        "FORM_TREND": row.get("FORM_TREND"),
    }
    snapshot["EFG"] = row.get("EFG")
    snapshot["TS"] = row.get("TS")
    snapshot["PTS"] = row.get("PTS")
    snapshot["POSS_EST"] = row.get("POSS_EST")
    return snapshot


def _attach_team_snapshots(df, team_df, window=ROLLING_WINDOW):
    """Overwrite slot-specific rolling fields with unified team snapshots."""
    result = df.copy()

    home_rows = []
    away_rows = []
    for _, row in result.iterrows():
        home_snap = _team_snapshot(team_df, row["HOME_TEAM"], row["GAME_DATE"], window)
        away_snap = _team_snapshot(team_df, row["AWAY_TEAM"], row["GAME_DATE"], window)

        if home_snap is None or away_snap is None:
            home_rows.append(None)
            away_rows.append(None)
            continue

        home_rows.append(home_snap)
        away_rows.append(away_snap)

    if any(v is None for v in home_rows + away_rows):
        # Let the downstream dropna logic remove early games without enough history.
        pass

    for prefix, snapshots in [("HOME", home_rows), ("AWAY", away_rows)]:
        example = next((snap for snap in snapshots if snap is not None), None)
        if example is None:
            continue
        for key in example.keys():
            result[f"{prefix}_{key}"] = [
                snap.get(key) if snap is not None else None for snap in snapshots
            ]

    result["COMBINED_PTS_ROLL10"] = result["HOME_ROLL10_PTS"] + result["AWAY_ROLL10_PTS"]
    result["COMBINED_FG3A_ROLL10"] = result["HOME_ROLL10_FG3A"] + result["AWAY_ROLL10_FG3A"]
    result["COMBINED_FTA_ROLL10"] = result["HOME_ROLL10_FTA"] + result["AWAY_ROLL10_FTA"]
    result["COMBINED_POSS_ROLL10"] = result["HOME_ROLL10_POSS_EST"] + result["AWAY_ROLL10_POSS_EST"]
    result["COMBINED_DEF_ROLL10"] = result["HOME_DEF_ROLL10"] + result["AWAY_DEF_ROLL10"]
    result["EXPECTED_TOTAL"] = (
        result["HOME_ROLL10_PTS"]
        + result["AWAY_ROLL10_PTS"]
        + result["HOME_DEF_ROLL10"]
        + result["AWAY_DEF_ROLL10"]
    ) / 2
    result["TOV_DIFF_ROLL10"] = result["HOME_ROLL10_TOV"] - result["AWAY_ROLL10_TOV"]
    result["OREB_DIFF_ROLL10"] = result["HOME_ROLL10_OREB"] - result["AWAY_ROLL10_OREB"]
    result["HOME_FORM_TREND"] = result["HOME_ROLL5_PTS"] - result["HOME_ROLL15_PTS"]
    result["AWAY_FORM_TREND"] = result["AWAY_ROLL5_PTS"] - result["AWAY_ROLL15_PTS"]

    return result


def add_rolling_features(games, window=ROLLING_WINDOW):
    """Add unified rolling averages of the last N games for both teams."""
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)
    team_df = _build_team_history(df)
    team_df = _add_team_rolls(team_df, window=window)
    df = _attach_team_snapshots(df, team_df, window=window)
    return df


def build_team_rolling_snapshot(games, team, as_of_date=None, window=ROLLING_WINDOW):
    """Return a team snapshot keyed by rolling feature name for live inference."""
    team_df = _build_team_history(games)
    return _team_snapshot(team_df, team, as_of_date=as_of_date, window=window)
