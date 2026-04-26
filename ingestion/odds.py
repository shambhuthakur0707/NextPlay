# -*- coding: utf-8 -*-
"""
NextPlay -- Market Lines Ingestion
==================================
Pulls NBA market lines from The Odds API and normalizes into the
project schema used by feature engineering and backtests.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from config import (
    API_DELAY,
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    ODDS_SPORT_KEY,
)


TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def _team_abbr(name):
    return TEAM_NAME_TO_ABBR.get(name, name)


def _parse_bookmaker_markets(bookmaker, home_team):
    home_line = None
    total_line = None

    for market in bookmaker.get("markets", []):
        if market.get("key") == "spreads":
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_team:
                    home_line = outcome.get("point")
        elif market.get("key") == "totals":
            points = [o.get("point") for o in market.get("outcomes", [])]
            points = [p for p in points if p is not None]
            if points:
                total_line = float(np.mean(points))

    return home_line, total_line


def pull_closing_market_lines(
    api_key=None,
    sport_key=ODDS_SPORT_KEY,
    date_from=None,
    date_to=None,
    regions="us",
    bookmakers="fanduel,draftkings,betmgm,caesars",
):
    """
    Pull market lines from The Odds API and normalize to one row per game.

    Notes:
    - Free tiers often expose the latest line snapshot only.
    - OPEN_* columns are populated with CLOSE_* when no open snapshot exists.
    """
    key = api_key or ODDS_API_KEY
    if not key:
        raise ValueError("ODDS_API_KEY is missing. Set it in environment or pass api_key.")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "bookmakers": bookmakers,
        "dateFormat": "iso",
    }
    if date_from:
        params["commenceTimeFrom"] = pd.Timestamp(date_from).isoformat()
    if date_to:
        params["commenceTimeTo"] = pd.Timestamp(date_to).isoformat()

    response = requests.get(url, params=params, timeout=30)

    # Some Odds API plans/endpoints reject date range filters on this route.
    # Retry once without commence-time filters so the pipeline can proceed.
    if response.status_code == 422 and (
        "commenceTimeFrom" in params or "commenceTimeTo" in params
    ):
        params.pop("commenceTimeFrom", None)
        params.pop("commenceTimeTo", None)
        response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()
    payload = response.json()

    rows = []
    for game in payload:
        commence = pd.to_datetime(game.get("commence_time"))
        home_name = game.get("home_team")
        away_name = game.get("away_team")

        # Some responses expose teams as a list; keep explicit handling for safety.
        if away_name is None and isinstance(game.get("teams"), list):
            teams = [t for t in game.get("teams", []) if t != home_name]
            away_name = teams[0] if teams else None

        home_team = _team_abbr(home_name)
        away_team = _team_abbr(away_name)

        spread_vals = []
        total_vals = []
        for bookmaker in game.get("bookmakers", []):
            home_line, total_line = _parse_bookmaker_markets(bookmaker, home_name)
            if home_line is not None:
                spread_vals.append(float(home_line))
            if total_line is not None:
                total_vals.append(float(total_line))

        close_home_line = float(np.median(spread_vals)) if spread_vals else np.nan
        close_total = float(np.median(total_vals)) if total_vals else np.nan

        rows.append(
            {
                "GAME_DATE": commence.normalize(),
                "PULL_TIME": datetime.utcnow().isoformat(),
                "HOME_TEAM": home_team,
                "AWAY_TEAM": away_team,
                "OPEN_HOME_LINE": close_home_line,
                "CLOSE_HOME_LINE": close_home_line,
                "OPEN_TOTAL": close_total,
                "CLOSE_TOTAL": close_total,
                "MARKET_SPREAD_MOVE": 0.0,
                "MARKET_TOTAL_MOVE": 0.0,
            }
        )

    time.sleep(API_DELAY)
    lines = pd.DataFrame(rows)
    if len(lines) == 0:
        return lines

    lines = lines.sort_values(["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)
    return lines
