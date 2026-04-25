# -*- coding: utf-8 -*-
"""NextPlay -- Shared utility functions."""

import numpy as np


def compute_streak(wl_list):
    """
    Compute running win/loss streak from a list of 'W'/'L' results.
    Positive = win streak, Negative = loss streak.
    The value at each position represents the streak ENTERING that game.
    """
    streaks = []
    current = 0
    for result in wl_list:
        streaks.append(current)
        if result == "W":
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
    return streaks


def style_label(threept_rate, paint_rate):
    """Classify a team's shot style based on 3PT and paint rates."""
    if threept_rate > 0.45:
        return "3PT-heavy"
    elif paint_rate > 0.52:
        return "Paint-dominant"
    elif threept_rate > 0.40:
        return "Balanced 3PT"
    else:
        return "Balanced"


def overtime_risk(pred_margin):
    """Estimate overtime probability from predicted margin."""
    abs_margin = abs(pred_margin)
    if abs_margin <= 2:
        return "HIGH OT risk -- game too close"
    elif abs_margin <= 5:
        return "MEDIUM OT risk"
    else:
        return "Low OT risk"


def blowout_risk(home_streak, away_streak, strength_diff):
    """Flag games at risk of becoming blowouts."""
    if abs(home_streak) >= 7 or abs(away_streak) >= 7:
        return "??  Blowout risk -- large momentum gap"
    elif abs(strength_diff) > 0.3:
        return "??  Blowout risk -- large quality gap"
    else:
        return "Normal game expected"


def win_probability(margin, scale=10):
    """Convert predicted margin to win probability using logistic function."""
    return 1 / (1 + np.exp(-margin / scale))


def confidence_tier(margin):
    """Simple confidence tier from margin."""
    abs_margin = abs(margin)
    if abs_margin >= 12:
        return "HIGH"
    elif abs_margin >= 7:
        return "MEDIUM"
    elif abs_margin >= 3:
        return "LOW"
    else:
        return "SKIP -- too close to call"
