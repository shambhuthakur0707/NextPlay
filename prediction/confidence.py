# -*- coding: utf-8 -*-
"""
NextPlay -- Confidence Scoring & Risk Assessment
===================================================
Calculates prediction confidence, overtime risk, and blowout risk.
"""
import numpy as np


def calculate_confidence(margin, home_momentum=0, away_momentum=0,
                         strength_diff=0, home_streak=0, away_streak=0):
    """
    Calculate a multi-factor confidence score for a prediction.

    Combines:
    - Predicted margin (strongest signal)
    - Momentum alignment
    - Team strength gap
    - Streak patterns

    Args:
        margin: predicted home score - away score
        home_momentum: home team EWM momentum
        away_momentum: away team EWM momentum
        strength_diff: home strength - away strength
        home_streak: home team win/loss streak
        away_streak: away team win/loss streak

    Returns:
        dict with confidence_score, tier, and breakdown
    """
    # Factor 1: Margin magnitude (0-40 points)
    abs_margin = abs(margin)
    margin_score = min(abs_margin * 3.3, 40)

    # Factor 2: Momentum alignment (0-20 points)
    # If margin favors home AND home has positive momentum, that's aligned
    momentum_diff = home_momentum - away_momentum
    if (margin > 0 and momentum_diff > 0) or (margin < 0 and momentum_diff < 0):
        momentum_score = min(abs(momentum_diff) * 4, 20)
    else:
        momentum_score = max(0, 10 - abs(momentum_diff) * 2)

    # Factor 3: Strength gap alignment (0-20 points)
    if (margin > 0 and strength_diff > 0) or (margin < 0 and strength_diff < 0):
        strength_score = min(abs(strength_diff) * 40, 20)
    else:
        strength_score = max(0, 10 - abs(strength_diff) * 20)

    # Factor 4: Streak alignment (0-20 points)
    streak_diff = home_streak - away_streak
    if (margin > 0 and streak_diff > 0) or (margin < 0 and streak_diff < 0):
        streak_score = min(abs(streak_diff) * 3, 20)
    else:
        streak_score = max(0, 10 - abs(streak_diff))

    total = margin_score + momentum_score + strength_score + streak_score

    # Determine tier
    if total >= 75:
        tier = "HIGH"
    elif total >= 55:
        tier = "MEDIUM"
    elif total >= 35:
        tier = "LOW"
    else:
        tier = "SKIP -- too close to call"

    return {
        "confidence_score": round(total, 1),
        "tier": tier,
        "breakdown": {
            "margin": round(margin_score, 1),
            "momentum": round(momentum_score, 1),
            "strength": round(strength_score, 1),
            "streak": round(streak_score, 1),
        },
    }


def overtime_risk(pred_margin):
    """
    Estimate overtime probability from predicted margin.

    Args:
        pred_margin: predicted home score - away score

    Returns:
        dict with risk level, probability estimate, and description
    """
    abs_margin = abs(pred_margin)

    if abs_margin <= 1:
        return {
            "level": "VERY HIGH",
            "probability": "~15%",
            "description": "Essentially a coin flip -- high OT risk",
        }
    elif abs_margin <= 3:
        return {
            "level": "HIGH",
            "probability": "~10%",
            "description": "Very tight game -- elevated OT risk",
        }
    elif abs_margin <= 5:
        return {
            "level": "MODERATE",
            "probability": "~6%",
            "description": "Close game -- moderate OT risk",
        }
    elif abs_margin <= 8:
        return {
            "level": "LOW",
            "probability": "~3%",
            "description": "Comfortable lead expected",
        }
    else:
        return {
            "level": "MINIMAL",
            "probability": "<2%",
            "description": "Clear favorite -- OT unlikely",
        }


def blowout_risk(home_streak=0, away_streak=0, strength_diff=0,
                 pred_margin=0, home_momentum=0, away_momentum=0):
    """
    Assess risk of a blowout (>20 pt margin).

    Args:
        home_streak: home team streak (positive=winning)
        away_streak: away team streak
        strength_diff: home strength - away strength
        pred_margin: predicted margin
        home_momentum: home team momentum score
        away_momentum: away team momentum score

    Returns:
        dict with risk level, favored team, and description
    """
    risk_score = 0
    signals = []

    # Large predicted margin
    if abs(pred_margin) >= 12:
        risk_score += 3
        signals.append(f"Large predicted margin ({pred_margin:+.1f})")

    # Extreme streaks
    if abs(home_streak) >= 7:
        risk_score += 2
        signals.append(f"Home on {'winning' if home_streak > 0 else 'losing'} "
                       f"streak ({home_streak:+d})")
    if abs(away_streak) >= 7:
        risk_score += 2
        signals.append(f"Away on {'winning' if away_streak > 0 else 'losing'} "
                       f"streak ({away_streak:+d})")

    # Large strength gap
    if abs(strength_diff) > 0.25:
        risk_score += 2
        signals.append(f"Significant strength gap ({strength_diff:+.2f})")

    # Momentum divergence
    momentum_gap = abs(home_momentum - away_momentum)
    if momentum_gap > 5:
        risk_score += 1
        signals.append(f"Wide momentum gap ({momentum_gap:.1f})")

    # Determine level
    if risk_score >= 6:
        level = "HIGH"
        desc = "??  Blowout likely -- consider avoiding totals bets"
    elif risk_score >= 4:
        level = "MODERATE"
        desc = "Potential for lopsided game"
    elif risk_score >= 2:
        level = "LOW"
        desc = "Competitive game expected"
    else:
        level = "MINIMAL"
        desc = "Evenly matched -- blowout unlikely"

    favored = "HOME" if pred_margin > 0 else "AWAY"

    return {
        "level": level,
        "risk_score": risk_score,
        "favored": favored,
        "description": desc,
        "signals": signals,
    }
