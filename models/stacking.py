# -*- coding: utf-8 -*-
"""
NextPlay -- Stacked Total Model Utilities
=========================================
Helpers for the total-points meta learner.
The total model consumes home/away predictions plus a small
set of context features so it stays tied to the base forecasts.
"""

import pandas as pd

from config import STACKED_TOTAL_FEATURES


def build_total_meta_features(base_features, pred_home, pred_away,
                              feature_cols=None):
    """Build the meta-feature matrix used by the stacked total model."""
    if not isinstance(base_features, pd.DataFrame):
        base_features = pd.DataFrame(base_features)

    meta = pd.DataFrame(index=base_features.index)
    meta["PRED_HOME"] = pred_home
    meta["PRED_AWAY"] = pred_away
    meta["PRED_SUM"] = meta["PRED_HOME"] + meta["PRED_AWAY"]
    meta["PRED_MARGIN"] = meta["PRED_HOME"] - meta["PRED_AWAY"]

    columns = feature_cols or STACKED_TOTAL_FEATURES
    for col in columns:
        if col in meta.columns:
            continue
        if col in base_features.columns:
            meta[col] = base_features[col]
        else:
            meta[col] = 0.0

    return meta[columns].copy()
