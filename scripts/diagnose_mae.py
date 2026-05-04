# -*- coding: utf-8 -*-
"""Diagnostic script: why is total MAE stuck at ~15?"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")

from config import FEATURE_COLS_FINAL, STACKED_TOTAL_FEATURES

df = pd.read_csv("data/model_ready_final.csv")

# ── Market feature analysis ─────────────────────────────────────
print("=" * 60)
print("MARKET FEATURE ANALYSIS")
print("=" * 60)
for col in ["MARKET_TOTAL_LINE", "MARKET_HOME_LINE"]:
    if col not in df.columns:
        print(f"{col}: NOT IN CSV")
        continue
    print(f"\n{col}:")
    print(f"  mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    print(f"  min={df[col].min():.2f}, max={df[col].max():.2f}")
    print(f"  nunique={df[col].nunique()}")
    print(f"  first 10: {df[col].head(10).tolist()}")
    print(f"  last 10:  {df[col].tail(10).tolist()}")

# Check if market is just a copy of another feature
mtl = df["MARKET_TOTAL_LINE"]
for ref_col in ["EXPECTED_TOTAL", "EWM_EXPECTED_TOTAL", "SOS_EXPECTED_TOTAL",
                 "COMBINED_PTS_ROLL10", "TOTAL_PTS"]:
    corr_val = mtl.corr(df[ref_col])
    diff = (mtl - df[ref_col]).abs()
    print(f"\n  vs {ref_col}:")
    print(f"    corr = {corr_val:.4f}")
    print(f"    mean_diff = {diff.mean():.4f}, max_diff = {diff.max():.4f}")
    exact_match = (diff < 0.001).sum()
    print(f"    exact matches = {exact_match} / {len(df)}")

# ── PTS_MARGIN check ────────────────────────────────────────────
print("\n" + "=" * 60)
print("GARBAGE TIME / OT FILTER CHECK")
print("=" * 60)
if "PTS_MARGIN" in df.columns:
    blowouts = (df["PTS_MARGIN"].abs() > 25).sum()
    ot_suspect = ((df["TOTAL_PTS"] > 240) & (df["PTS_MARGIN"].abs() < 10)).sum()
    print(f"PTS_MARGIN present: yes")
    print(f"Blowouts (>25pt):     {blowouts} ({blowouts/len(df):.1%})")
    print(f"Likely OT games:      {ot_suspect} ({ot_suspect/len(df):.1%})")
    print(f"Would be removed:     {blowouts + ot_suspect} ({(blowouts + ot_suspect)/len(df):.1%})")
else:
    print("PTS_MARGIN: NOT IN CSV — filter can't run during training!")

# ── Naive baselines ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("NAIVE BASELINES (in-sample, full dataset)")
print("=" * 60)
mean_total = df["TOTAL_PTS"].mean()
baselines = {
    f"Predict mean ({mean_total:.0f})": mean_total,
    "EXPECTED_TOTAL":      df["EXPECTED_TOTAL"],
    "EWM_EXPECTED_TOTAL":  df["EWM_EXPECTED_TOTAL"],
    "COMBINED_PTS_ROLL10": df["COMBINED_PTS_ROLL10"],
    "SOS_EXPECTED_TOTAL":  df["SOS_EXPECTED_TOTAL"],
    "MARKET_TOTAL_LINE":   df["MARKET_TOTAL_LINE"],
}
for name, pred in baselines.items():
    mae = np.abs(df["TOTAL_PTS"] - pred).mean()
    print(f"  {name:30s}  MAE = {mae:.2f}")

# ── Time-split baseline ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TIME-SPLIT BASELINES (80/20)")
print("=" * 60)
df = df.sort_values("GAME_DATE").reset_index(drop=True)
split = int(len(df) * 0.80)
train = df.iloc[:split]
test = df.iloc[split:]
print(f"Train: {len(train)}, Test: {len(test)}")

# Mean baseline
train_mean = train["TOTAL_PTS"].mean()
mae_mean = np.abs(test["TOTAL_PTS"] - train_mean).mean()
print(f"  Predict train mean ({train_mean:.0f}):  MAE = {mae_mean:.2f}")

# Feature-based baselines on test set
for col in ["EXPECTED_TOTAL", "EWM_EXPECTED_TOTAL", "COMBINED_PTS_ROLL10",
            "SOS_EXPECTED_TOTAL", "MARKET_TOTAL_LINE"]:
    mae = np.abs(test["TOTAL_PTS"] - test[col]).mean()
    print(f"  Predict {col:30s}  MAE = {mae:.2f}")

# ── Quick RF model on test set ──────────────────────────────────
print("\n" + "=" * 60)
print("QUICK MODEL TEST (80/20 split)")
print("=" * 60)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

feature_cols = [c for c in FEATURE_COLS_FINAL if c in df.columns]

X_train = train[feature_cols].fillna(0)
X_test = test[feature_cols].fillna(0)

# Direct TOTAL_PTS model (not stacked)
rf = RandomForestRegressor(n_estimators=400, max_depth=9,
                           min_samples_leaf=15, random_state=42, n_jobs=-1)
rf.fit(X_train, train["TOTAL_PTS"])
pred_total = rf.predict(X_test)
mae_direct = mean_absolute_error(test["TOTAL_PTS"], pred_total)
print(f"  Direct RF -> TOTAL_PTS: MAE = {mae_direct:.2f}")

# Home + Away separate, then sum
rf_h = RandomForestRegressor(n_estimators=400, max_depth=9,
                              min_samples_leaf=15, random_state=42, n_jobs=-1)
rf_h.fit(X_train, train["HOME_PTS"])
pred_h = rf_h.predict(X_test)
mae_h = mean_absolute_error(test["HOME_PTS"], pred_h)

rf_a = RandomForestRegressor(n_estimators=400, max_depth=9,
                              min_samples_leaf=15, random_state=42, n_jobs=-1)
rf_a.fit(X_train, train["AWAY_PTS"])
pred_a = rf_a.predict(X_test)
mae_a = mean_absolute_error(test["AWAY_PTS"], pred_a)

pred_sum = pred_h + pred_a
mae_sum = mean_absolute_error(test["TOTAL_PTS"], pred_sum)
print(f"  RF Home:  MAE = {mae_h:.2f}")
print(f"  RF Away:  MAE = {mae_a:.2f}")
print(f"  Home+Away sum: MAE = {mae_sum:.2f}")

# ── Feature importance for direct total model ────────────────────
print("\n" + "=" * 60)
print("TOP 20 FEATURE IMPORTANCES (direct total model)")
print("=" * 60)
imp = pd.Series(rf.feature_importances_, index=feature_cols)
imp = imp.sort_values(ascending=False)
for i, (feat, val) in enumerate(imp.head(20).items()):
    print(f"  {i+1:2d}. {feat:35s} {val:.4f}")

# ── Check target noise floor ────────────────────────────────────
print("\n" + "=" * 60)
print("IRREDUCIBLE NOISE ESTIMATE")
print("=" * 60)
# For same matchup, how variable is TOTAL_PTS?
df["MATCHUP"] = df["HOME_TEAM"] + "_" + df["AWAY_TEAM"]
matchup_std = df.groupby("MATCHUP")["TOTAL_PTS"].agg(["std", "count"])
matchup_std = matchup_std[matchup_std["count"] >= 3]
print(f"Matchups with 3+ games: {len(matchup_std)}")
print(f"Avg within-matchup std:  {matchup_std['std'].mean():.1f}")
print(f"This means even a PERFECT model can't get below ~{matchup_std['std'].mean() * 0.8:.1f} MAE")
print(f"(assuming roughly normal errors, MAE ≈ 0.8 * std)")
