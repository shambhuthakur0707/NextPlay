# -*- coding: utf-8 -*-
"""
Patch PTS_MARGIN into existing CSV and retrain models with all fixes applied.
Avoids the slow full_rebuild pipeline.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

# ── Step 1: Patch PTS_MARGIN into CSV ──────────────────────────
print("=" * 55)
print("STEP 1: Patching PTS_MARGIN into CSV")
print("=" * 55)

df = pd.read_csv("data/model_ready_final.csv")
print(f"Shape: {df.shape}")
print(f"PTS_MARGIN in CSV: {'PTS_MARGIN' in df.columns}")

if "PTS_MARGIN" not in df.columns:
    df["PTS_MARGIN"] = df["HOME_PTS"] - df["AWAY_PTS"]
    df.to_csv("data/model_ready_final.csv", index=False)
    print(f"Added PTS_MARGIN. New shape: {df.shape}")
else:
    print("PTS_MARGIN already present, skipping.")

print(f"PTS_MARGIN: mean={df['PTS_MARGIN'].mean():.2f}, std={df['PTS_MARGIN'].std():.2f}")
blowouts = (df["PTS_MARGIN"].abs() > 25).sum()
print(f"Blowouts (>25pt): {blowouts} ({blowouts/len(df):.1%})")

# ── Step 2: Quick verification of config changes ──────────────
print()
print("=" * 55)
print("STEP 2: Verifying config changes")
print("=" * 55)

from config import FEATURE_COLS_FINAL, STACKED_TOTAL_FEATURES, META_COLS, MARKET_FEATURES

print(f"MARKET_FEATURES: {MARKET_FEATURES}")
print(f"FEATURE_COLS_FINAL count: {len(FEATURE_COLS_FINAL)}")
print(f"STACKED_TOTAL_FEATURES: {STACKED_TOTAL_FEATURES}")
print(f"PTS_MARGIN in META_COLS: {'PTS_MARGIN' in META_COLS}")

market_in_features = [f for f in FEATURE_COLS_FINAL if "MARKET" in f]
print(f"Market features in FEATURE_COLS_FINAL: {market_in_features}")

market_in_stacked = [f for f in STACKED_TOTAL_FEATURES if "MARKET" in f]
print(f"Market features in STACKED_TOTAL_FEATURES: {market_in_stacked}")

# ── Step 3: Train models with fixes ───────────────────────────
print()
print("=" * 55)
print("STEP 3: Training models (market removed, PTS_MARGIN active, OOF stacking)")
print("=" * 55)

from models.train import train_models, save_models

result = train_models(df, apply_filter=True)
save_models(result)

# ── Step 4: Compare with baseline ─────────────────────────────
print()
print("=" * 55)
print("STEP 4: Comparison with naive baselines")
print("=" * 55)

df_sorted = df.sort_values("GAME_DATE").reset_index(drop=True)
split = int(len(df_sorted) * 0.80)
test = df_sorted.iloc[split:]

mean_total = df_sorted.iloc[:split]["TOTAL_PTS"].mean()
mae_mean = np.abs(test["TOTAL_PTS"] - mean_total).mean()
print(f"  Predict mean ({mean_total:.0f}):  MAE = {mae_mean:.2f}")

for col in ["EXPECTED_TOTAL", "EWM_EXPECTED_TOTAL", "COMBINED_PTS_ROLL10"]:
    if col in test.columns:
        mae = np.abs(test["TOTAL_PTS"] - test[col]).mean()
        print(f"  Predict {col:30s}  MAE = {mae:.2f}")

print(f"\n  MODEL Total MAE: {result['mae_total']:.2f}")
print(f"  MODEL Home MAE:  {result['mae_home']:.2f}")
print(f"  MODEL Away MAE:  {result['mae_away']:.2f}")

improvement = mae_mean - result["mae_total"]
print(f"\n  Improvement over mean baseline: {improvement:.2f} pts")
