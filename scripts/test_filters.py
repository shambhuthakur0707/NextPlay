# Check OT filter thresholds
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

df = pd.read_csv("data/model_ready_final.csv")

# Current threshold analysis
for threshold in [235, 240, 245, 250, 255, 260]:
    for margin in [5, 8, 10]:
        n = ((df["TOTAL_PTS"] > threshold) & (df["PTS_MARGIN"].abs() < margin)).sum()
        pct = n / len(df) * 100
        print(f"  TOTAL>{threshold}, margin<{margin}: {n:4d} ({pct:.1f}%)")
    print()

# Just blowouts alone:
blowouts = (df["PTS_MARGIN"].abs() > 25).sum()
print(f"Blowouts >25pt: {blowouts} ({blowouts/len(df):.1%})")
print(f"Blowouts >30pt: {(df['PTS_MARGIN'].abs() > 30).sum()}")

# Retrain with relaxed OT threshold (250/5)
print()
print("=" * 55)
print("RETRAINING with tightened OT filter (>250pt, <5 margin)")
print("=" * 55)

from config import RF_PARAMS, FEATURE_COLS_FINAL, STACKED_TOTAL_FEATURES
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from models.stacking import build_total_meta_features

feature_cols = [c for c in FEATURE_COLS_FINAL if c in df.columns]
meta_feature_cols = STACKED_TOTAL_FEATURES

df = df.sort_values("GAME_DATE").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# Tighter filter: blowout > 25pt, OT > 250pt AND margin < 5
before = len(train)
train_clean = train[
    (train["PTS_MARGIN"].abs() <= 25) &
    ~((train["TOTAL_PTS"] > 250) & (train["PTS_MARGIN"].abs() < 5))
].copy()
after = len(train_clean)
print(f"Filter: {before} -> {after} (removed {before - after}, {(before-after)/before:.1%})")

train_clean = train_clean.dropna(subset=feature_cols)
test = test.dropna(subset=feature_cols)

X_train = train_clean[feature_cols]
X_test = test[feature_cols]
train_weights = np.ones(len(train_clean), dtype=float)

print(f"Training: {len(train_clean)} games")
print(f"Testing:  {len(test)} games")
print(f"Features: {len(feature_cols)}")

# Model A
rf_A = RandomForestRegressor(**RF_PARAMS)
rf_A.fit(X_train, train_clean["HOME_PTS"], sample_weight=train_weights)
mae_home = mean_absolute_error(test["HOME_PTS"], rf_A.predict(X_test))

# Model B
rf_B = RandomForestRegressor(**RF_PARAMS)
rf_B.fit(X_train, train_clean["AWAY_PTS"], sample_weight=train_weights)
mae_away = mean_absolute_error(test["AWAY_PTS"], rf_B.predict(X_test))

# Model C with OOF
oof_home = np.zeros(len(train_clean))
oof_away = np.zeros(len(train_clean))
kf = KFold(n_splits=5, shuffle=False)

for fold_train_idx, fold_val_idx in kf.split(X_train):
    fold_w = train_weights[fold_train_idx]
    
    rf_A_fold = RandomForestRegressor(**RF_PARAMS)
    rf_A_fold.fit(X_train.iloc[fold_train_idx], 
                  train_clean["HOME_PTS"].iloc[fold_train_idx],
                  sample_weight=fold_w)
    oof_home[fold_val_idx] = rf_A_fold.predict(X_train.iloc[fold_val_idx])
    
    rf_B_fold = RandomForestRegressor(**RF_PARAMS)
    rf_B_fold.fit(X_train.iloc[fold_train_idx],
                  train_clean["AWAY_PTS"].iloc[fold_train_idx],
                  sample_weight=fold_w)
    oof_away[fold_val_idx] = rf_B_fold.predict(X_train.iloc[fold_val_idx])

train_meta = build_total_meta_features(
    train_clean[feature_cols], oof_home, oof_away,
    feature_cols=meta_feature_cols,
)
test_meta = build_total_meta_features(
    test[feature_cols], rf_A.predict(X_test), rf_B.predict(X_test),
    feature_cols=meta_feature_cols,
)
rf_C = RandomForestRegressor(**RF_PARAMS)
rf_C.fit(train_meta, train_clean["TOTAL_PTS"], sample_weight=train_weights)
pred_total = rf_C.predict(test_meta)
mae_total = mean_absolute_error(test["TOTAL_PTS"], pred_total)

print()
print(f"Home MAE:  {mae_home:.2f}")
print(f"Away MAE:  {mae_away:.2f}")
print(f"Total MAE: {mae_total:.2f}")

# Also try NO OT filter at all, just blowouts
print()
print("=" * 55)
print("COMPARISON: Blowout-only filter (no OT filter)")
print("=" * 55)

train2 = df.iloc[:split_idx].copy()
train2 = train2[train2["PTS_MARGIN"].abs() <= 25].copy()
train2 = train2.dropna(subset=feature_cols)
X_train2 = train2[feature_cols]
w2 = np.ones(len(train2))

rf_A2 = RandomForestRegressor(**RF_PARAMS)
rf_A2.fit(X_train2, train2["HOME_PTS"], sample_weight=w2)

rf_B2 = RandomForestRegressor(**RF_PARAMS)
rf_B2.fit(X_train2, train2["AWAY_PTS"], sample_weight=w2)

oof_h2 = np.zeros(len(train2))
oof_a2 = np.zeros(len(train2))
kf2 = KFold(n_splits=5, shuffle=False)
for ti, vi in kf2.split(X_train2):
    rf_af = RandomForestRegressor(**RF_PARAMS)
    rf_af.fit(X_train2.iloc[ti], train2["HOME_PTS"].iloc[ti], sample_weight=w2[ti])
    oof_h2[vi] = rf_af.predict(X_train2.iloc[vi])
    
    rf_bf = RandomForestRegressor(**RF_PARAMS)
    rf_bf.fit(X_train2.iloc[ti], train2["AWAY_PTS"].iloc[ti], sample_weight=w2[ti])
    oof_a2[vi] = rf_bf.predict(X_train2.iloc[vi])

tm2 = build_total_meta_features(train2[feature_cols], oof_h2, oof_a2, feature_cols=meta_feature_cols)
tm2t = build_total_meta_features(test[feature_cols], rf_A2.predict(X_test), rf_B2.predict(X_test), feature_cols=meta_feature_cols)
rf_C2 = RandomForestRegressor(**RF_PARAMS)
rf_C2.fit(tm2, train2["TOTAL_PTS"], sample_weight=w2)

print(f"Filter: {split_idx} -> {len(train2)} (removed {split_idx - len(train2)}, blowouts only)")
print(f"Home MAE:  {mean_absolute_error(test['HOME_PTS'], rf_A2.predict(X_test)):.2f}")
print(f"Away MAE:  {mean_absolute_error(test['AWAY_PTS'], rf_B2.predict(X_test)):.2f}")
print(f"Total MAE: {mean_absolute_error(test['TOTAL_PTS'], rf_C2.predict(tm2t)):.2f}")

# Also try NO filter at all
print()
print("=" * 55)
print("COMPARISON: No filter at all")
print("=" * 55)

train3 = df.iloc[:split_idx].copy().dropna(subset=feature_cols)
X_train3 = train3[feature_cols]
w3 = np.ones(len(train3))

rf_A3 = RandomForestRegressor(**RF_PARAMS)
rf_A3.fit(X_train3, train3["HOME_PTS"])
rf_B3 = RandomForestRegressor(**RF_PARAMS)
rf_B3.fit(X_train3, train3["AWAY_PTS"])

oof_h3 = np.zeros(len(train3))
oof_a3 = np.zeros(len(train3))
kf3 = KFold(n_splits=5, shuffle=False)
for ti, vi in kf3.split(X_train3):
    rf_af = RandomForestRegressor(**RF_PARAMS)
    rf_af.fit(X_train3.iloc[ti], train3["HOME_PTS"].iloc[ti])
    oof_h3[vi] = rf_af.predict(X_train3.iloc[vi])
    rf_bf = RandomForestRegressor(**RF_PARAMS)
    rf_bf.fit(X_train3.iloc[ti], train3["AWAY_PTS"].iloc[ti])
    oof_a3[vi] = rf_bf.predict(X_train3.iloc[vi])

tm3 = build_total_meta_features(train3[feature_cols], oof_h3, oof_a3, feature_cols=meta_feature_cols)
tm3t = build_total_meta_features(test[feature_cols], rf_A3.predict(X_test), rf_B3.predict(X_test), feature_cols=meta_feature_cols)
rf_C3 = RandomForestRegressor(**RF_PARAMS)
rf_C3.fit(tm3, train3["TOTAL_PTS"])

print(f"No filter: {len(train3)} games")
print(f"Home MAE:  {mean_absolute_error(test['HOME_PTS'], rf_A3.predict(X_test)):.2f}")
print(f"Away MAE:  {mean_absolute_error(test['AWAY_PTS'], rf_B3.predict(X_test)):.2f}")
print(f"Total MAE: {mean_absolute_error(test['TOTAL_PTS'], rf_C3.predict(tm3t)):.2f}")
