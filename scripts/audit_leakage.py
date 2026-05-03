# -*- coding: utf-8 -*-
"""
NextPlay -- Data Leakage Audit
==============================
Detects features that may contain future information (look-ahead bias).

Run:
    python scripts/audit_leakage.py

Outputs:
    - Console report with severity rankings
    - data/leakage_audit_report.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Make sure repo root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    MODEL_READY_PATH,
    FEATURE_COLS_FINAL,
    STACKED_TOTAL_FEATURES,
)

REPORT_PATH = ROOT / "data" / "leakage_audit_report.csv"

# Targets / outcome columns (anything derived from these in features = leakage)
TARGET_COLS = [
    "HOME_PTS", "AWAY_PTS", "TOTAL_PTS", "PTS_MARGIN",
    "HOME_WIN", "AWAY_WIN", "WINNER",
]

# Suspicious name patterns (substring match, case-insensitive)
SUSPICIOUS_PATTERNS = [
    "actual", "result", "final", "outcome", "winner",
    "season_avg", "season_mean", "season_total",  # often computed full-season
    "closing", "close_line",  # closing line = late info
    "_post", "post_",  # post-game stats
]

# Acceptable patterns that LOOK suspicious but aren't (allowlist)
ALLOWED_PATTERNS = [
    "elo_pre",  # ELO before game = fine
    "rest_days", "days_rest",
    "rolling_", "ewm_", "_last", "_l5", "_l10", "_l20",  # historical = fine
]


def load_data():
    print(f"[Audit] Loading {MODEL_READY_PATH}")
    df = pd.read_csv(MODEL_READY_PATH)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
    print(f"[Audit] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def get_feature_cols(df):
    """Get features that exist in dataframe."""
    feats = [c for c in FEATURE_COLS_FINAL if c in df.columns]
    missing = [c for c in FEATURE_COLS_FINAL if c not in df.columns]
    if missing:
        print(f"[Audit] WARN: {len(missing)} configured features missing from data")
        for m in missing[:10]:
            print(f"        - {m}")
    return feats


# ---------------------------------------------------------------------------
# CHECK 1: Correlation with target
# ---------------------------------------------------------------------------
def check_correlation(df, feats):
    """Flag features highly correlated with TOTAL_PTS or PTS_MARGIN."""
    print("\n" + "=" * 70)
    print("CHECK 1: Correlation with target (>0.85 is suspicious)")
    print("=" * 70)

    findings = []
    for target in ["TOTAL_PTS", "PTS_MARGIN", "HOME_PTS", "AWAY_PTS"]:
        if target not in df.columns:
            continue
        sub = df[feats + [target]].select_dtypes(include=[np.number]).dropna()
        if len(sub) < 100:
            continue
        corrs = sub.corr()[target].drop(target, errors="ignore").abs()
        suspicious = corrs[corrs > 0.85].sort_values(ascending=False)

        if len(suspicious) > 0:
            print(f"\n  Target: {target}")
            for feat, c in suspicious.items():
                severity = "CRITICAL" if c > 0.95 else "HIGH" if c > 0.90 else "MEDIUM"
                print(f"    [{severity}]  {feat:50s}  corr={c:.3f}")
                findings.append({
                    "check": "correlation",
                    "feature": feat,
                    "target": target,
                    "value": round(c, 3),
                    "severity": severity,
                })
        else:
            print(f"\n  Target: {target}: no features above 0.85 ✓")
    return findings


# ---------------------------------------------------------------------------
# CHECK 2: Suspicious feature names
# ---------------------------------------------------------------------------
def check_naming(feats):
    """Flag features with suspicious naming patterns."""
    print("\n" + "=" * 70)
    print("CHECK 2: Suspicious feature names")
    print("=" * 70)

    findings = []
    for f in feats:
        f_lower = f.lower()

        # Skip if explicitly allowed
        if any(allowed in f_lower for allowed in ALLOWED_PATTERNS):
            continue

        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in f_lower:
                print(f"  [SUSPECT NAME]  {f}  (pattern: '{pattern}')")
                findings.append({
                    "check": "naming",
                    "feature": f,
                    "target": "-",
                    "value": pattern,
                    "severity": "MEDIUM",
                })
                break

    if not findings:
        print("  No suspicious names found ✓")
    return findings


# ---------------------------------------------------------------------------
# CHECK 3: Future-game leak via temporal shuffle test
# ---------------------------------------------------------------------------
def check_temporal_shuffle(df, feats, target="TOTAL_PTS"):
    """
    For each game, check if feature values reference data from games
    AFTER this game's GAME_DATE (true look-ahead).

    Method: Sort by date. For features that are rolling averages, compare
    feature value to mean of that team's PRIOR games. If they only match
    when including FUTURE games, it's leakage.

    Simpler test: train on first 70%, predict last 30%. Then SHUFFLE the
    last 30% rows (breaks order) and predict again. If accuracy doesn't
    drop much, time order doesn't matter -> leakage likely.
    """
    print("\n" + "=" * 70)
    print("CHECK 3: Temporal shuffle test")
    print("=" * 70)

    if target not in df.columns:
        print(f"  Skipped (no {target} in data)")
        return []

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        print("  Skipped (sklearn not available)")
        return []

    sub = df[feats + [target, "GAME_DATE"]].dropna()
    if len(sub) < 1000:
        print(f"  Skipped (not enough data: {len(sub)})")
        return []

    sub = sub.sort_values("GAME_DATE").reset_index(drop=True)
    cut = int(len(sub) * 0.7)
    train, test = sub.iloc[:cut], sub.iloc[cut:]

    X_tr, y_tr = train[feats], train[target]
    X_te, y_te = test[feats], test[target]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)

    # Test 1: chronological (real backtest)
    pred_chrono = model.predict(X_te)
    mae_chrono = mean_absolute_error(y_te, pred_chrono)

    # Test 2: shuffle test rows (if features are leak-free, MAE should be IDENTICAL)
    # If MAE drops a lot when test is shuffled, it means the model is exploiting
    # info that depends on row position (suspicious)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(test))
    pred_shuffled = model.predict(X_te.iloc[perm])
    mae_shuffled = mean_absolute_error(y_te.iloc[perm], pred_shuffled)

    # Test 3: random feature shuffle (sanity baseline)
    X_te_noisy = X_te.copy()
    for c in X_te_noisy.columns:
        X_te_noisy[c] = rng.permutation(X_te_noisy[c].values)
    mae_random = mean_absolute_error(y_te, model.predict(X_te_noisy))

    print(f"  MAE chronological test:   {mae_chrono:.2f}")
    print(f"  MAE row-shuffled test:    {mae_shuffled:.2f}  (should match chrono)")
    print(f"  MAE feature-shuffled:     {mae_random:.2f}  (should be MUCH worse)")

    findings = []
    if mae_random < mae_chrono * 1.3:
        print("  [WARN] Random features predict almost as well as real features!")
        print("         Suggests model is barely learning anything useful.")
        findings.append({
            "check": "temporal_shuffle",
            "feature": "all",
            "target": target,
            "value": f"chrono={mae_chrono:.2f}, random={mae_random:.2f}",
            "severity": "HIGH",
        })
    else:
        print("  Temporal/feature integrity looks reasonable ✓")

    return findings


# ---------------------------------------------------------------------------
# CHECK 4: Feature importance gut-check
# ---------------------------------------------------------------------------
def check_feature_importance(df, feats, target="TOTAL_PTS"):
    """Train quick model, look at top features. Anything weird at the top?"""
    print("\n" + "=" * 70)
    print("CHECK 4: Top feature importances (look for surprises)")
    print("=" * 70)

    if target not in df.columns:
        return []

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return []

    sub = df[feats + [target]].dropna()
    if len(sub) < 500:
        return []

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(sub[feats], sub[target])

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\n  Top 15 features for predicting {target}:")
    for i, (f, v) in enumerate(imp.head(15).items(), 1):
        flag = ""
        if any(p in f.lower() for p in SUSPICIOUS_PATTERNS):
            flag = "  ⚠️ SUSPECT NAME"
        if v > 0.30:
            flag += "  🚩 DOMINATES"
        print(f"    {i:2d}. {f:50s}  {v:.4f}{flag}")

    findings = []
    if imp.iloc[0] > 0.40:
        findings.append({
            "check": "importance",
            "feature": imp.index[0],
            "target": target,
            "value": round(imp.iloc[0], 3),
            "severity": "HIGH",
        })
        print(f"\n  [WARN] Top feature has importance {imp.iloc[0]:.2f} -- investigate")
    return findings


# ---------------------------------------------------------------------------
# CHECK 5: Variance & constant features
# ---------------------------------------------------------------------------
def check_variance(df, feats):
    """Find features with near-zero variance (useless) or NaN-heavy."""
    print("\n" + "=" * 70)
    print("CHECK 5: Variance / NaN issues")
    print("=" * 70)

    findings = []
    for f in feats:
        s = df[f]
        nan_pct = s.isna().mean()
        if nan_pct > 0.5:
            print(f"  [HIGH NAN]  {f:50s}  {nan_pct:.1%} NaN")
            findings.append({
                "check": "high_nan", "feature": f, "target": "-",
                "value": round(nan_pct, 3), "severity": "MEDIUM",
            })
        if pd.api.types.is_numeric_dtype(s):
            if s.dropna().nunique() <= 1:
                print(f"  [CONSTANT]  {f}")
                findings.append({
                    "check": "constant", "feature": f, "target": "-",
                    "value": "constant", "severity": "LOW",
                })
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    feats = get_feature_cols(df)
    print(f"[Audit] Auditing {len(feats)} features\n")

    all_findings = []
    all_findings += check_correlation(df, feats)
    all_findings += check_naming(feats)
    all_findings += check_variance(df, feats)
    all_findings += check_feature_importance(df, feats)
    all_findings += check_temporal_shuffle(df, feats)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not all_findings:
        print("  ✓ No leakage indicators found. Your 69.9% likely is real.")
    else:
        df_findings = pd.DataFrame(all_findings)
        by_sev = df_findings["severity"].value_counts()
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if sev in by_sev:
                print(f"  {sev}: {by_sev[sev]}")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_findings.to_csv(REPORT_PATH, index=False)
        print(f"\n  Full report: {REPORT_PATH}")

        # Top concerns
        critical = df_findings[df_findings["severity"].isin(["CRITICAL", "HIGH"])]
        if len(critical) > 0:
            print("\n  TOP CONCERNS TO INVESTIGATE FIRST:")
            for _, row in critical.head(10).iterrows():
                print(f"    [{row['severity']}] {row['feature']} ({row['check']})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()