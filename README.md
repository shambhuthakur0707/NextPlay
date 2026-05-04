# NextPlay - NBA AI Score Predictor

NextPlay is an end-to-end NBA score prediction system with data ingestion, feature engineering, model training, live prediction, and backtesting workflows.

Current codebase status:
- Model version: V9
- Base model feature set: 155 engineered features (down from 160 after audit fixes)
- Total-points model: stacked meta-model (18 meta features)
- Live prediction supports injury adjustments and optional sportsbook-total blending
- Source of truth for version/features: `config.py` (`MODEL_VERSION`, `FEATURE_COLS_FINAL`, `STACKED_TOTAL_FEATURES`)

---

## ⚠️ CURRENT STATE & WHERE TO CONTINUE ⚠️

> **Read this section first if resuming work on this project.**

### What Was Done (May 2026 Session)

A full leakage audit was run against `model_ready_final.csv` (3,642 rows, 151 columns).
The audit revealed the following issues, all of which have been fixed:

#### Fixes Applied

**1. `config.py` — 4 changes**
- Removed `HOME_TRAVEL_DIST` from `REST_STREAK_FEATURES` — it was hardcoded to `0.0`
  in `rest_streak.py` (home team never travels), making it a constant useless feature.
- Removed `MODEL_TOTAL_VS_MARKET` and `MODEL_MARGIN_VS_MARKET` from `MARKET_FEATURES` —
  these require a trained model to compute and cannot exist in training data.
  They were causing silent NaN columns across all 3,642 rows.
- Removed `IS_PLAYOFF` from `META_COLS` — it was already in `PLAYOFF_FEATURES`,
  causing duplicate columns in the final DataFrame selection in `pipeline.py`.
- Trimmed `STACKED_TOTAL_FEATURES` — removed `PLAYOFF_HOME_BOOST`, `IS_ELIMINATION`,
  `IS_CLOSEOUT` from Model C meta-features because they were confirmed missing from
  the CSV per the audit, causing Model C to silently train on NaN meta-features.

**2. `features/ewm.py` — full rewrite**
- The old version grouped by `HOME_TEAM` / `AWAY_TEAM` slot, meaning each team's EWM
  was computed from only ~half their games (home games for HOME_EWM_*, away for AWAY_EWM_*).
- Rewrote to use a unified team-history table (same pattern `rolling.py` already used),
  computing `shift(1).ewm()` across all games then joining back by `GAME_ID`.

**3. `features/playoff.py` — vectorized intensity**
- Replaced `_compute_intensity`'s row-by-row `.iloc[idx].get()` loop with fully
  vectorized numpy. The old version was ~100x slower and `.get()` on a pandas Series
  raises `KeyError` on missing columns rather than returning the default.

**4. `features/rest_streak.py` — travel distance fix**
- `TRAVEL_DIFF` was computed as `HOME_TRAVEL_DIST - AWAY_TRAVEL_DIST` but
  `HOME_TRAVEL_DIST` was removed. Fixed to: `TRAVEL_DIFF = -AWAY_TRAVEL_DIST`.

**5. `fast_optimize_total.py` — leaned down**
- Removed slow GB-500, GB-800, RF-600, RF-800 candidates.
- Reduced walk-forward window from 800 to 400, step from 50 to 100.
- Now runs in under 10 minutes on 8GB RAM.

#### Current MAE After Fixes

After rebuild with fixes applied:
- Walk-forward Total MAE: **~15.3** (RF-400 baseline, LightGBM matched it)
- This MAE is NOT yet meaningful improvement — see root cause below.

#### Root Cause of Stubbornly High MAE — Market Features Are Fake

The Odds API free plan only returns **current/upcoming** odds, not historical.
When the pipeline merged market features onto 3,642 games, only 8 rows (last 2 days)
had real data. The remaining 3,634 rows were filled with synthetic/interpolated values.

**Current workaround:** `MARKET_FEATURES = []` in `config.py` (market features disabled).
A model trained with fake market data performs worse than one trained without them.

---

### ⏭️ NEXT STEPS — Resume From Here

#### STEP 1 (Immediate) — Get Real Historical Odds Data

This is the single highest-leverage remaining task. Real market lines
(`MARKET_TOTAL_LINE`) typically reduce total MAE by 2-4 points on their own.

You need odds for 3 seasons: **2023-24, 2024-25, 2025-26** (~3,600 games).

**Option A — Basketball Reference (free, no key needed)**

Install lxml first:
```powershell
pip install lxml --break-system-packages
```

Then test if it works:
```python
import pandas as pd
url = 'https://www.basketball-reference.com/leagues/NBA_2024_games.html'
tables = pd.read_html(url)
print(tables[0].columns.tolist())
print(tables[0].head())
```

Basketball Reference game pages include Vegas lines in the Notes column.
Pull all 3 seasons (NBA_2024, NBA_2025, NBA_2026) and write an ingestion
script to parse and merge into the market pipeline.

**Option B — Kaggle dataset**

Search kaggle.com for: `"NBA betting lines historical"` or `"NBA odds totals spreads"`
Look for datasets covering 2023-2026 with columns: `date, home_team, away_team, total_line, spread`
Download and drop into `data/` folder, then write a merge script.

**Option C — Log going forward (long-term)**

Run this daily before games tip off (~11am ET) to build a real historical dataset
over the next season:
```powershell
$env:ODDS_API_KEY = "4a74120e462f25bb4889a0e7a0edada3"
$env:PYTHONPATH = "C:\Users\Shambhu\NextPlay"
python -c "
from ingestion.odds import fetch_odds
import pandas as pd
from config import MARKET_LINES_PATH
df = fetch_odds()
existing = pd.read_csv(MARKET_LINES_PATH)
combined = pd.concat([existing, df]).drop_duplicates()
combined.to_csv(MARKET_LINES_PATH, index=False)
print(f'Saved {len(df)} new lines. Total: {len(combined)}')
"
```

#### STEP 2 — Re-enable Market Features After Getting Real Data

Once you have a real historical odds CSV, update `config.py`:
```python
MARKET_FEATURES = [
    "MARKET_TOTAL_LINE", "MARKET_HOME_LINE",
    "MARKET_TOTAL_MOVE", "MARKET_SPREAD_MOVE",
    "MARKET_HOME_IMPLIED", "MARKET_AWAY_IMPLIED",
]
```

Then rebuild:
```powershell
$env:ODDS_API_KEY = "4a74120e462f25bb4889a0e7a0edada3"
$env:PYTHONPATH = "C:\Users\Shambhu\NextPlay"
python -m pipelines.full_rebuild
```

#### STEP 3 — Re-run Leakage Audit

```powershell
$env:PYTHONPATH = "C:\Users\Shambhu\NextPlay"
python scripts/run_checks.py
```

Target after market features added:
| Metric | Current | Target |
|---|---|---|
| Missing features | ~2 | 0 |
| Chrono MAE | 15.3 | ~11–13 |
| Row-shuffled MAE | 15.3 (same) | ~16–17 (higher = good) |
| Feature-shuffled MAE | ~16 | ~20+ |

The gap between chronological and row-shuffled MAE is the key signal.
Right now they are identical, meaning the model is not learning time-ordered patterns.
This should open up once real market features are present.

#### STEP 4 — Restore Playoff Meta-Features to Model C

After confirming playoff columns are present in the rebuilt CSV:
```python
python -c "
import pandas as pd
df = pd.read_csv('data/model_ready_final.csv')
cols = ['PLAYOFF_HOME_BOOST','PLAYOFF_ROAD_PENALTY','IS_ELIMINATION',
        'IS_CLOSEOUT','SERIES_GAME_NUM','PLAYOFF_INTENSITY']
print(df[cols].describe())
print('Missing:', df[cols].isnull().sum())
"
```

If all non-null, restore in `config.py`:
```python
STACKED_TOTAL_FEATURES = [
    "PRED_HOME", "PRED_AWAY", "PRED_SUM", "PRED_MARGIN",
    "HOME_ROLL10_PTS", "AWAY_ROLL10_PTS",
    "HOME_DEF_ROLL10", "AWAY_DEF_ROLL10",
    "HOME_ELO", "AWAY_ELO", "ELO_DIFF",
    "COMBINED_PTS_ROLL10",
    "IS_PLAYOFF",
    "PLAYOFF_INTENSITY",
    "PLAYOFF_HOME_BOOST",   # re-added after confirmation
    "IS_ELIMINATION",       # re-added after confirmation
    "IS_CLOSEOUT",          # re-added after confirmation
]
```

#### STEP 5 — Run Optimizer and Backtest

```powershell
python fast_optimize_total.py
python pipelines/backtest.py
python pipelines/market_backtest.py
```

Target total MAE: **< 13.0** with real market features.
LightGBM is expected to win over RF once features have real signal.

#### STEP 6 — Install XGBoost and CatBoost (Optional)

Currently skipped because not installed:
```powershell
pip install xgboost catboost --break-system-packages
```

Then rerun optimizer to include XGB and CAT candidates.

---

## Environment Setup

### Always run these before any pipeline command

```powershell
$env:ODDS_API_KEY = "4a74120e462f25bb4889a0e7a0edada3"
$env:PYTHONPATH = "C:\Users\Shambhu\NextPlay"
```

To make permanent (runs automatically in every new PowerShell session):
```powershell
Add-Content $PROFILE "`n`$env:ODDS_API_KEY = '4a74120e462f25bb4889a0e7a0edada3'"
Add-Content $PROFILE "`n`$env:PYTHONPATH = 'C:\Users\Shambhu\NextPlay'"
```

### Always run pipelines as modules (not direct file paths)

```powershell
# CORRECT
python -m pipelines.full_rebuild

# WRONG -- causes ModuleNotFoundError: No module named 'config'
python pipelines/full_rebuild.py
```

---

## What Is Implemented

- Multi-season ingestion from `nba_api` for team game logs (2023-24 to 2025-26)
- Feature pipeline with rolling form, matchup, rest/travel, context, defensive form,
  EWM momentum, SoS/ratings, market lines, player impact, margin, and ELO
- Three-model prediction stack:
    - Model A: home score
    - Model B: away score
    - Model C: stacked total model using predictions from A/B + context meta features
- Garbage-time filtering in training/backtests:
    - Removes blowouts over 25 points
    - Filters likely OT outliers with rule-based thresholds
- Live predictor with:
    - Dynamic feature-row building for a matchup date
    - Optional injury adjustment from player-impact table
    - Optional market total blending (`MARKET_TOTAL_BLEND_WEIGHT = 0.65`)
- Streamlit dashboard for:
    - Today slate view
    - Manual matchup prediction
    - Model metrics view
    - Team rankings

---

## Feature Coverage (From `config.py`)

`FEATURE_COLS_FINAL` currently contains **155 base features**:

| Category | Count | Notes |
|---|---:|---|
| Rolling home | 18 | |
| Rolling away | 17 | |
| Combined rolling | 6 | |
| Matchup history | 3 | |
| Rest, travel, streak | 8 | HOME_TRAVEL_DIST removed (was constant 0) |
| Context | 5 | |
| Defensive | 4 | |
| Shot profile | 17 | |
| EWM momentum | 12 | Rewritten to use unified team history |
| SoS and adjusted ratings | 23 | |
| Market lines | 0 | Disabled -- awaiting real historical odds |
| Player impact | 20 | |
| Margin rolling | 2 | |
| ELO | 4 | |
| Playoff | 12 | Vectorized, leakage-safe |
| **Total** | **155** | |

Model C (`STACKED_TOTAL_FEATURES`) uses **18 meta features**.
`PLAYOFF_HOME_BOOST`, `IS_ELIMINATION`, `IS_CLOSEOUT` temporarily removed from
Model C meta-features pending confirmation they are present in rebuilt CSV.

---

## Setup

### 1) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install lxml --break-system-packages
```

### 2) Set environment variables

```powershell
$env:ODDS_API_KEY = "4a74120e462f25bb4889a0e7a0edada3"
$env:PYTHONPATH = "C:\Users\Shambhu\NextPlay"
```

---

## Run

### Streamlit dashboard

```bash
streamlit run app.py
```

### Quick CLI prediction

```bash
python -c "from models.train import load_models; from prediction.predict import predict_game; import pandas as pd; from config import MODEL_READY_PATH; models=load_models(); df=pd.read_csv(MODEL_READY_PATH); df['GAME_DATE']=pd.to_datetime(df['GAME_DATE']); print(predict_game('BOS','LAL',df,models,verbose=False))"
```

---

## Pipelines And Utilities

### Full rebuild (data -> features -> training)

```powershell
python -m pipelines.full_rebuild
```

Programmatic usage:
```python
from pipelines.full_rebuild import full_rebuild
model_df, result = full_rebuild(skip_api=True)
```

### Playoff model workflow

Two model bundles exist:
- Regular: `model_A_home.pkl`, `model_B_away.pkl`, `model_C_total.pkl`
- Playoff: `model_A_playoff_home.pkl`, `model_B_playoff_away.pkl`, `model_C_playoff_total.pkl`

Runtime switching controlled by `USE_PLAYOFF_MODELS`, `PLAYOFF_SEASON_START_MONTH`,
`PLAYOFF_SEASON_END_MONTH` in `config.py`. During April-June, playoff models are
selected automatically when available, falling back to regular models if missing.

```bash
python scripts/fresh_pull_and_train_playoff.py
python scripts/smoke_model_switch.py
```

### Nightly update

```python
import pandas as pd
from config import MODEL_READY_PATH, SHOT_PROFILES_PATH, PLAYER_IMPACT_PATH
from models.train import load_models
from pipelines.nightly import nightly_update

models = load_models()
model_df = pd.read_csv(MODEL_READY_PATH)
shot_df = pd.read_csv(SHOT_PROFILES_PATH)
player_impact_df = pd.read_csv(PLAYER_IMPACT_PATH)

nightly_update(
    model_df=model_df,
    models=models,
    shot_df=shot_df,
    player_impact_df=player_impact_df,
)
```

### Backtests

```bash
python pipelines/backtest.py
python pipelines/market_backtest.py
```

### Model optimization

```bash
python fast_optimize_total.py
```

### Quality checks

```bash
python scripts/run_checks.py
python -m unittest tests/test_metadata_sync.py
python -m unittest tests/test_model_switching.py
```

### Enable pre-push checks

```powershell
./scripts/install_git_hooks.ps1
```

---

## Known Issues / Technical Debt

| Issue | Status | Fix |
|---|---|---|
| Market features contain fake data (only 8 real rows) | OPEN | Source historical odds from Basketball Reference or Kaggle |
| `HOME_TRAVEL_DIST` was constant 0 | FIXED | Removed from feature set |
| EWM computed per home/away slot (half data per team) | FIXED | Unified team history |
| `IS_PLAYOFF` duplicate in META_COLS + PLAYOFF_FEATURES | FIXED | Removed from META_COLS |
| `MODEL_TOTAL_VS_MARKET` in training features (inference-only) | FIXED | Removed from MARKET_FEATURES |
| Model C training on NaN playoff meta-features | FIXED | Trimmed STACKED_TOTAL_FEATURES |
| XGBoost / CatBoost not installed | OPEN | `pip install xgboost catboost` |
| Temporal shuffle MAE gap = 0 (model not learning patterns) | OPEN | Will resolve once real market features added |

---

## Project Layout

```text
NextPlay/
    app.py
    config.py
    optimize.py
    fast_optimize_total.py
    requirements.txt
    data/
        model_ready_final.csv       -- 3,642 games, 157 features
        market_lines.csv            -- WARNING: only 8 real rows (last 2 days)
        gamelogs_raw.csv
        gamelogs_all.csv
        games_all.csv
        shot_profiles.csv
        player_impact_true.csv
        elo_ratings.csv
    ingestion/
        gamelogs.py
        odds.py
        players.py
        shots.py
    features/
        pipeline.py
        rolling.py                  -- uses unified team history, shift(1) correct
        matchup.py
        rest_streak.py              -- HOME_TRAVEL_DIST removed, TRAVEL_DIFF fixed
        context.py
        defensive.py
        ewm.py                      -- REWRITTEN: unified team history
        sos.py
        market.py
        players.py
        elo.py
        shots.py
        playoff.py                  -- FIXED: vectorized _compute_intensity
    models/
        train.py
        stacking.py
        evaluate.py
    prediction/
        feature_builder.py
        predict.py
        logger.py
        confidence.py
    pipelines/
        full_rebuild.py
        nightly.py
        backtest.py
        market_backtest.py
    utils/
        helpers.py
```

---

## Core Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- lightgbm
- lxml
- nba_api
- plotly
- requests