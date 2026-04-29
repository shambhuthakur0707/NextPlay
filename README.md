# NextPlay - NBA AI Score Predictor

NextPlay is an end-to-end NBA score prediction system with data ingestion, feature engineering, model training, live prediction, and backtesting workflows.

Current codebase status:
- Model version: V9
- Base model feature set: 160 engineered features
- Total-points model: stacked meta-model (17 meta features)
- Live prediction supports injury adjustments and optional sportsbook-total blending
- Source of truth for version/features: `config.py` (`MODEL_VERSION`, `FEATURE_COLS_FINAL`, `STACKED_TOTAL_FEATURES`)

## What Is Implemented

- Multi-season ingestion from `nba_api` for team game logs
- Feature pipeline with rolling form, matchup, rest/travel, context, defensive form, EWM momentum, SoS/ratings, market lines, player impact, margin, and ELO
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

## Feature Coverage (From `config.py`)

`FEATURE_COLS_FINAL` currently contains 160 base features:

| Category | Count |
|---|---:|
| Rolling home | 18 |
| Rolling away | 17 |
| Combined rolling | 6 |
| Matchup history | 3 |
| Rest, travel, streak | 9 |
| Context | 5 |
| Defensive | 4 |
| Shot profile | 17 |
| EWM momentum | 12 |
| SoS and adjusted ratings | 23 |
| Market lines | 8 |
| Player impact | 20 |
| Margin rolling | 2 |
| ELO | 4 |
| Playoff | 12 |
| **Total** | **160** |

The stacked total model uses `STACKED_TOTAL_FEATURES` (12 columns).

## Setup

### 1) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Optional market API configuration

Market ingestion and market-aware prediction require The Odds API key.

PowerShell example:

```powershell
$env:ODDS_API_KEY = "your_key_here"
# optional overrides
$env:ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
$env:ODDS_SPORT_KEY = "basketball_nba"
```

## Run

### Streamlit dashboard

```bash
streamlit run app.py
```

### Quick CLI prediction

```bash
python -c "from models.train import load_models; from prediction.predict import predict_game; import pandas as pd; from config import MODEL_READY_PATH; models=load_models(); df=pd.read_csv(MODEL_READY_PATH); df['GAME_DATE']=pd.to_datetime(df['GAME_DATE']); print(predict_game('BOS','LAL',df,models,verbose=False))"
```

## Pipelines And Utilities

### Full rebuild (data -> features -> training)

```bash
python pipelines/full_rebuild.py
```

Programmatic usage (example):

```python
from pipelines.full_rebuild import full_rebuild

# Uses existing CSVs, includes market lines if available, and runs optimization.
model_df, result = full_rebuild(skip_api=True)
```

### Playoff model workflow

You now have two model bundles:

- Regular bundle: `model_A_home.pkl`, `model_B_away.pkl`, `model_C_total.pkl`
- Playoff bundle: `model_A_playoff_home.pkl`, `model_B_playoff_away.pkl`, `model_C_playoff_total.pkl`

Training behavior:

- `full_rebuild(..., include_playoffs=True)` pulls both Regular Season and Playoffs.
- Regular models train as before on the configured train/test split.
- Playoff models train on the full dataset with playoff rows upweighted and playoff-specific features enabled.

Runtime switching behavior:

- Controlled by `USE_PLAYOFF_MODELS`, `PLAYOFF_SEASON_START_MONTH`, and `PLAYOFF_SEASON_END_MONTH` in `config.py`.
- During playoff months (default April-June), playoff models are selected when available.
- If playoff files are missing, logic falls back to regular models.

Train/retrain playoff models from fresh API data:

```bash
python scripts/fresh_pull_and_train_playoff.py
```

Quick smoke verification (artifact presence + switching):

```bash
python scripts/smoke_model_switch.py
```

### Nightly update (last-night recap + tonight predictions)

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
python optimize.py
python fast_optimize_total.py
```

### Metadata sanity checks

```bash
python -m unittest tests/test_metadata_sync.py
python -m unittest tests/test_model_switching.py
```

### Unified quality checks (local + CI)

```bash
python scripts/run_checks.py
```

### Enable automatic pre-push checks

PowerShell:

```powershell
./scripts/install_git_hooks.ps1
```

After this, every `git push` will run `scripts/run_checks.py` and block the push if checks fail.

## Project Layout

```text
NextPlay/
    app.py
    config.py
    optimize.py
    fast_optimize_total.py
    requirements.txt
    data/
    ingestion/
        gamelogs.py
        odds.py
        players.py
        shots.py
    features/
        pipeline.py
        rolling.py
        matchup.py
        rest_streak.py
        context.py
        defensive.py
        ewm.py
        sos.py
        market.py
        players.py
        elo.py
        shots.py
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

## Core Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- lightgbm
- nba_api
- plotly
- requests
