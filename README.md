# 🏀 NextPlay — NBA AI Score Predictor

AI-powered NBA game score predictions using **84 engineered features** and ensemble RandomForest models.

## Features

| Category | Count | Examples |
|---|---|---|
| Rolling Form | 23 | Last 10 game averages (PTS, FG%, AST, TOV) |
| Rest & Momentum | 6 | Rest days, win/loss streaks |
| Context | 5 | Season stage, home court strength |
| Defensive | 4 | Points allowed rolling averages |
| Shot Profiles | 17 | 3PT rate, paint rate, PTS per shot |
| Advanced | 29 | EWM, Strength of Schedule, player impact |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# Or predict from CLI
python -c "
from prediction.predict import predict_game
from models.train import load_models
import pandas as pd
from config import MODEL_READY_PATH

models = load_models()
df = pd.read_csv(MODEL_READY_PATH)
predict_game('BOS', 'LAL', df, models)
"
```

## Project Structure

```
NextPlay/
├── app.py                 # Streamlit dashboard
├── config.py              # Constants, paths, feature lists
├── requirements.txt       # Dependencies
│
├── data/                  # CSV data + trained model pickles
├── ingestion/             # NBA API data pulling
│   ├── gamelogs.py        # Team game logs
│   ├── shots.py           # Shot chart profiles
│   └── players.py         # Player impact data
│
├── features/              # Feature engineering pipeline
│   ├── rolling.py         # 10-game rolling averages
│   ├── rest_streak.py     # Rest days + win streaks
│   ├── context.py         # Season stage, home court
│   ├── defensive.py       # Defensive rolling features
│   ├── shots.py           # Shot profile clash features
│   ├── ewm.py            # Exponential weighted averages
│   ├── sos.py            # Strength of Schedule
│   ├── players.py        # Player impact features
│   └── pipeline.py       # Orchestrator — runs all steps
│
├── models/                # Training + evaluation
│   ├── train.py           # Train RF models A/B/C
│   └── evaluate.py        # MAE evaluation, error analysis
│
├── prediction/            # Prediction engine
│   ├── predict.py         # predict_game() — core prediction
│   ├── logger.py          # Prediction logging + error tracking
│   └── confidence.py      # Confidence scoring, OT/blowout risk
│
├── pipelines/             # Automation
│   ├── nightly.py         # Nightly update pipeline
│   ├── full_rebuild.py    # Full pipeline rebuild
│   └── backtest.py        # Historical backtesting
│
└── utils/                 # Shared utilities
    └── helpers.py         # streak, style_label, win_prob
```

## Models

Three RandomForest models predict different targets:

- **Model A** — Home team score
- **Model B** — Away team score  
- **Model C** — Total points

Training includes garbage time filtering (removes blowouts >25pt and OT games) for cleaner signal.

## Pipelines

### Nightly Update
```bash
python -c "from pipelines.nightly import nightly_update; ..."
```
Pulls last night's results, compares vs predictions, predicts tonight's games.

### Full Rebuild
```bash
python pipelines/full_rebuild.py
```
Complete rebuild: API pulls → feature engineering → model training (~10 min).

### Backtesting
```bash
python pipelines/backtest.py
```
Walk-forward backtest across all historical data.

## Model Version History

| Version | Features | Total MAE | Key Addition |
|---|---|---|---|
| V1 | 37 | 15.27 | Rolling averages |
| V2 | 41 | 14.99 | Defensive features |
| V3 | 58 | 14.89 | Shot profiles |
| V4 | 70 | 14.82 | EWM momentum |
| V5 | 82 | 14.80 | Player impact |
| **V7** | **84** | **14.52** | **SoS + garbage time filter** |

## Tech Stack

- **ML**: scikit-learn RandomForest (400 trees)
- **Data**: nba_api (official NBA stats)
- **Dashboard**: Streamlit + Plotly
- **Features**: 84 engineered features (V7)
