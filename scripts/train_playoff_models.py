#!/usr/bin/env python3
"""Train playoff-only models on playoff games from the feature dataset."""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config import MODEL_READY_PATH
from models.train import train_playoff_models, save_playoff_models


def main():
    try:
        print("Loading feature dataset...")
        model_df = pd.read_csv(MODEL_READY_PATH)
        model_df["GAME_DATE"] = pd.to_datetime(model_df["GAME_DATE"])

        print(f"Dataset loaded: {len(model_df)} games")
        playoff_count = model_df.get("IS_PLAYOFF", False).sum() if "IS_PLAYOFF" in model_df.columns else 0
        print(f"Playoff games found: {playoff_count}")

        print("\nTraining playoff-only models...")
        playoff_result = train_playoff_models(model_df)

        if playoff_result is None:
            print("[FAIL] Could not train playoff models (no playoff data).")
            return 1

        print("\nSaving playoff models...")
        save_playoff_models(playoff_result)

        print("\n" + "=" * 55)
        print("[OK] PLAYOFF MODEL TRAINING COMPLETE")
        print("=" * 55)
        return 0

    except Exception as exc:
        print("Error during playoff model training:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
