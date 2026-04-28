#!/usr/bin/env python3
"""Pull fresh playoff data and train playoff models."""
from __future__ import annotations
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GAMELOGS_ALL_PATH
from pipelines.full_rebuild import full_rebuild
from models.train import train_playoff_models, save_playoff_models


def main():
    try:
        # Remove old gamelogs file to force fresh pull
        if os.path.exists(GAMELOGS_ALL_PATH):
            print(f"Removing old gamelogs: {GAMELOGS_ALL_PATH}")
            os.remove(GAMELOGS_ALL_PATH)
            print("[OK] Old gamelogs removed\n")
        
        print("=" * 55)
        print("STEP 1: PULL FRESH DATA WITH PLAYOFFS")
        print("=" * 55)
        
        model_df, result = full_rebuild(
            skip_api=False,  # Pull fresh from API
            run_model_optimization=False, 
            include_playoffs=True,
            include_market_lines=False
        )
        
        print("\n" + "=" * 55)
        print("DATASET SUMMARY")
        print("=" * 55)
        print(f"Total games: {len(model_df)}")
        
        if "IS_PLAYOFF" in model_df.columns:
            playoff_count = model_df["IS_PLAYOFF"].sum()
            regular_count = len(model_df) - playoff_count
            print(f"Playoff games: {playoff_count}")
            print(f"Regular season games: {regular_count}")
        else:
            print("[WARN] IS_PLAYOFF column not found in model_df")
            return 1
        
        if playoff_count == 0:
            print("\n[WARN] No playoff games were found in the pulled data.")
            print("This may be because it's not playoff season yet.")
            return 0
        
        print("\n" + "=" * 55)
        print("STEP 2: TRAIN PLAYOFF MODELS")
        print("=" * 55)
        
        playoff_result = train_playoff_models(model_df)
        
        if playoff_result is None:
            print("[FAIL] Could not train playoff models.")
            return 1
        
        print("\nSaving playoff models...")
        save_playoff_models(playoff_result)
        
        print("\n" + "=" * 55)
        print("[OK] COMPLETE")
        print("=" * 55)
        return 0
        
    except Exception as exc:
        import traceback
        print("Error:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
