#!/usr/bin/env python3
"""Rebuild dataset with playoffs included, then train playoff models."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.full_rebuild import full_rebuild
from models.train import train_playoff_models, save_playoff_models


def main():
    try:
        print("=" * 55)
        print("STEP 1: REBUILD WITH PLAYOFFS INCLUDED")
        print("=" * 55)
        
        model_df, result = full_rebuild(
            skip_api=True, 
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
            regular_count = (~model_df["IS_PLAYOFF"]).sum()
            print(f"Playoff games: {playoff_count}")
            print(f"Regular season games: {regular_count}")
        else:
            print("[WARN] IS_PLAYOFF column not found in model_df")
            return 1
        
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
