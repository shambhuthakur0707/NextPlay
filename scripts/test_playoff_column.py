#!/usr/bin/env python3
"""Test if IS_PLAYOFF column is preserved in the feature pipeline."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.full_rebuild import full_rebuild

try:
    print("Testing feature pipeline with include_playoffs=False...")
    model_df, result = full_rebuild(
        skip_api=True, 
        run_model_optimization=False, 
        include_playoffs=False,
        include_market_lines=False
    )
    
    print(f"\nResults:")
    print(f"  Total games: {len(model_df)}")
    print(f"  IS_PLAYOFF in columns: {'IS_PLAYOFF' in model_df.columns}")
    if "IS_PLAYOFF" in model_df.columns:
        playoff_count = model_df["IS_PLAYOFF"].sum()
        print(f"  Playoff games: {playoff_count}")
        print(f"  Regular season games: {len(model_df) - playoff_count}")
    else:
        print("  [WARN] IS_PLAYOFF column not found!")
    
    exit(0)
except Exception as exc:
    import traceback
    print("Error:")
    traceback.print_exc()
    exit(1)
