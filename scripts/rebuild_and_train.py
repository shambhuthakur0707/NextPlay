#!/usr/bin/env python3
"""Run a controlled full rebuild and training experiment with playoffs included and downweighted."""
from __future__ import annotations
import traceback

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.full_rebuild import full_rebuild
from config import WEIGHT_PLAYOFF_ROWS


def main():
    try:
        print("Running full_rebuild with skip_api=True and include_playoffs=True (local CSVs)...")
        model_df, result = full_rebuild(skip_api=True, run_model_optimization=False, include_playoffs=True)
        print("\nRebuild + training finished. Results summary:")
        if result is None:
            print("No result returned from full_rebuild")
            return 1
        print(f"Games in model_df: {len(model_df)}")
        if 'mae_total' in result:
            print(f"Total MAE: {result['mae_total']:.2f}")
        else:
            print('No MAE in result dict')
        return 0
    except Exception as exc:
        print('Error during rebuild_and_train:')
        traceback.print_exc()
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
