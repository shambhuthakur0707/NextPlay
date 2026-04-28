#!/usr/bin/env python3
"""Smoke check playoff model artifacts and season-based model switching."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    MODEL_A_PLAYOFF_PATH,
    MODEL_B_PLAYOFF_PATH,
    MODEL_C_PLAYOFF_PATH,
)
from models.train import load_models, load_playoff_models
from pipelines.nightly import select_active_models


class _DummyModel:
    pass


def main() -> int:
    print("=== Smoke: Playoff Artifact Files ===")
    playoff_paths = [
        MODEL_A_PLAYOFF_PATH,
        MODEL_B_PLAYOFF_PATH,
        MODEL_C_PLAYOFF_PATH,
    ]
    missing = [p for p in playoff_paths if not Path(p).exists()]
    for p in playoff_paths:
        exists = Path(p).exists()
        print(f"{Path(p).name}: {'OK' if exists else 'MISSING'}")

    if missing:
        print("[FAIL] Missing playoff model files")
        return 1

    print("\n=== Smoke: Load Model Bundles ===")
    regular_models = load_models()
    playoff_models = load_playoff_models()

    if playoff_models is None:
        print("[FAIL] Playoff models failed to load")
        return 2

    print("Regular models: OK")
    print("Playoff models: OK")

    print("\n=== Smoke: Model Switching Logic ===")
    models = {
        "regular": regular_models,
        "playoff": playoff_models,
    }

    active_apr, label_apr = select_active_models(models, now=datetime(2026, 4, 15))
    active_jan, label_jan = select_active_models(models, now=datetime(2026, 1, 15))

    print(f"April selection : {label_apr}")
    print(f"January selection: {label_jan}")

    # Fallback check when playoff bundle is missing
    fallback_models = {
        "regular": _DummyModel(),
        "playoff": None,
    }
    _, fallback_label = select_active_models(
        fallback_models,
        now=datetime(2026, 4, 15),
    )
    print(f"Fallback label (playoff missing in April): {fallback_label}")

    if label_apr != "playoff":
        print("[FAIL] April should select playoff models")
        return 3
    if label_jan != "regular":
        print("[FAIL] January should select regular models")
        return 4
    if fallback_label != "regular":
        print("[FAIL] Missing playoff bundle should fallback to regular")
        return 5

    print("\n[OK] Smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
