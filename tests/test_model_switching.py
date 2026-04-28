import unittest
from datetime import datetime
from unittest import mock

from config import PLAYOFF_SEASON_START_MONTH, PLAYOFF_SEASON_END_MONTH
from pipelines import nightly


class ModelSwitchingTests(unittest.TestCase):
    def _outside_month(self):
        if PLAYOFF_SEASON_START_MONTH <= PLAYOFF_SEASON_END_MONTH:
            candidate = (PLAYOFF_SEASON_END_MONTH % 12) + 1
            return candidate

        # Wrapped window (e.g., Nov-Feb): pick a month strictly between end and start.
        candidate = (PLAYOFF_SEASON_END_MONTH % 12) + 1
        if candidate == PLAYOFF_SEASON_START_MONTH:
            candidate = (candidate % 12) + 1
        return candidate

    def test_is_playoff_season_true_inside_window(self):
        dt = datetime(2026, PLAYOFF_SEASON_START_MONTH, 15)
        self.assertTrue(nightly.is_playoff_season(now=dt))

    def test_is_playoff_season_false_outside_window(self):
        dt = datetime(2026, self._outside_month(), 15)
        self.assertFalse(nightly.is_playoff_season(now=dt))

    def test_select_active_models_uses_playoff_when_available(self):
        models = {"regular": object(), "playoff": object()}
        dt = datetime(2026, PLAYOFF_SEASON_START_MONTH, 15)

        with mock.patch.object(nightly, "USE_PLAYOFF_MODELS", True):
            active, label = nightly.select_active_models(models, now=dt)

        self.assertIs(active, models["playoff"])
        self.assertEqual(label, "playoff")

    def test_select_active_models_falls_back_to_regular_if_playoff_missing(self):
        models = {"regular": object(), "playoff": None}
        dt = datetime(2026, PLAYOFF_SEASON_START_MONTH, 15)

        with mock.patch.object(nightly, "USE_PLAYOFF_MODELS", True):
            active, label = nightly.select_active_models(models, now=dt)

        self.assertIs(active, models["regular"])
        self.assertEqual(label, "regular")

    def test_select_active_models_uses_regular_when_playoff_feature_disabled(self):
        models = {"regular": object(), "playoff": object()}
        dt = datetime(2026, PLAYOFF_SEASON_START_MONTH, 15)

        with mock.patch.object(nightly, "USE_PLAYOFF_MODELS", False):
            active, label = nightly.select_active_models(models, now=dt)

        self.assertIs(active, models["regular"])
        self.assertEqual(label, "regular")


if __name__ == "__main__":
    unittest.main()
