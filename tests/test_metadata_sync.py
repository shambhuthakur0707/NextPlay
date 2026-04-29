import pathlib
import re
import unittest

from utils.metadata import feature_category_counts, project_metadata_snapshot


ROOT = pathlib.Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
APP_PATH = ROOT / "app.py"


class MetadataSyncTests(unittest.TestCase):
    def setUp(self):
        self.meta = project_metadata_snapshot()
        self.readme = README_PATH.read_text(encoding="utf-8")
        self.app = APP_PATH.read_text(encoding="utf-8")

    def test_readme_version_line_matches_config(self):
        expected = f"- Model version: {self.meta['model_version']}"
        self.assertIn(expected, self.readme)

    def test_readme_base_feature_count_matches_config(self):
        expected = f"- Base model feature set: {self.meta['base_feature_count']} engineered features"
        self.assertIn(expected, self.readme)

    def test_readme_stacked_feature_count_matches_config(self):
        expected = (
            f"- Total-points model: stacked meta-model "
            f"({self.meta['stacked_total_feature_count']} meta features)"
        )
        self.assertIn(expected, self.readme)

    def test_readme_feature_table_total_matches_config(self):
        match = re.search(r"\| \*\*Total\*\* \| \*\*(\d+)\*\* \|", self.readme)
        self.assertIsNotNone(match, "README feature coverage table total row is missing.")
        self.assertEqual(int(match.group(1)), self.meta["base_feature_count"])

    def test_feature_category_counts_include_playoff(self):
        counts = feature_category_counts()
        names = [name for name, _, _ in counts]
        self.assertIn("Playoff", names)
        self.assertEqual(
            sum(count for _, count, _ in counts),
            self.meta["base_feature_count"],
        )

    def test_app_uses_metadata_helper(self):
        self.assertIn("project_metadata_snapshot", self.app)
        self.assertIn("feature_category_counts", self.app)

    def test_app_does_not_have_stale_v7_labels(self):
        stale_labels = [
            "RandomForest V7",
            "84 engineered features (V7)",
            "NBA AI Score Predictor — V7",
        ]
        for label in stale_labels:
            self.assertNotIn(label, self.app)


if __name__ == "__main__":
    unittest.main()
