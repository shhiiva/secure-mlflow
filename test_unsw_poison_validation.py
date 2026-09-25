import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unsw_poison_validation import (
    activation_clustering,
    canonical_sha256,
    fusion,
    select_poison_indices,
    trigger_copy,
    validate_resume_configuration,
)


class UnswIntegrityTests(unittest.TestCase):
    def test_activation_clustering_handles_degenerate_cluster(self):
        class DegenerateModel:
            coefs_ = [np.zeros((3, 2)), np.zeros((2, 1))]
            intercepts_ = [np.zeros(2), np.zeros(1)]

        result = activation_clustering(
            DegenerateModel(), np.zeros((100, 3), dtype=float), seed=42
        )
        self.assertEqual(result["silhouette_status"], "DEGENERATE_SINGLE_CLUSTER")
        self.assertEqual(result["score"], 0.0)

    def test_poison_count_matches_rate(self):
        labels = np.asarray([0, 1] * 500)
        selected = select_poison_indices(labels, 0.05, 42)
        self.assertEqual(len(selected), 50)
        self.assertEqual(len(set(selected.tolist())), 50)

    def test_s2_uses_source_class_only(self):
        labels = np.asarray([0] * 80 + [1] * 20)
        selected = select_poison_indices(labels, 0.10, 42, source_label=0)
        self.assertTrue(np.all(labels[selected] == 0))

    def test_deterministic_distinct_seeds(self):
        labels = np.asarray([0, 1] * 500)
        first = select_poison_indices(labels, 0.05, 42)
        repeated = select_poison_indices(labels, 0.05, 42)
        different = select_poison_indices(labels, 0.05, 43)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))

    def test_trigger_copy_does_not_modify_clean_data(self):
        clean = pd.DataFrame({"sbytes": [1.0, 2.0], "dbytes": [3.0, 4.0]})
        modified = trigger_copy(clean, "sbytes", 99.0)
        self.assertEqual(clean["sbytes"].tolist(), [1.0, 2.0])
        self.assertEqual(modified["sbytes"].tolist(), [99.0, 99.0])

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "configuration.json"
            old = {"seed": 1}
            old["configuration_sha256"] = canonical_sha256(old)
            path.write_text(json.dumps(old), encoding="utf-8")
            new = {"seed": 2}
            new["configuration_sha256"] = canonical_sha256(new)
            with self.assertRaises(RuntimeError):
                validate_resume_configuration(path, new, True)

    def test_fusion_threshold_is_fixed(self):
        result = fusion(
            {"neural_cleanse": 0.2, "strip": 0.2, "activation_clustering": 0.2},
            {"neural_cleanse": 0.1, "strip": 0.1, "activation_clustering": 0.1},
        )
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["decision"], "MALICIOUS")


if __name__ == "__main__":
    unittest.main()
