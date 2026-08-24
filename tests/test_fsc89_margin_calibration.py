import copy
import unittest

import numpy as np
import torch

from scripts.calibrate_fsc89_margin import (
    BASE_CLASSES,
    PSEUDO_NOVEL_CLASSES,
    SEALED_TEST_CLASSES,
    make_episode_scores,
    select_threshold,
    validate_geometry,
)


def _synthetic_geometry(seed=7, feature_dim=32):
    generator = torch.Generator().manual_seed(seed)
    base_means = torch.randn(len(BASE_CLASSES), feature_dim, generator=generator)
    pseudo_means = torch.randn(
        len(PSEUDO_NOVEL_CLASSES), feature_dim, generator=generator
    )

    def around(center, count):
        noise = 0.01 * torch.randn(count, feature_dim, generator=generator)
        return center.unsqueeze(0) + noise

    return {
        "schema_version": 1,
        "artifact_type": "fsc89_pseudo_unseen_geometry",
        "dataset": "FSC-89",
        "offline": True,
        "encoder": "current model.encode",
        "feature_api": "model.encode",
        "checkpoint_sha256": "0" * 64,
        "training_classes_zero_based": list(BASE_CLASSES),
        "base_classes_zero_based": list(BASE_CLASSES),
        "pseudo_novel_classes_zero_based": list(PSEUDO_NOVEL_CLASSES),
        "sealed_test_classes_zero_based": list(SEALED_TEST_CLASSES),
        "base_prototype_source": "train_mean_from_same_encoder",
        "test_csv_opened": False,
        "encoded_class_max": 68,
        "base_train_means": base_means,
        "base_query_features": [around(center, 2) for center in base_means],
        "pseudo_support_features": [around(center, 30) for center in pseudo_means],
        "pseudo_query_features": [around(center, 3) for center in pseudo_means],
    }


class Fsc89MarginCalibrationTest(unittest.TestCase):
    def test_episode_uses_exactly_five_supports_per_pseudo_novel_class(self):
        geometry = _synthetic_geometry()
        validate_geometry(geometry)
        base_score, novel_score, audit = make_episode_scores(
            geometry, ways=5, rng=np.random.default_rng(3420)
        )

        self.assertEqual(len(audit), 5)
        self.assertTrue(all(len(indices) == 5 for indices in audit.values()))
        self.assertTrue(all(len(set(indices)) == 5 for indices in audit.values()))
        self.assertEqual(len(base_score), 2 * len(BASE_CLASSES))
        self.assertEqual(len(novel_score), 3 * 5)

    def test_scalar_threshold_separates_base_and_novel_margins(self):
        records = [
            (5, torch.tensor([-2.0, -1.0]), torch.tensor([1.0, 2.0])),
            (10, torch.tensor([-1.5, -0.5]), torch.tensor([0.5, 1.5])),
        ]
        threshold, auroc = select_threshold(records)

        self.assertGreaterEqual(threshold, -0.5)
        self.assertLessEqual(threshold, 0.5)
        self.assertEqual(auroc, 1.0)

    def test_calibration_and_audit_support_draws_are_disjoint(self):
        geometry = _synthetic_geometry()
        _, _, calibration = make_episode_scores(
            geometry,
            ways=10,
            rng=np.random.default_rng(10),
            partition="calibration",
            calibration_fraction=0.8,
        )
        _, _, audit = make_episode_scores(
            geometry,
            ways=10,
            rng=np.random.default_rng(11),
            partition="audit",
            calibration_fraction=0.8,
        )

        for class_id in PSEUDO_NOVEL_CLASSES:
            self.assertTrue(all(index < 24 for index in calibration[class_id]))
            self.assertTrue(all(index >= 24 for index in audit[class_id]))
            self.assertTrue(set(calibration[class_id]).isdisjoint(audit[class_id]))

    def test_geometry_validation_fails_closed_on_sealed_class_access(self):
        geometry = copy.deepcopy(_synthetic_geometry())
        geometry["encoded_class_max"] = 69

        with self.assertRaisesRegex(ValueError, "encoded_class_max"):
            validate_geometry(geometry)


if __name__ == "__main__":
    unittest.main()
