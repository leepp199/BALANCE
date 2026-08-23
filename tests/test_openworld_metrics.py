import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from threshold_free import _unknown_auroc


def test_unknown_auroc_uses_negative_known_margin():
    labels = np.array([0, 1, 80, 81])
    known_margin = np.array([0.9, 0.7, -0.2, -0.5])
    assert _unknown_auroc(labels, known_margin, 80) == 1.0


def test_unknown_auroc_is_nan_for_single_group():
    labels = np.array([0, 1, 2])
    known_margin = np.array([0.9, 0.7, 0.5])
    assert np.isnan(_unknown_auroc(labels, known_margin, 80))
