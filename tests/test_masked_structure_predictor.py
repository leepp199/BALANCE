import unittest

import torch

from models.lsrb import MaskedStructurePredictor


class MaskedStructurePredictorTest(unittest.TestCase):
    def test_masked_loss_and_gradients(self):
        torch.manual_seed(3)
        module = MaskedStructurePredictor(16, 8, hidden_dim=8, mask_ratio=0.3)
        feature_map = torch.randn(2, 16, 7, 4, requires_grad=True)
        centers = torch.randn(8, 16)
        loss, accuracy, mask, targets = module(feature_map, centers)
        self.assertEqual(tuple(mask.shape), (2, 7, 4))
        self.assertEqual(tuple(targets.shape), (2, 7, 4))
        self.assertTrue(mask.flatten(1).any(1).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(feature_map.grad)


if __name__ == "__main__":
    unittest.main()
