import unittest

import torch

from models.lsrb import LatentStructureReferenceBank, descriptor_matrix


class LatentStructureReferenceBankTest(unittest.TestCase):
    def test_shapes_normalization_and_finiteness(self):
        torch.manual_seed(7)
        bank = LatentStructureReferenceBank(torch.randn(8, 512), temperature=0.2)
        feature_map = torch.randn(3, 512, 7, 4)
        outputs = bank.compute(feature_map)
        self.assertEqual(tuple(outputs.assignments.shape), (3, 7, 4))
        self.assertEqual(tuple(outputs.structural_response.shape), (3, 8))
        self.assertEqual(tuple(outputs.structure_residual.shape), (3,))
        self.assertTrue(torch.isfinite(outputs.structural_response).all())
        self.assertTrue(torch.isfinite(outputs.structure_residual).all())
        self.assertTrue(torch.allclose(outputs.structural_response.sum(-1), torch.ones(3)))

    def test_descriptor_matrix(self):
        descriptors = descriptor_matrix(torch.randn(2, 512, 7, 4))
        self.assertEqual(tuple(descriptors.shape), (56, 512))
        self.assertTrue(torch.allclose(descriptors.norm(dim=-1), torch.ones(56), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
