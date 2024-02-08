import unittest
import numpy as np
import torch
from scipy import linalg
from repro_gan import metrics

class TestCalculateFID(unittest.TestCase):
    def test_calculate_FID(self):
        # Create dummy data
        real = torch.tensor(np.random.randn(10, 64, 3152))
        fake = torch.tensor(np.random.randn(10, 64, 3152))
        netD = DummyNetD()

        # Call the function
        fid = calculate_FID(real, fake, netD)

        # Assert the result
        self.assertIsInstance(fid, float)

class DummyNetD:
    def __call__(self, x, interm=False):
        return torch.randn(x.shape[0], 64)

if __name__ == '__main__':
    unittest.main()import unittest

class TestCalculateFID(unittest.TestCase):
    def test_calculate_FID(self):
        real = torch.tensor(np.random.randn(10, 64, 3152))
        fake = torch.tensor(np.random.randn(10, 64, 3152))
        netD = DummyNetD()

        fid = calculate_FID(real, fake, netD)

        self.assertIsInstance(fid, float)

class DummyNetD:
    def __call__(self, x, interm=False):
        return torch.randn(x.shape[0], 64)

if __name__ == '__main__':
    unittest.main()