from cmv import fftFlowVector
from itertools import product
import numpy as np
import unittest


def cut(a, x, y, s):
    a[y:y+s, x:x+s] = 0


class TestCMV(unittest.TestCase):

    def test_fftFlowVector_ones(self):
        deltas = np.arange(-10, 11)

        for dx, dy in product(deltas, deltas):
            img1 = np.ones((40, 40), dtype=float)
            img2 = np.ones((40, 40), dtype=float)

            cut(img1, 20, 20, 10)
            cut(img2, 20+dx, 20+dy, 10)

            v = fftFlowVector(img1, img2)

            self.assertAlmostEqual(dx, v[1])
            self.assertAlmostEqual(dy, v[0])

    def test_fftFlowVector_ones_noise(self):
        rand = np.random.RandomState(1234)
        deltas = np.arange(-10, 11)

        for dx, dy in product(deltas, deltas):
            img1 = np.ones((40, 40), dtype=float) + rand.uniform(0, 0.25, (40, 40))
            img2 = np.ones((40, 40), dtype=float) + rand.uniform(0, 0.25, (40, 40))

            cut(img1, 20, 20, 10)
            cut(img2, 20+dx, 20+dy, 10)

            v = fftFlowVector(img1, img2)

            self.assertAlmostEqual(dx, v[1])
            self.assertAlmostEqual(dy, v[0])


if __name__ == "__main__":
    unittest.main()
