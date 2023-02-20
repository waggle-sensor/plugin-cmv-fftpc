from cmv import fftFlowVector
from itertools import product
import numpy as np
import unittest


def cut(a, x, y, s):
    a[y:y+s, x:x+s] = 0


def gen_box_with_hole(box_size, hole_x, hole_y, hole_size):
    img = np.ones((box_size, box_size), dtype=float) 
    cut(img, hole_x, hole_y, hole_size)
    return img


class TestCMV(unittest.TestCase):

    def test_fftFlowVector_ones(self):
        """
        test_fftFlowVector_ones checks that fftFlowVector detects the motion
        of a cutout rectangle shifted by a known offset.
        """
        deltas = np.arange(-10, 11)

        for dx, dy in product(deltas, deltas):
            img1 = gen_box_with_hole(40, 20, 20, 10)
            img2 = gen_box_with_hole(40, 20+dx, 20+dy, 10)

            v = fftFlowVector(img1, img2)

            self.assertAlmostEqual(dx, v[1])
            self.assertAlmostEqual(dy, v[0])

    def test_fftFlowVector_ones_noise(self):
        """
        test_fftFlowVector_ones_noise checks that fftFlowVector detects the motion
        of a cutout rectangle shifted by a known offset with the addition of noise.
        """
        rand = np.random.RandomState(1234)
        deltas = np.arange(-10, 11)

        for dx, dy in product(deltas, deltas):
            img1 = gen_box_with_hole(40, 20, 20, 10) + rand.uniform(0, 0.25, (40, 40))
            img2 = gen_box_with_hole(40, 20+dx, 20+dy, 10) + rand.uniform(0, 0.25, (40, 40))

            v = fftFlowVector(img1, img2)

            self.assertAlmostEqual(dx, v[1])
            self.assertAlmostEqual(dy, v[0])


if __name__ == "__main__":
    unittest.main()
