import unittest
from scenes import generate, validate


class TestPreparation(unittest.TestCase):
    def test_six_exact_pairs(self):
        scenes = generate(); self.assertTrue(validate(scenes)); self.assertEqual(len(scenes), 12)

    def test_opposite_detour_labels(self):
        scenes = generate()
        for a, b in zip(scenes[0::2], scenes[1::2]):
            self.assertEqual(a["detour_sign"], -b["detour_sign"])


if __name__ == "__main__": unittest.main()
