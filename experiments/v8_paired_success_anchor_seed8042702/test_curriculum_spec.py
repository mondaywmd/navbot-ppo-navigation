import unittest
import math
from curriculum_spec import (GOAL_CLEARANCE, LEVELS, START_CLEARANCE,
                             direct_path_blocked, generate, legal, mirror)


class CurriculumTests(unittest.TestCase):
    def test_all_levels_are_legal_and_mirrored(self):
        for level, cfg in LEVELS.items():
            for scene in generate(level, 100):
                other = mirror(scene)
                self.assertTrue(legal(scene)); self.assertTrue(legal(other))
                self.assertTrue(direct_path_blocked(scene))
                self.assertTrue(direct_path_blocked(other))
                for obstacle in scene["obstacles"]:
                    self.assertGreaterEqual(math.hypot(obstacle["x"], obstacle["y"]),
                                            obstacle["radius"] + START_CLEARANCE)
                    self.assertGreaterEqual(math.hypot(obstacle["x"]-scene["target_x"],
                                                       obstacle["y"]-scene["target_y"]),
                                            obstacle["radius"] + GOAL_CLEARANCE)
                self.assertEqual(len(scene["obstacles"]), cfg["obstacles"])
                self.assertEqual(scene["target_x"], other["target_x"])
                self.assertEqual(scene["target_y"], -other["target_y"])
                for a, b in zip(scene["obstacles"], other["obstacles"]):
                    self.assertEqual(a["x"], b["x"]); self.assertEqual(a["y"], -b["y"])
                    self.assertEqual(a["radius"], b["radius"])

    def test_generation_is_reproducible(self):
        self.assertEqual(generate(3, 20), generate(3, 20))


if __name__ == "__main__": unittest.main()
