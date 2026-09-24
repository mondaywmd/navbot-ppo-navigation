import math
import numpy as np
from safety_shield import filter_command

class Scan:
    angle_min = -math.pi; angle_increment = 2 * math.pi / 360
    range_min = 0.12; range_max = 3.5
    def __init__(self, front):
        self.ranges = np.full(360, 3.5); self.ranges[175:186] = front

class SideScan(Scan):
    def __init__(self, side_angle_index, distance):
        super().__init__(1.0); self.ranges[side_angle_index] = distance

assert filter_command(0.25, 0.1, Scan(1.0))[0] == 0.25
slow = filter_command(0.25, 0.1, Scan(0.435))
assert 0.12 < slow[0] < 0.13 and slow[2] == "slow"
assert filter_command(0.25, 0.1, Scan(0.31))[:2] == (0.0, 0.1)
assert filter_command(0.25, 0.1, Scan(1.0), front_ttc=0.7)[:2] == (0.0, 0.1)
assert filter_command(0.20, 0.5, SideScan(225, 0.31))[:2] == (0.0, 0.0)
assert filter_command(0.20, -0.5, SideScan(135, 0.31))[:2] == (0.0, 0.0)
assert filter_command(0.20, -0.5, SideScan(225, 0.31))[0] > 0.0
assert filter_command(0.20, 0.5, SideScan(225, 0.40), turn_side_ttc=float("inf"))[0] == 0.20
assert filter_command(0.20, 0.5, SideScan(225, 0.40), turn_side_ttc=0.8)[0] < 0.20
print("SAFETY_SHIELD_TEST_PASS")
