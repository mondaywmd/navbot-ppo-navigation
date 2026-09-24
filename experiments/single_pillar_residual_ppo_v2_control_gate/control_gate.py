"""LiDAR-gated control authority for the v2 single-pillar experiment."""

import math

import numpy as np

from residual_action import (
    MAX_ANGULAR_RESIDUAL,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_RESIDUAL,
    MAX_LINEAR_SPEED,
    goal_seeking_command,
)


ENTER_DISTANCE = 0.80
EXIT_DISTANCE = 1.10
EXIT_CLEAR_SCANS = 3
GATED_BASE_LINEAR_MAX = 0.09


def minimum_valid_range(scan):
    values = [
        float(value) for value in scan.ranges
        if math.isfinite(value) and scan.range_min <= value <= scan.range_max
    ]
    return min(values) if values else float(scan.range_max)


class LidarControlGate:
    def __init__(self):
        self.active = False
        self.clear_scans = 0

    def reset(self):
        self.active = False
        self.clear_scans = 0

    def update(self, front_clearance):
        clearance = float(front_clearance)
        if not self.active:
            if clearance < ENTER_DISTANCE:
                self.active = True
                self.clear_scans = 0
        elif clearance > EXIT_DISTANCE:
            self.clear_scans += 1
            if self.clear_scans >= EXIT_CLEAR_SCANS:
                self.active = False
                self.clear_scans = 0
        else:
            self.clear_scans = 0
        return self.active


def compose_gated_command(distance, heading_error, residual_action, gate_active):
    """Use residual authority only while LiDAR has granted obstacle takeover."""
    nominal = goal_seeking_command(distance, heading_error)
    if not gate_active:
        return nominal

    residual = np.asarray(residual_action, dtype=np.float32)
    if residual.shape != (2,) or not np.all(np.isfinite(residual)):
        raise ValueError("residual_action must be a finite shape-(2,) vector")
    residual = np.clip(residual, -1.0, 1.0)
    command = np.asarray(
        [min(float(nominal[0]), GATED_BASE_LINEAR_MAX), 0.0],
        dtype=np.float32,
    )
    command += residual * np.asarray(
        [MAX_LINEAR_RESIDUAL, MAX_ANGULAR_RESIDUAL], dtype=np.float32
    )
    command[0] = np.clip(command[0], 0.0, MAX_LINEAR_SPEED)
    command[1] = np.clip(command[1], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
    return command
