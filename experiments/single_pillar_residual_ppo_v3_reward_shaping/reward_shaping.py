"""Reward shaping used only by the v3 control-gated experiment."""

import numpy as np


CLEARANCE_GAIN = 40.0
CLEARANCE_DELTA_LIMIT = 0.10
TAKEOVER_HEADING_SCALE = 0.10


def shaped_step_reward(
    distance_rate,
    heading_improvement,
    heading_alignment,
    clearance_delta,
    gate_active,
):
    if not gate_active:
        return (
            100.0 * distance_rate
            - 1.0
            + 12.0 * heading_improvement
            + 0.5 * heading_alignment
        )

    bounded_clearance_delta = float(np.clip(
        clearance_delta, -CLEARANCE_DELTA_LIMIT, CLEARANCE_DELTA_LIMIT
    ))
    return (
        100.0 * distance_rate
        - 1.0
        + TAKEOVER_HEADING_SCALE * (
            12.0 * heading_improvement + 0.5 * heading_alignment
        )
        + CLEARANCE_GAIN * bounded_clearance_delta
    )
