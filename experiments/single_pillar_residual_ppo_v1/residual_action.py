"""Bounded residual action interface for the fixed single-pillar task.

The PPO policy emits two values in [-1, 1].  They are small corrections to a
direct-to-goal controller, not complete velocity commands.  Keeping this
module ROS-free makes the command semantics testable before training is wired
into Gazebo.
"""

import math

import numpy as np


MAX_LINEAR_SPEED = 0.25
MAX_ANGULAR_SPEED = 0.60
MAX_LINEAR_RESIDUAL = 0.05
MAX_ANGULAR_RESIDUAL = 0.30
HEADING_GAIN = 0.8


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_radians(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def goal_seeking_command(distance, heading_error):
    """Return the nominal physical command ``(linear, angular)``."""
    distance = max(0.0, float(distance))
    heading_error = wrap_radians(float(heading_error))

    if distance <= 0.20:
        linear = 0.0
    elif distance <= 0.50:
        linear = 0.05 + (distance - 0.20) / 0.30 * 0.05
    elif distance <= 1.00:
        linear = 0.10 + (distance - 0.50) * 0.30
    else:
        linear = MAX_LINEAR_SPEED

    angular = clamp(HEADING_GAIN * heading_error,
                    -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

    # Preserve the direct-goal controller's turn-first behavior.
    if abs(heading_error) > math.radians(70.0):
        linear = 0.0
    elif abs(heading_error) > math.radians(6.0):
        linear = min(linear, 0.15)

    return np.asarray([linear, angular], dtype=np.float32)


def compose_residual_command(distance, heading_error, residual_action):
    """Add a normalized, tightly bounded PPO correction to the nominal command."""
    residual = np.asarray(residual_action, dtype=np.float32)
    if residual.shape != (2,):
        raise ValueError("residual_action must have shape (2,)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_action must contain only finite values")

    residual = np.clip(residual, -1.0, 1.0)
    nominal = goal_seeking_command(distance, heading_error)
    command = nominal + residual * np.asarray(
        [MAX_LINEAR_RESIDUAL, MAX_ANGULAR_RESIDUAL], dtype=np.float32
    )
    command[0] = np.clip(command[0], 0.0, MAX_LINEAR_SPEED)
    command[1] = np.clip(command[1], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
    return command
