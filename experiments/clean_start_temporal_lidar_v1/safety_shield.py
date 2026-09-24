"""Deterministic full-scan forward collision shield (no learned parameters)."""
import math
import numpy as np

HARD_DISTANCE = 0.32
SLOW_DISTANCE = 0.55
EMERGENCY_DISTANCE = 0.24
TTC_LIMIT = 1.0
FRONT_HALF_ANGLE = math.radians(35.0)
TURN_INNER_MIN_ANGLE = math.radians(20.0)
TURN_INNER_MAX_ANGLE = math.radians(105.0)
TURN_HARD_DISTANCE = 0.32
TURN_SLOW_DISTANCE = 0.50
TURN_TTC_LIMIT = 1.5

def valid_scan(scan):
    values = np.asarray(scan.ranges, dtype=np.float32)
    valid = np.isfinite(values) & (values >= scan.range_min) & (values <= scan.range_max)
    return np.where(valid, values, float(scan.range_max))

def front_minimum(scan):
    values = valid_scan(scan)
    angles = scan.angle_min + np.arange(len(values)) * scan.angle_increment
    wrapped = (angles + math.pi) % (2 * math.pi) - math.pi
    return float(values[np.abs(wrapped) <= FRONT_HALF_ANGLE].min())

def turning_side_minimum(scan, angular):
    if abs(float(angular)) < 1e-6:
        return float(scan.range_max)
    values = valid_scan(scan)
    angles = scan.angle_min + np.arange(len(values)) * scan.angle_increment
    wrapped = (angles + math.pi) % (2 * math.pi) - math.pi
    if angular > 0:
        mask = (wrapped >= TURN_INNER_MIN_ANGLE) & (wrapped <= TURN_INNER_MAX_ANGLE)
    else:
        mask = (wrapped <= -TURN_INNER_MIN_ANGLE) & (wrapped >= -TURN_INNER_MAX_ANGLE)
    return float(values[mask].min())

def filter_command(linear, angular, scan, front_ttc=float("inf"),
                   turn_side_ttc=float("inf")):
    values = valid_scan(scan)
    nearest = float(values.min()); front = front_minimum(scan)
    turn_side = turning_side_minimum(scan, angular)
    reason = "clear"; scale = 1.0
    if nearest <= EMERGENCY_DISTANCE:
        return 0.0, 0.0, "emergency_stop"
    if front <= HARD_DISTANCE or front_ttc <= TTC_LIMIT:
        return 0.0, float(angular), "predictive_stop"
    if turn_side <= TURN_HARD_DISTANCE:
        return 0.0, 0.0, "turn_side_stop"
    if turn_side < TURN_SLOW_DISTANCE and turn_side_ttc <= TURN_TTC_LIMIT:
        turn_scale = (turn_side - TURN_HARD_DISTANCE) / (TURN_SLOW_DISTANCE - TURN_HARD_DISTANCE)
        linear = max(0.0, float(linear)) * float(np.clip(turn_scale, 0.0, 1.0))
        angular = float(angular) * float(np.clip(turn_scale, 0.0, 1.0))
        reason = "turn_side_slow"
    if front < SLOW_DISTANCE:
        scale = (front - HARD_DISTANCE) / (SLOW_DISTANCE - HARD_DISTANCE)
        reason = "slow"
    return max(0.0, float(linear)) * float(np.clip(scale, 0.0, 1.0)), float(angular), reason
