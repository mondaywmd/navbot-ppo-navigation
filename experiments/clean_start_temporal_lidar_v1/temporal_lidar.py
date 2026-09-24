"""Dense-scan sector minima and approach-rate features."""
import numpy as np


SECTORS = 36


def sanitize_ranges(ranges, range_min, range_max):
    values = np.asarray(ranges, dtype=np.float32)
    valid = np.isfinite(values) & (values >= range_min) & (values <= range_max)
    return np.where(valid, values, float(range_max)).astype(np.float32)


def sector_minima(ranges, range_min, range_max, sectors=SECTORS):
    values = sanitize_ranges(ranges, range_min, range_max)
    if values.ndim != 1 or len(values) < sectors or len(values) % sectors:
        raise ValueError("scan length must be a one-dimensional multiple of sectors")
    return values.reshape(sectors, len(values) // sectors).min(axis=1)


def temporal_features(previous, current, dt, hard_distance=0.32,
                      closing_deadband=0.06):
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape or previous.ndim != 1:
        raise ValueError("previous/current sector arrays must have equal 1-D shape")
    if dt <= 0:
        raise ValueError("dt must be positive")
    closing_rate = np.maximum((previous - current) / float(dt), 0.0)
    ttc = np.full(current.shape, np.inf, dtype=np.float32)
    closing_rate[closing_rate < closing_deadband] = 0.0
    closing = closing_rate > 0.0
    ttc[closing] = np.maximum(current[closing] - hard_distance, 0.0) / closing_rate[closing]
    return closing_rate.astype(np.float32), ttc

def signed_range_rate(previous, current, dt, limit=1.0):
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape or previous.ndim != 1 or dt <= 0:
        raise ValueError("invalid temporal range inputs")
    return np.clip((previous - current) / float(dt), -limit, limit).astype(np.float32)
