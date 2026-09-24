import numpy as np

from temporal_lidar import sanitize_ranges, sector_minima, temporal_features


raw = np.full(360, 3.5, dtype=np.float32)
raw[24] = 0.41; raw[25] = 0.37; raw[100] = np.inf; raw[200] = np.nan
clean = sanitize_ranges(raw, 0.12, 3.5)
assert np.isfinite(clean).all() and clean[100] == 3.5 and clean[200] == 3.5
sectors = sector_minima(raw, 0.12, 3.5)
assert sectors.shape == (36,) and abs(float(sectors[2]) - 0.37) < 1e-6
previous = np.full(36, 1.0, dtype=np.float32)
current = previous.copy(); current[5] = 0.8; current[6] = 1.2
closing, ttc = temporal_features(previous, current, 0.2)
assert abs(float(closing[5]) - 1.0) < 1e-6
assert closing[6] == 0.0 and np.isinf(ttc[6])
assert abs(float(ttc[5]) - 0.48) < 1e-5
print("TEMPORAL_LIDAR_TEST_PASS sectors=36 closing=%.3f ttc=%.3f" %
      (closing[5], ttc[5]))
