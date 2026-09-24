"""Live two-frame smoke; does not command robot motion."""
import json
import os
import rospy
from sensor_msgs.msg import LaserScan
from temporal_lidar import sector_minima, temporal_features

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "results", "temporal_scan_smoke.json")
    if os.path.exists(path): raise RuntimeError("refusing overwrite: " + path)
    rospy.init_node("clean_start_temporal_scan_smoke", anonymous=True)
    first = rospy.wait_for_message("/scan", LaserScan, timeout=5)
    second = rospy.wait_for_message("/scan", LaserScan, timeout=5)
    a = sector_minima(first.ranges, first.range_min, first.range_max)
    b = sector_minima(second.ranges, second.range_min, second.range_max)
    dt = (second.header.stamp - first.header.stamp).to_sec()
    closing, ttc = temporal_features(a, b, dt)
    result = {"passed": len(first.ranges) == len(second.ranges) == 360,
              "training": False, "scan_points": 360, "sectors": len(a), "dt": dt,
              "minimum_current": float(b.min()),
              "maximum_stationary_closing_rate": float(closing.max()),
              "finite_ttc_sectors": int((ttc < float("inf")).sum())}
    if not result["passed"] or not 0.15 <= dt <= 0.25:
        raise AssertionError("unexpected live scan contract: %r" % result)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as output: json.dump(result, output, indent=2, sort_keys=True)
    print("TEMPORAL_SCAN_PASS " + json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__": main()
