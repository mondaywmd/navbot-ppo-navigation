"""Export a standalone dense-LiDAR URDF without modifying legacy files."""
import argparse
import os

import rospy

from install_dense_lidar import dense_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise RuntimeError("refusing overwrite: " + args.output)
    rospy.init_node("clean_start_export_dense_urdf", anonymous=True)
    original = rospy.get_param("/robot_description")
    dense = dense_description(original)
    with open(args.output, "w") as output:
        output.write(dense)
    print("DENSE_URDF_EXPORTED bytes=%d path=%s" % (len(dense), args.output), flush=True)


if __name__ == "__main__":
    main()
