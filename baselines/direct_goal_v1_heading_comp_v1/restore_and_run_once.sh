#!/usr/bin/env bash
set -euo pipefail
BASE=/root/catkin_ws/src/project_ppo/baselines/direct_goal_v1_heading_comp_v1
SRC=/root/catkin_ws/src/project_ppo/src

cp "$BASE/raw_cmdvel_direct_goal.py" "$SRC/raw_cmdvel_direct_goal.py"
cp "$BASE/environment_new.py" "$SRC/environment_new.py"

export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11345}

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
python3 -m py_compile "$SRC/raw_cmdvel_direct_goal.py"
python3 "$SRC/raw_cmdvel_direct_goal.py"
