# Direct Goal V1 + Heading Compensation

Verified no-obstacle controller.

- Cruise speed: 0.25 m/s
- Brakes near target
- Confirms physical angular velocity is stable before driving
- Small heading-hold compensation only; it does not re-plan routes
- Uses LiDAR collision stop and 0.20 m arrival radius

Run once:
docker exec -it navbot-ppo bash /root/catkin_ws/src/project_ppo/baselines/direct_goal_v1_heading_comp_v1/restore_and_run_once.sh
