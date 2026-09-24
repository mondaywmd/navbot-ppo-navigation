"""Collect only complete, safe, strictly paired expert trajectories."""
import csv
import json
import math
import os
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from control_gate import minimum_valid_range
from curriculum_spec import generate, mirror
from residual_networks import ResidualActor
from validate_multilevel_expert import PathExpert
from wide_env import WideStaticEnv

SEED = 8042705
ACCEPTED_PAIRS_PER_LEVEL = 8
MAX_CANDIDATES_PER_LEVEL = 30
MIN_CLEARANCE = 0.25
MAX_STEPS = 300
ROOT = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(ROOT, "..", "single_pillar_residual_ppo_v5_balanced_curriculum",
    "runs", "v5_v4init_pillar_y_curriculum_10k_seed29", "checkpoints", "actor_iter0016_step00010421.pth"))


def past_all_obstacles(x, y, scene):
    tx, ty = scene["target_x"], scene["target_y"]; length = math.hypot(tx, ty)
    projection = (x*tx + y*ty) / length
    furthest = max((p["x"]*tx + p["y"]*ty)/length + p["radius"] for p in scene["obstacles"])
    return projection > furthest + 0.20


def run(env, scene, teacher):
    observation = env.reset_wide(scene); expert = PathExpert(scene)
    past = np.zeros(2, dtype=np.float32); trace = []; minimum = float("inf"); outcome = "timeout"
    for step in range(1, MAX_STEPS + 1):
        actor_input = observation.copy(); action = expert.action(env)
        with torch.inference_mode(): teacher_action = teacher(actor_input).squeeze(0).numpy().astype(np.float32)
        x, y = float(env.position.x), float(env.position.y)
        goal_distance = math.hypot(scene["target_x"]-x, scene["target_y"]-y)
        clearance_before = minimum_valid_range(env.latest_scan)
        keep = past_all_obstacles(x, y, scene) and clearance_before >= 0.50 and goal_distance <= 1.20
        observation, _, done, arrive, command, gate, before = env.step_residual(action, past)
        clearance = min(float(before), minimum_valid_range(env.latest_scan)); minimum = min(minimum, clearance)
        trace.append({"observation": actor_input, "expert_action": action.copy(),
                      "teacher_action": teacher_action, "keep_v5": keep,
                      "gate": bool(gate), "clearance": clearance,
                      "robot_x": x, "robot_y": y, "goal_distance": goal_distance,
                      "command": np.asarray(command).copy()})
        past = action
        if arrive: outcome = "success"; break
        if done: outcome = "collision"; break
    env.pub_cmd_vel.publish(Twist())
    return trace, {"outcome": outcome, "steps": step, "min_lidar": minimum}


def main():
    outdir = os.path.join(ROOT, "datasets", "paired_success_bc_seed8042705")
    if os.path.exists(outdir): raise RuntimeError("refusing overwrite: " + outdir)
    os.makedirs(outdir); rospy.init_node("v8_collect_paired_success_bc", anonymous=True)
    teacher = ResidualActor(16, 2); teacher.load_state_dict(torch.load(V5, map_location="cpu")); teacher.eval()
    env = WideStaticEnv(); accepted_traces = []; audit = []; accepted_by_level = {}
    try:
        for level in (1, 2, 3):
            accepted = 0
            candidates = generate(level, MAX_CANDIDATES_PER_LEVEL, seed=SEED)
            for candidate_index, base in enumerate(candidates, 1):
                if accepted >= ACCEPTED_PAIRS_PER_LEVEL: break
                pair_rows = []
                for scene in (base, mirror(base)):
                    trace, result = run(env, scene, teacher); pair_rows.append((scene, trace, result))
                    print("BC_SEARCH level=%d candidate=%d side=%s outcome=%s steps=%d min=%.3f" %
                          (level, candidate_index, "mirror" if scene["name"].endswith("_mirror") else "base",
                           result["outcome"], result["steps"], result["min_lidar"]), flush=True)
                qualified = all(result["outcome"] == "success" and result["min_lidar"] >= MIN_CLEARANCE
                                for _, _, result in pair_rows)
                audit.append({"level": level, "candidate": candidate_index, "accepted": int(qualified),
                    "base_outcome": pair_rows[0][2]["outcome"], "base_steps": pair_rows[0][2]["steps"],
                    "base_min_lidar": pair_rows[0][2]["min_lidar"],
                    "mirror_outcome": pair_rows[1][2]["outcome"], "mirror_steps": pair_rows[1][2]["steps"],
                    "mirror_min_lidar": pair_rows[1][2]["min_lidar"]})
                if qualified:
                    accepted += 1
                    for side_index, (scene, trace, _) in enumerate(pair_rows):
                        accepted_traces.append((level, accepted, side_index, scene, trace))
            if accepted != ACCEPTED_PAIRS_PER_LEVEL:
                raise RuntimeError("level %d accepted only %d pairs" % (level, accepted))
            accepted_by_level[str(level)] = accepted
        observations=[]; actions=[]; teacher_actions=[]; keep_masks=[]; levels=[]; pair_ids=[]; sides=[]; episode_ids=[]
        metadata=[]
        for episode_id,(level,pair_id,side,scene,trace) in enumerate(accepted_traces):
            for row in trace:
                observations.append(row["observation"]); actions.append(row["expert_action"])
                teacher_actions.append(row["teacher_action"]); keep_masks.append(row["keep_v5"])
                levels.append(level); pair_ids.append(pair_id); sides.append(side); episode_ids.append(episode_id)
            metadata.append({"episode_id":episode_id,"level":level,"pair_id":pair_id,
                             "side":"base" if side==0 else "mirror","scene":scene,"samples":len(trace)})
        np.savez_compressed(os.path.join(outdir,"paired_success_bc.npz"),
            observations=np.asarray(observations,dtype=np.float32), actions=np.asarray(actions,dtype=np.float32),
            v5_teacher_actions=np.asarray(teacher_actions,dtype=np.float32), keep_v5_mask=np.asarray(keep_masks,dtype=np.bool_),
            levels=np.asarray(levels,dtype=np.int64), pair_ids=np.asarray(pair_ids,dtype=np.int64),
            mirror_sides=np.asarray(sides,dtype=np.int64), episode_ids=np.asarray(episode_ids,dtype=np.int64),
            dataset_role=np.asarray("v8_paired_success_bc_training_only"), seed=np.asarray(SEED,dtype=np.int64))
        with open(os.path.join(outdir,"episodes.json"),"w") as f: json.dump(metadata,f,indent=2,sort_keys=True)
        with open(os.path.join(outdir,"candidate_audit.csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
        summary={"seed":SEED,"accepted_pairs_by_level":accepted_by_level,"accepted_pairs":sum(accepted_by_level.values()),
                 "episodes":len(accepted_traces),"samples":len(observations),"keep_v5_samples":int(sum(keep_masks)),
                 "min_clearance_required":MIN_CLEARANCE,"training_started":False}
        with open(os.path.join(outdir,"summary.json"),"w")as f:json.dump(summary,f,indent=2,sort_keys=True)
        print("BC_COLLECTION_COMPLETE "+json.dumps(summary,sort_keys=True),flush=True)
    finally: env.restore_center();env.pub_cmd_vel.publish(Twist());print("CENTER_RESTORED_ZERO",flush=True)


if __name__=="__main__":main()
