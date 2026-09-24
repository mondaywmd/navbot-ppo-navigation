"""Pure, testable V6 sampling and permanent-test leakage exclusion."""

import json
import math

import numpy as np


TRAIN_SEED = 61129
DEFAULT_RUN_NAME = 'v6_v5init_failure_modes_10k_seed61129'
EXCLUSION_RADII = {
    'pillar_x': 0.08,
    'pillar_y': 0.06,
    'target_y': 0.06,
    'robot_yaw_deg': 4.0,
}
MODE_RANGES = {
    'positive_target_positive_yaw': {
        'pillar_x': (0.78, 1.18),
        'pillar_y': (-0.16, 0.16),
        'target_y': (0.04, 0.15),
        'robot_yaw_deg': (4.0, 15.0),
    },
    'far_pillar_negative_target_recovery': {
        'pillar_x': (1.10, 1.25),
        'pillar_y': (-0.16, 0.16),
        'target_y': (-0.15, -0.04),
        'robot_yaw_deg': (-12.0, 12.0),
    },
}


def load_permanent_tasks(path):
    with open(path) as source:
        payload = json.load(source)
    tasks = payload['tasks']
    if len(tasks) != 12:
        raise ValueError('expected exactly 12 permanent OOD tasks')
    return tasks


def near_permanent_task(candidate, task, radii=EXCLUSION_RADII):
    """Conservative joint L-infinity neighborhood around one test task."""
    return all(
        abs(float(candidate[key]) - float(task[key])) <= float(radii[key])
        for key in ('pillar_x', 'pillar_y', 'target_y', 'robot_yaw_deg')
    )


def leaking_task_name(candidate, permanent_tasks):
    for task in permanent_tasks:
        if near_permanent_task(candidate, task):
            return task['task_name']
    return None


def point_segment_distance(candidate):
    x1, y1 = 2.0, float(candidate['target_y'])
    px, py = float(candidate['pillar_x']), float(candidate['pillar_y'])
    length_sq = x1 * x1 + y1 * y1
    t = max(0.0, min(1.0, (px * x1 + py * y1) / length_sq))
    return math.hypot(px - t * x1, py - t * y1)


def legal_blocked_scene(candidate):
    # The configured arena ranges already provide wall clearance.  Explicitly
    # reject robot overlap and pillars too far from the direct blocked route.
    return (
        math.hypot(candidate['pillar_x'], candidate['pillar_y']) > 0.55
        and point_segment_distance(candidate) <= 0.42
    )


class V6ScenarioSampler:
    """Balanced online sampler; accepted scenes are never stored as a dataset."""

    def __init__(self, permanent_tasks, seed=TRAIN_SEED):
        self.permanent_tasks = permanent_tasks
        self.rng = np.random.RandomState(seed)
        self.accepted_count = 0
        self.rejected_leakage = 0
        self.rejected_illegal = 0

    def sample(self, mode):
        bounds = MODE_RANGES[mode]
        for _ in range(10000):
            candidate = {'mode': mode}
            for key, (low, high) in bounds.items():
                candidate[key] = float(self.rng.uniform(low, high))
            if not legal_blocked_scene(candidate):
                self.rejected_illegal += 1
                continue
            if leaking_task_name(candidate, self.permanent_tasks) is not None:
                self.rejected_leakage += 1
                continue
            self.accepted_count += 1
            return candidate
        raise RuntimeError('could not sample a legal non-leaking V6 scene')

    def sample_balanced(self, episode_index):
        modes = tuple(MODE_RANGES)
        return self.sample(modes[int(episode_index) % len(modes)])

