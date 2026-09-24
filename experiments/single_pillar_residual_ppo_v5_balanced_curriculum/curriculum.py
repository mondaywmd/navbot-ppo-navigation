"""V5 two-level curriculum specification; no Gazebo or training side effects."""

from dataclasses import dataclass

import numpy as np


HELD_OUT_LIMITS = {
    'pillar_y': 0.10,
    'robot_yaw_deg': 5.0,
    'target_y': 0.10,
}


@dataclass(frozen=True)
class CurriculumLevel:
    name: str
    pillar_y_limit: float
    robot_yaw_deg_limit: float
    target_y_limit: float

    def sample(self, rng):
        """Sample only reset parameters; these values are never Actor inputs."""
        return {
            'pillar_y': float(rng.uniform(-self.pillar_y_limit, self.pillar_y_limit)),
            'robot_yaw_deg': float(rng.uniform(
                -self.robot_yaw_deg_limit, self.robot_yaw_deg_limit
            )),
            'target_y': float(rng.uniform(-self.target_y_limit, self.target_y_limit)),
        }


LEVELS = (
    CurriculumLevel('level1_pillar_y_03m', 0.030, 0.0, 0.0),
    CurriculumLevel('level2_pillar_y_06m', 0.060, 0.0, 0.0),
)


def seeded_scenarios(level, count, seed):
    rng = np.random.RandomState(seed)
    return [level.sample(rng) for _ in range(count)]
