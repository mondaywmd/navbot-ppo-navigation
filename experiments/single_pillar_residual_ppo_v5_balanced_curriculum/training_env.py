"""Seeded two-level pillar-y curriculum for a later V5 PPO fine-tune."""

from curriculum import LEVELS, seeded_scenarios
from scenario_env import V5ScenarioEnv


class V5CurriculumEnv(V5ScenarioEnv):
    """Randomize only reset pose; never append world coordinates to observation."""

    def __init__(self, seed=29, level1_episodes=12):
        super().__init__()
        self.seed = int(seed)
        self.level1_episodes = int(level1_episodes)
        self.episode_index = 0
        self.sampled_scenes = []

    def reset(self):
        level = LEVELS[0] if self.episode_index < self.level1_episodes else LEVELS[1]
        # One deterministic sample per episode, independent of process timing.
        self.scenario = seeded_scenarios(
            level, 1, self.seed + self.episode_index
        )[0]
        self.sampled_scenes.append({
            'episode': self.episode_index + 1,
            'level': level.name,
            **self.scenario,
        })
        self.episode_index += 1
        return super().reset()

