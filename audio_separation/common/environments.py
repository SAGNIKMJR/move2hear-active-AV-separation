r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging

import habitat
from habitat import Config, Dataset
from audio_separation.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AAViSSEnv")
class AAViSSEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._config = config
        self._core_env_config = config.TASK_CONFIG
        self._goal_reached_once = False

        self._previous_target_distance = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._env_step = 0
        self._goal_reached_once = False
        observation = super().reset()
        logging.debug(super().current_episode)
        self._previous_target_distance = self.habitat_env.current_episode.info[0]["geodesic_distance"]
        return observation

    def step(self, *args, **kwargs):
        observation, reward, done, info = super().step(*args, **kwargs)
        self._env_step += 1
        return observation, reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0
        # for FarTgt
        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            (not self._env.sim._is_episode_active)
            and self._env.sim.reaching_goal
        ):
            return True
        return False

    def _goal_reached(self):
        if (
            self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
