from typing import Any, Type, Union

import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode

from habitat.tasks.nav.nav import NavigationTask, Measure, EmbodiedTask, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)

from habitat.sims.habitat_simulator.actions import HabitatSimActions


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.TARGET_CLASS = episode.info[0]["target_label"]
        agent_cfg.AUDIO_SOURCE_POSITIONS = []
        for source in episode.goals:
            agent_cfg.AUDIO_SOURCE_POSITIONS.append(source.position)
        agent_cfg.SOUND_NAMES = []
        for source_info in episode.info:
            agent_cfg.SOUND_NAMES.append(source_info["sound"])
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="AAViSS")
class AAViSSTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return self._sim._is_episode_active


@registry.register_sensor
class MixedBinAudioMagSensor(Sensor):
    r"""Mixed binaural spectrogram magnitude at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mixed_bin_audio_mag"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_mixed_bin_audio_mag_spec()


@registry.register_sensor
class MixedBinAudioPhaseSensor(Sensor):
    r"""Mixed binaural spectrogram phase at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mixed_bin_audio_phase"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_mixed_bin_audio_phase_spec()


@registry.register_sensor
class GtMonoComponentsSensor(Sensor):
    r"""Ground truth monaural components at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_mono_comps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_gt_mono_audio_components()


@registry.register_sensor
class GtBinComponentsSensor(Sensor):
    r"""Ground truth binaural components at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_bin_comps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_gt_bin_audio_components()


@registry.register_sensor(name="TargetClassSensor")
class TargetClassSensor(Sensor):
    r"""Target class for the current episode
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "target_class"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return [self._sim.target_class]


@registry.register_measure
class GeoDistanceToTargetAudioSource(Measure):
    r"""Geodesic distance to target audio source for every time step
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "geo_distance_to_target_audio_source"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_episode_distance = episode.info[0]["geodesic_distance"]
        self._metric = None
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(
        self, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._metric = distance_to_target


@registry.register_measure
class NormalizedGeoDistanceToTargetAudioSource(Measure):
    r"""Normalized geodesic distance to target audio source for every time step
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "normalized_geo_distance_to_target_audio_source"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_episode_distance = episode.info[0]["geodesic_distance"]
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if self._start_end_episode_distance == 0:
            self._metric = -1
        else:
            self._metric = distance_to_target / self._start_end_episode_distance


@registry.register_task_action
class PauseAction(SimulatorTaskAction):
    name: str = "PAUSE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.PAUSE)
