from typing import List, Optional, Union
import os
import shutil

from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "AAViSSEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.PARALLEL_GPU_IDS = []
_C.MODEL_DIR = ''
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_OPTION = []
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.EXTRA_DEPTH = False
_C.DEBUG = False
_C.NUM_SOUNDS_IN_MIX = 2
_C.COMPUTE_EVAL_METRICS = False
_C.EVAL_METRICS_TO_COMPUTE = ['si_sdr',]
_C.EPS_SCENES = []
_C.EPS_SCENES_N_IDS = []
_C.JOB_ID = 1

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.num_updates_per_cycle = 1
_C.RL.PPO.pretrained_passive_separators_ckpt = ""
_C.RL.PPO.train_passive_separators = False
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.bin_separation_loss_coef = 1.0
_C.RL.PPO.mono_conversion_loss_coef = 1.0
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr_pol = 1e-3
_C.RL.PPO.lr_sep = 1e-3
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.nav_reward_weight = 0.0
_C.RL.PPO.sep_reward_weight = 1.0
_C.RL.PPO.extra_reward_multiplier = 10.0
_C.RL.PPO.deterministic_eval = False
_C.RL.PPO.use_ddppo = False
_C.RL.PPO.ddppo_distrib_backend = "NCCL"
_C.RL.PPO.short_rollout_threshold = 0.25
_C.RL.PPO.sync_frac = 0.6
_C.RL.PPO.master_port = 8738
_C.RL.PPO.master_addr = "127.0.0.1"
_C.RL.PPO.switch_policy = False
_C.RL.PPO.time_thres_for_pol_switch = 80

# -----------------------------------------------------------------------------
# Pretraining passive separator
# -----------------------------------------------------------------------------
_C.Pretrain = CN()
_C.Pretrain.Passive = CN()
_C.Pretrain.Passive.lr = 5.0e-4
_C.Pretrain.Passive.eps = 1.0e-5
_C.Pretrain.Passive.max_grad_norm = 0.8
_C.Pretrain.Passive.NUM_EPOCHS = 1000

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()

########## ACTIONS ###########
# -----------------------------------------------------------------------------
# PAUSE ACTION
# -----------------------------------------------------------------------------
_TC.TASK.ACTIONS.PAUSE = CN()
_TC.TASK.ACTIONS.PAUSE.TYPE = "PauseAction"

########## SENSORS ###########
# -----------------------------------------------------------------------------
# MIXED BINAURAL AUDIO MAGNITUDE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.MIXED_BIN_AUDIO_MAG_SENSOR = CN()
_TC.TASK.MIXED_BIN_AUDIO_MAG_SENSOR.TYPE = "MixedBinAudioMagSensor"
_TC.TASK.MIXED_BIN_AUDIO_MAG_SENSOR.FEATURE_SHAPE = [512, 32, 2]
# -----------------------------------------------------------------------------
# MIXED BINAURAL AUDIO PHASE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.MIXED_BIN_AUDIO_PHASE_SENSOR = CN()
_TC.TASK.MIXED_BIN_AUDIO_PHASE_SENSOR.TYPE = "MixedBinAudioPhaseSensor"
_TC.TASK.MIXED_BIN_AUDIO_PHASE_SENSOR.FEATURE_SHAPE = [512, 32, 2]
# -----------------------------------------------------------------------------
# GROUND-TRUTH MONO COMPONENTS SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.GT_MONO_COMPONENTS_SENSOR = CN()
_TC.TASK.GT_MONO_COMPONENTS_SENSOR.TYPE = "GtMonoComponentsSensor"
# default for 1 sound in the mixture ([mag, phase])
_TC.TASK.GT_MONO_COMPONENTS_SENSOR.FEATURE_SHAPE = [512, 32, 2]
# -----------------------------------------------------------------------------
# GROUND-TRUTH BINAURAL COMPONENTS SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.GT_BIN_COMPONENTS_SENSOR = CN()
_TC.TASK.GT_BIN_COMPONENTS_SENSOR.TYPE = "GtBinComponentsSensor"
# default for 1 sound in the mixture ([mag_l, phase_l, mag_r, phase_r])
_TC.TASK.GT_BIN_COMPONENTS_SENSOR.FEATURE_SHAPE = [512, 32, 4]
# -----------------------------------------------------------------------------
# TARGET CLASS SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.TARGET_CLASS_SENSOR = SIMULATOR_SENSOR.clone()
_TC.TASK.TARGET_CLASS_SENSOR.TYPE = "TargetClassSensor"

########## MEASURES IN INFO ###########
# -----------------------------------------------------------------------------
# Geodesic Distance to Target Audio Source Measure
# -----------------------------------------------------------------------------
_TC.TASK.GEODESIC_DISTANCE_TO_TARGET_AUDIO_SOURCE = CN()
_TC.TASK.GEODESIC_DISTANCE_TO_TARGET_AUDIO_SOURCE.TYPE = "GeoDistanceToTargetAudioSource"
# -----------------------------------------------------------------------------
# Normalized Geodesic Distance to Target Audio Source Measure
# -----------------------------------------------------------------------------
_TC.TASK.NORMALIZED_GEODESIC_DISTANCE_TO_TARGET_AUDIO_SOURCE = CN()
_TC.TASK.NORMALIZED_GEODESIC_DISTANCE_TO_TARGET_AUDIO_SOURCE.TYPE = "NormalizedGeoDistanceToTargetAudioSource"

# -----------------------------------------------------------------------------
# simulator config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.SEED = -1
_TC.SIMULATOR.SCENE_DATASET = "mp3d"
_TC.SIMULATOR.MAX_EPISODE_STEPS = 20
_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.RENDERED_OBSERVATIONS = "../sound_spaces/scene_observations_new/"

# -----------------------------------------------------------------------------
# audio config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.MONO_DIR = "data/audio_data/VoxCelebV1TenClasses_MITMusic_ESC50/train_preprocessed"
_TC.SIMULATOR.AUDIO.RIR_DIR = "../sound_spaces/binaural_rirs/mp3d"
_TC.SIMULATOR.AUDIO.META_DIR = "../sound_spaces/metadata/mp3d"
_TC.SIMULATOR.AUDIO.PASSIVE_DATASET_VERSION = "v1"
_TC.SIMULATOR.AUDIO.SOURCE_AGENT_LOCATION_DATAPOINTS_DIR = "data/passive_datasets/"
_TC.SIMULATOR.AUDIO.PASSIVE_TRAIN_AUDIO_DIR = "data/audio_data/VoxCelebV1TenClasses_MITMusic_ESC50/train_preprocessed"
_TC.SIMULATOR.AUDIO.PASSIVE_NONOVERLAPPING_VAL_AUDIO_DIR = "data/audio_data/VoxCelebV1TenClasses_MITMusic_ESC50/val_preprocessed"
_TC.SIMULATOR.AUDIO.NUM_PASSIVE_DATAPOINTS_PER_SCENE = 30000
_TC.SIMULATOR.AUDIO.NUM_PASSIVE_DATAPOINTS_PER_SCENE_EVAL = 1000
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.NUM_WORKER = 4
_TC.SIMULATOR.AUDIO.BATCH_SIZE = 128
_TC.SIMULATOR.AUDIO.GT_MONO_MAG_NORM = 0.0
_TC.SIMULATOR.AUDIO.NORM_TYPE = "l2"
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 16000

# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'


def merge_from_path(config, config_paths):
	"""
	merge config with configs from config paths
	:param config: original unmerged config
	:param config_paths: config paths to merge configs from
	:return: merged config
	"""
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	return config


def get_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None,
		model_dir: Optional[str] = None,
		run_type: Optional[str] = None
) -> CN:
	r"""Create a unified config with default values overwritten by values from
	`config_paths` and overwritten by options from `opts`.
	Args:
		config_paths: List of config paths or string that contains comma
		separated list of config paths.
		opts: Config options (keys, values) in a list (e.g., passed from
		command line into the config. For example, `opts = ['FOO.BAR',
		0.5]`. Argument can be used for parameter sweeping or quick tests.
		model_dir: suffix for output dirs
		run_type: either train or eval
	"""
	config = merge_from_path(_C.clone(), config_paths)
	config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

	if opts:
		config.CMD_TRAILING_OPTS = opts
		config.merge_from_list(opts)

	assert model_dir is not None, "set --model-dir"
	config.MODEL_DIR = model_dir
	config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, config.TENSORBOARD_DIR)
	config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
	config.LOG_FILE = os.path.join(config.MODEL_DIR, config.LOG_FILE)
	config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, 'data')

	dirs = [config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
	if run_type == 'train':
		# check dirs
		if any([os.path.exists(d) for d in dirs]):
			for d in dirs:
				if os.path.exists(d):
					print('{} exists'.format(d))
			key = input('Output directory already exists! Overwrite the folder? (y/n)')
			if key == 'y':
				for d in dirs:
					if os.path.exists(d):
						shutil.rmtree(d)

	config.TASK_CONFIG.defrost()
	config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV

	config.TASK_CONFIG.TASK.GT_MONO_COMPONENTS_SENSOR.FEATURE_SHAPE[2] *= config.NUM_SOUNDS_IN_MIX
	config.TASK_CONFIG.TASK.GT_BIN_COMPONENTS_SENSOR.FEATURE_SHAPE[2] *= config.NUM_SOUNDS_IN_MIX

	config.TASK_CONFIG.SIMULATOR.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

	if config.RL.PPO.switch_policy:
		config.EVAL.USE_CKPT_CONFIG = False
		config.NUM_PROCESSES = 1

	config.TASK_CONFIG.freeze()

	config.freeze()

	return config


def get_task_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None
) -> habitat.Config:
	r"""
	get config after merging configs stored in yaml files and command line arguments
	:param config_paths: paths to configs
	:param opts: optional command line arguments
	:return: merged config
	"""
	config = _TC.clone()
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	if opts:
		config.merge_from_list(opts)

	config.freeze()
	return config
