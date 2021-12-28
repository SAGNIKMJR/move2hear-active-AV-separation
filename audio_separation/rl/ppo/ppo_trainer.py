import contextlib
import os
import time
import logging
from collections import deque
from typing import Dict
import json
import random
import pickle
import gzip

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from torch import distributed as distrib

from habitat import Config, logger
from audio_separation.common.base_trainer import BaseRLTrainer
from audio_separation.common.baseline_registry import baseline_registry
from audio_separation.common.env_utils import construct_envs, override_rewards
from audio_separation.common.environments import get_env_class
from audio_separation.common.rollout_storage import RolloutStoragePol, RolloutStorageSep
from audio_separation.common.tensorboard_utils import TensorboardWriter
from audio_separation.rl.ppo.ddppo_utils import (
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
)
from audio_separation.common.utils import (
    batch_obs,
    linear_decay,
)
from audio_separation.common.eval_metrics import STFT_L2_distance, compute_waveform_quality
from audio_separation.rl.ppo.policy import Move2HearPolicy
from audio_separation.rl.ppo.ppo import PPO, DDPPO


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO and DDPPO algorithm
    PPO paper: https://arxiv.org/abs/1707.06347.
    """
    # supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if self.config.RL.PPO.use_ddppo:
            interrupted_state = load_interrupted_state()
            if interrupted_state is not None:
                self.config = interrupted_state["config"]

    def _setup_actor_critic_agent(self, world_rank=0) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            world_rank: ddppo world rank

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG

        assert self.config.EXTRA_DEPTH or self.config.EXTRA_RGB, "set atleast one of EXTRA_RGB and EXTRA_DEPTH to true"

        freeze_passive_separators = ((ppo_cfg.pretrained_passive_separators_ckpt != "") and\
                                     (not ppo_cfg.train_passive_separators))
        # switch policy (nav. to quality-improvement) during eval for Far-target
        switch_policy = ppo_cfg.switch_policy

        if switch_policy:
            self.actor_critic_nav = Move2HearPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                goal_sensor_uuid=task_cfg.TASK.GOAL_SENSOR_UUID,
                hidden_size=ppo_cfg.hidden_size,
                extra_rgb=self.config.EXTRA_RGB,
                extra_depth=self.config.EXTRA_DEPTH,
                use_ddppo=ppo_cfg.use_ddppo,
                world_rank=world_rank,
            )
            self.actor_critic_nav.to(self.device)

            if ppo_cfg.use_ddppo:
                self.agent_nav = DDPPO(
                    actor_critic=self.actor_critic_nav,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )
            else:
                self.agent_nav = PPO(
                    actor_critic=self.actor_critic_nav,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )

            self.actor_critic_qualImprov = Move2HearPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                goal_sensor_uuid=task_cfg.TASK.GOAL_SENSOR_UUID,
                hidden_size=ppo_cfg.hidden_size,
                extra_rgb=self.config.EXTRA_RGB,
                extra_depth=self.config.EXTRA_DEPTH,
                use_ddppo=ppo_cfg.use_ddppo,
                world_rank=world_rank,
                # use_smartnav_for_eval_pol_mix=ppo_cfg.use_smartnav_for_eval_pol_mix,
            )
            self.actor_critic_qualImprov.to(self.device)

            if ppo_cfg.use_ddppo:
                self.agent_qualImprov = DDPPO(
                    actor_critic=self.actor_critic_qualImprov,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )
            else:
                self.agent_qualImprov = PPO(
                    actor_critic=self.actor_critic_qualImprov,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )
        else:
            self.actor_critic = Move2HearPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                goal_sensor_uuid=task_cfg.TASK.GOAL_SENSOR_UUID,
                hidden_size=ppo_cfg.hidden_size,
                extra_rgb=self.config.EXTRA_RGB,
                extra_depth=self.config.EXTRA_DEPTH,
                use_ddppo=ppo_cfg.use_ddppo,
                world_rank=world_rank,
            )
            self.actor_critic.to(self.device)

            if ppo_cfg.use_ddppo:
                self.agent = DDPPO(
                    actor_critic=self.actor_critic,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )
            else:
                self.agent = PPO(
                    actor_critic=self.actor_critic,
                    clip_param=ppo_cfg.clip_param,
                    ppo_epoch=ppo_cfg.ppo_epoch,
                    num_mini_batch=ppo_cfg.num_mini_batch,
                    value_loss_coef=ppo_cfg.value_loss_coef,
                    bin_separation_loss_coef=ppo_cfg.bin_separation_loss_coef,
                    mono_conversion_loss_coef=ppo_cfg.mono_conversion_loss_coef,
                    entropy_coef=ppo_cfg.entropy_coef,
                    lr_pol=ppo_cfg.lr_pol,
                    lr_sep=ppo_cfg.lr_sep,
                    eps=ppo_cfg.eps,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    freeze_passive_separators=freeze_passive_separators,
                )

        if switch_policy:
            self.actor_critic_nav.to(self.device)
            self.actor_critic_nav.train()
            self.actor_critic_qualImprov.to(self.device)
            self.actor_critic_qualImprov.train()
        else:
            self.actor_critic.to(self.device)
            self.actor_critic.train()

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
        self, rollouts_pol, rollouts_sep, current_episode_reward, current_episode_step, current_episode_dist_probs,
            current_episode_bin_losses, current_episode_mono_losses, current_episode_monoFromMem_losses, episode_rewards,
            episode_counts, episode_steps, episode_dist_probs, episode_bin_losses_allSteps, episode_mono_losses_lastStep,
            episode_mono_losses_allSteps, episode_monoFromMem_losses_lastStep, episode_monoFromMem_losses_allSteps,
            episode_ndgs, episode_dgs,
    ):
        r"""
        collects rollouts for training separator in supervised fashion and the policy with PPO
        :param rollouts_pol: rollout storage for policy
        :param rollouts_sep: rollout storage for separator
        :param current_episode_reward: reward for the current epispde
        :param current_episode_step: number of steps for the current episode
        :param current_episode_dist_probs: policy distribution for all actions for current episode
        :param current_episode_bin_losses: binaural losses for passive separator for current episode
        :param current_episode_mono_losses: monaural losses for passive separator for current episode
        :param current_episode_monoFromMem_losses: monaural losses on memory predictions for passive separator for current episode
        :param episode_rewards: rewards for all episodes
        :param episode_counts: number of all episodes
        :param episode_steps: number of episode steps for all episodes
        :param episode_dist_probs: policy distribution for all actions for all episodes
        :param episode_bin_losses_allSteps: binaural losses over all steps for passive separator for all episodes
        :param episode_mono_losses_lastStep: monaural losses at last step for passive separator for all episodes
        :param episode_mono_losses_allSteps: monaural losses over all steps for passive separator for all episodes
        :param episode_monoFromMem_losses_lastStep: monaural losses on memory predictions at last step for passive
                                                    separator for all episodes
        :param episode_monoFromMem_losses_allSteps: monaural losses on memory predictions over all steps for passive
                                                    separator for all episodes
        :param episode_ndgs: normalized distance to goal value for all episodes
        :param episode_dgs: distance
        :return: 1. pth_time: time needed for pytorch forward pass
                 2. env_time: time needed for environment simulation with Habitat
                 3. self.envs.num_envs: number of active environments in the simulator
        """
        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG
        pth_time = 0.0
        env_time = 0.0

        t_pred_current_step = time.time()
        # get binaural and mono predictions, and sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts_pol.step] for k, v in rollouts_pol.observations.items()
            }

            # passive-separate mono given target class
            pred_binSepMasks =\
                self.actor_critic.get_binSepMasks(
                    step_observation,
                )
            pred_mono =\
                self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                   mixed_audio=step_observation["mixed_bin_audio_mag"],
                                                   )

            # mask the previous memory_aggregated prediction with 0 only at episode reset
            prev_pred_monoFromMem_masked =\
                rollouts_pol.prev_pred_monoFromMem[rollouts_pol.step] *\
                rollouts_pol.masks[rollouts_pol.step].unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                       *pred_mono.size()[1:],
                                                                                       )
            pred_monoFromMem =\
                self.actor_critic.get_monoFromMem(
                    pred_mono.detach(),
                    prev_pred_monoFromMem_masked.detach()
                )

            # get actions
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states_pol,
                distribution_probs,
            ) = self.actor_critic.act(
                step_observation,
                rollouts_pol.recurrent_hidden_states_pol[rollouts_pol.step],
                rollouts_pol.masks[rollouts_pol.step],
                pred_binSepMasks=pred_binSepMasks,
                pred_mono=pred_mono,
                pred_monoFromMem=pred_monoFromMem,
            )

        pth_time += time.time() - t_pred_current_step
        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        env_time += time.time() - t_step_env

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations, self.device)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )
        ndgs = torch.tensor(
                [[info['normalized_geo_distance_to_target_audio_source']] for info in infos]
            )
        dgs = torch.tensor(
                [[info['geo_distance_to_target_audio_source']] for info in infos]
            )

        t_pred_next_step = time.time()
        # next step predictions needed to compute reward for present step
        with torch.no_grad():
            next_pred_binSepMasks =\
                self.actor_critic.get_binSepMasks(batch)
            next_pred_mono =\
                self.actor_critic.convert_bin2mono(next_pred_binSepMasks.detach(),
                                                   mixed_audio=batch["mixed_bin_audio_mag"],
                                                   )

            pred_monoFromMem_masked = pred_monoFromMem *\
                                      masks.to(self.device).unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                             *pred_monoFromMem.size()[1:])
            next_pred_monoFromMem =\
                self.actor_critic.get_monoFromMem(
                    next_pred_mono.detach(),
                    pred_monoFromMem_masked.detach()
                )
        pth_time += time.time() - t_pred_current_step
        t_update_stats = time.time()

        # target gt mono is always the first 2 channels among the components
        gt_mono_mag =\
            step_observation["gt_mono_comps"][..., 0::2].clone()[..., :1]
        next_gt_mono_mag =\
            batch["gt_mono_comps"][..., 0::2].clone()[..., :1]

        # for training quality-improvement policy. Hence, override nav2AudioTarget rewards with quality-improvement rewards.
        # this works because all processes have equal number of steps
        if (ppo_cfg.sep_reward_weight == 1.) and (ppo_cfg.nav_reward_weight == 0.):
            rewards = override_rewards(rewards,
                                       dones,
                                       next_pred_monoFromMem,
                                       next_gt_mono_mag.clone(),
                                       reward_type="quality_improvement",
                                       pred_monoFromMem=pred_monoFromMem,
                                       gt_mono_mag=gt_mono_mag.clone(),
                                       )

            if current_episode_step[0].item() == task_cfg.ENVIRONMENT.MAX_EPISODE_STEPS - 2:
                rewards_extra = override_rewards(rewards,
                                                 dones,
                                                 next_pred_monoFromMem,
                                                 next_gt_mono_mag.clone(),
                                                 reward_type="extra",
                                                 extra_reward_multiplier=ppo_cfg.extra_reward_multiplier,
                                                 )

                assert len(rewards_extra) == len(rewards)
                rewards = (np.array(rewards) + np.array(rewards_extra)).tolist()

        _, monoFromMem_losses =\
             STFT_L2_distance(step_observation["mixed_bin_audio_mag"],
                              pred_binSepMasks.detach(),
                              step_observation["gt_bin_comps"].clone(),
                              pred_monoFromMem,
                              step_observation["gt_mono_comps"].clone(),
                              )
        bin_losses, mono_losses =\
             STFT_L2_distance(step_observation["mixed_bin_audio_mag"],
                              pred_binSepMasks.detach(),
                              step_observation["gt_bin_comps"].clone(),
                              pred_mono,
                              step_observation["gt_mono_comps"].clone(),
                              )

        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        current_episode_reward += rewards
        current_episode_step += 1
        current_episode_dist_probs += distribution_probs.detach().cpu()
        current_episode_bin_losses += bin_losses
        current_episode_mono_losses += mono_losses
        current_episode_monoFromMem_losses += monoFromMem_losses

        # current_episode_reward is accumulating rewards across multiple updates,
        # as long as the current episode is not finished
        # the current episode reward is added to the episode rewards only if the current episode is done
        # the episode count will also increase by 1
        episode_rewards += (1 - masks) * current_episode_reward
        episode_ndgs += (1 - masks) * ndgs
        episode_dgs += (1 - masks) * dgs
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_dist_probs += (1 - masks) * (current_episode_dist_probs / current_episode_step)
        episode_bin_losses_allSteps += (1 - masks) * (current_episode_bin_losses / current_episode_step)
        episode_mono_losses_lastStep += (1 - masks) * mono_losses
        episode_mono_losses_allSteps += (1 - masks) * (current_episode_mono_losses / current_episode_step)
        episode_monoFromMem_losses_lastStep += (1 - masks) * monoFromMem_losses
        episode_monoFromMem_losses_allSteps += (1 - masks) * (current_episode_monoFromMem_losses / current_episode_step)

        # zeroing out current values when done
        current_episode_reward *= masks
        current_episode_step *= masks
        current_episode_bin_losses *= masks
        current_episode_mono_losses *= masks
        current_episode_monoFromMem_losses *= masks
        current_episode_dist_probs *= masks

        rollouts_pol.insert(
            batch,
            recurrent_hidden_states_pol,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
            pred_binSepMasks=pred_binSepMasks,
            pred_mono=pred_mono,
            pred_monoFromMem=pred_monoFromMem,
        )

        rollouts_sep.insert(
            batch,
            masks,
            pred_monoFromMem=pred_monoFromMem,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_pol(self, rollouts_pol):
        """
        updates Move2Hear policy
        :param rollouts_pol: rollout storage for the policy
        :return: 1. time.time() - t_update_model: time needed for policy update
                 2. value_loss: PPO value loss in this update
                 3. action_loss: PPO actions loss in this update
                 4. dist_entropy: PPO entropy loss in this update
        """
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts_pol.observations.items()
            }
            # while training the qual-improv policy, this doesn't really compute the next value, but the last value of
            # the current episode. However, it doesn't matter because for qual-improvment, T=20 and ppo_cfg.num_steps=20.
            # So, compute_returns in the next step multiplies last_value with 0 since it's the next_value is expected
            # to be from a new episode. If T != ppo_cfg.num_steps, this could lead to a bug. It doesn't matter while
            # training the nav policy
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts_pol.recurrent_hidden_states_pol[-1],
                rollouts_pol.masks[-1],
                pred_binSepMasks=rollouts_pol.pred_binSepMasks[-1],
                pred_mono=rollouts_pol.pred_mono[-1],
                pred_monoFromMem=rollouts_pol.prev_pred_monoFromMem[-1],
            ).detach()

        rollouts_pol.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )
        value_loss, action_loss, dist_entropy = self.agent.update_pol(rollouts_pol)
        rollouts_pol.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _update_sep(self, rollouts_sep):
        """
        updates Move2Hear acoustic memory (passive separators frozen)
        :param rollouts_sep:
        :return:    1. time.time() - t_update_model: time needed for separator (acoustic memory) update
                    2. bin_loss: binaural loss for the passive separator in this update (for debugging)
                    3. mono_loss: monaural loss for the passive separator in this update (for debugging)
                    4. monoFromMem_loss: computed on the output of the acoustic memory)
        """
        t_update_model = time.time()

        bin_loss, mono_loss, monoFromMem_loss = self.agent.update_sep(rollouts_sep)
        rollouts_sep.after_update()

        return (
            time.time() - t_update_model,
            bin_loss,
            mono_loss,
            monoFromMem_loss
        )

    def _load_pretrained_passive_separators(self):
        r"""
        loads pretrained passive separators and freezes them for final Move2Hear training
        :return: None
        """
        ppo_cfg = self.config.RL.PPO

        assert ppo_cfg.pretrained_passive_separators_ckpt != ""
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(ppo_cfg.pretrained_passive_separators_ckpt, map_location="cpu")
        self.agent.load_pretrained_passive_separators(ckpt_dict["state_dict"])

        # freezing parameters of passive binaural separator
        assert hasattr(self.agent.actor_critic, "binSep_enc")
        self.agent.actor_critic.binSep_enc.eval()
        for param in self.agent.actor_critic.binSep_enc.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
        assert hasattr(self.agent.actor_critic, "binSep_dec")
        self.agent.actor_critic.binSep_dec.eval()
        for param in self.agent.actor_critic.binSep_dec.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

        # freezing parameters of passive bin2mono converter
        assert hasattr(self.agent.actor_critic, "bin2mono_enc")
        self.agent.actor_critic.bin2mono_enc.eval()
        for param in self.agent.actor_critic.bin2mono_enc.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
        assert hasattr(self.agent.actor_critic, "bin2mono_dec")
        self.agent.actor_critic.bin2mono_dec.eval()
        for param in self.agent.actor_critic.bin2mono_dec.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

    def train(self) -> None:
        r"""Main method for cyclic training of Move2Hear policy and separator.

        Returns:
            None
        """
        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG

        if ppo_cfg.use_ddppo:
            self.local_rank, tcp_store =\
                init_distrib_slurm(
                    ppo_cfg.ddppo_distrib_backend,
                    master_port=ppo_cfg.master_port,
                    master_addr=ppo_cfg.master_addr,
                )
            add_signal_handlers()

            num_rollouts_done_store = distrib.PrefixStore(
                "rollout_tracker", tcp_store
            )
            num_rollouts_done_store.set("num_done", "0")

            self.world_rank = distrib.get_rank()
            self.world_size = distrib.get_world_size()

            self.config.defrost()
            self.config.TORCH_GPU_ID = self.local_rank
            self.config.SIMULATOR_GPU_ID = self.local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.SEED += (
                self.world_rank * self.config.NUM_PROCESSES
            )
            self.config.TASK_CONFIG.SIMULATOR.SEED = self.config.SEED
            self.config.freeze()

        if (not ppo_cfg.use_ddppo) or (ppo_cfg.use_ddppo and self.world_rank == 0):
            logger.info(f"config: {self.config}")

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True if ppo_cfg.use_ddppo else False,
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if ppo_cfg.use_ddppo:
            torch.cuda.set_device(self.device)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(world_rank=self.world_rank if ppo_cfg.use_ddppo else 0)
        self._load_pretrained_passive_separators()

        if ppo_cfg.use_ddppo:
            self.agent.init_distributed(find_unused_params=True)
        if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(param.numel() for param in self.agent.parameters() if param.requires_grad)
                )
            )

        rollouts_pol = RolloutStoragePol(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts_sep = RolloutStorageSep(
            ppo_cfg.num_steps * ppo_cfg.num_updates_per_cycle,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
        )
        rollouts_pol.to(self.device)
        rollouts_sep.to(self.device)

        observations = self.envs.reset()
        if ppo_cfg.use_ddppo:
            batch = batch_obs(observations, device=self.device)
        else:
            batch = batch_obs(observations)

        for sensor in rollouts_pol.observations:
            rollouts_pol.observations[sensor][0].copy_(batch[sensor])
            rollouts_sep.observations[sensor][0].copy_(batch[sensor])

        # episode_x accumulates over the entire training course
        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_dist_probs = torch.zeros(self.envs.num_envs, self.envs.action_spaces[0].n)
        episode_bin_losses_allSteps = torch.zeros(self.envs.num_envs, 1)
        episode_mono_losses_lastStep = torch.zeros(self.envs.num_envs, 1)
        episode_mono_losses_allSteps = torch.zeros(self.envs.num_envs, 1)
        episode_monoFromMem_losses_lastStep = torch.zeros(self.envs.num_envs, 1)
        episode_monoFromMem_losses_allSteps = torch.zeros(self.envs.num_envs, 1)
        episode_ndgs = torch.zeros(self.envs.num_envs, 1)
        episode_dgs = torch.zeros(self.envs.num_envs, 1)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        current_episode_dist_probs = torch.zeros(self.envs.num_envs, self.envs.action_spaces[0].n)
        current_episode_bin_losses = torch.zeros(self.envs.num_envs, 1)
        current_episode_mono_losses = torch.zeros(self.envs.num_envs, 1)
        current_episode_monoFromMem_losses = torch.zeros(self.envs.num_envs, 1)

        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_dist_probs = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_bin_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_mono_losses_lastStep = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_mono_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_monoFromMem_losses_lastStep = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_monoFromMem_losses_allSteps = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_ndg = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_dg = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler_pol = LambdaLR(
            optimizer=self.agent.optimizer_pol,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        lr_scheduler_sep = LambdaLR(
            optimizer=self.agent.optimizer_sep,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        if ppo_cfg.use_ddppo:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) if self.world_rank == 0 else contextlib.suppress()
        else:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )

        with writer_obj as writer:
            for update in range(int(self.config.NUM_UPDATES / ppo_cfg.num_updates_per_cycle)):
                count_steps_lst = []
                for sub_update in range(int(ppo_cfg.num_updates_per_cycle)):
                    actual_update_num = update * ppo_cfg.num_updates_per_cycle + sub_update
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler_pol.step()
                    if ppo_cfg.use_linear_clip_decay:
                        self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                            actual_update_num, self.config.NUM_UPDATES
                        )

                    if ppo_cfg.use_ddppo:
                        count_steps_delta = 0

                    for step in range(ppo_cfg.num_steps):
                        delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                            rollouts_pol,
                            rollouts_sep,
                            current_episode_reward,
                            current_episode_step,
                            current_episode_dist_probs,
                            current_episode_bin_losses,
                            current_episode_mono_losses,
                            current_episode_monoFromMem_losses,
                            episode_rewards,
                            episode_counts,
                            episode_steps,
                            episode_dist_probs,
                            episode_bin_losses_allSteps,
                            episode_mono_losses_lastStep,
                            episode_mono_losses_allSteps,
                            episode_monoFromMem_losses_lastStep,
                            episode_monoFromMem_losses_allSteps,
                            episode_ndgs,
                            episode_dgs,
                        )

                        pth_time += delta_pth_time
                        env_time += delta_env_time
                        if ppo_cfg.use_ddppo:
                            count_steps_delta += delta_steps
                            if (
                                step
                                >= ppo_cfg.num_steps * ppo_cfg.short_rollout_threshold
                            ) and int(num_rollouts_done_store.get("num_done")) > (
                                ppo_cfg.sync_frac * self.world_size
                            ):
                                break
                        else:
                            count_steps += delta_steps

                    if ppo_cfg.use_ddppo:
                        num_rollouts_done_store.add("num_done", 1)

                    delta_pth_time, value_loss, action_loss, dist_entropy = self._update_pol(
                        rollouts_pol
                    )
                    pth_time += delta_pth_time

                    # computing stats
                    if ppo_cfg.use_ddppo:
                        stat_idx = 0
                        stat_idx_num_actions = 0
                        stat_name_to_idx = {}
                        stat_name_to_idx_num_actions = {}
                        stack_lst_for_stats = []
                        stack_lst_for_stats_num_actions = []

                        stack_lst_for_stats.append(episode_rewards)
                        stat_name_to_idx["rewards"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats.append(episode_counts)
                        stat_name_to_idx["counts"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats.append(episode_steps)
                        stat_name_to_idx["steps"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats_num_actions.append(episode_dist_probs)
                        stat_name_to_idx_num_actions["dist_probs"] = stat_idx_num_actions
                        stat_idx_num_actions += 1

                        stack_lst_for_stats.append(episode_bin_losses_allSteps)
                        stat_name_to_idx["avg_bin_losses_allSteps"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats.append(episode_mono_losses_lastStep)
                        stat_name_to_idx["mono_losses_lastStep"] = stat_idx
                        stat_idx += 1
                        stack_lst_for_stats.append(episode_mono_losses_allSteps)
                        stat_name_to_idx["avg_mono_losses_allSteps"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats.append(episode_monoFromMem_losses_lastStep)
                        stat_name_to_idx["monoFromMem_losses_lastStep"] = stat_idx
                        stat_idx += 1
                        stack_lst_for_stats.append(episode_monoFromMem_losses_allSteps)
                        stat_name_to_idx["avg_monoFromMem_losses_allSteps"] = stat_idx
                        stat_idx += 1

                        stack_lst_for_stats.append(episode_ndgs)
                        stat_name_to_idx["ndgs"] = stat_idx
                        stat_idx += 1
                        stack_lst_for_stats.append(episode_dgs)
                        stat_name_to_idx["dgs"] = stat_idx
                        stat_idx += 1

                        stats = torch.stack(stack_lst_for_stats, 0).to(self.device)
                        distrib.all_reduce(stats)
                        stats_num_actions = torch.stack(stack_lst_for_stats_num_actions, 0).to(self.device)
                        distrib.all_reduce(stats_num_actions)

                        window_episode_reward.append(stats[stat_name_to_idx["rewards"]].clone())
                        window_episode_counts.append(stats[stat_name_to_idx["counts"]].clone())
                        window_episode_step.append(stats[stat_name_to_idx["steps"]].clone())
                        window_episode_dist_probs.append(stats_num_actions[stat_name_to_idx_num_actions["dist_probs"]].clone())
                        window_episode_bin_losses_allSteps.append(stats[stat_name_to_idx["avg_bin_losses_allSteps"]].clone())
                        window_episode_mono_losses_lastStep.append(stats[stat_name_to_idx["mono_losses_lastStep"]].clone())
                        window_episode_mono_losses_allSteps.append(stats[stat_name_to_idx["avg_mono_losses_allSteps"]].clone())
                        window_episode_monoFromMem_losses_lastStep.append(stats[stat_name_to_idx["monoFromMem_losses_lastStep"]].clone())
                        window_episode_monoFromMem_losses_allSteps.append(stats[stat_name_to_idx["avg_monoFromMem_losses_allSteps"]].clone())
                        window_episode_ndg.append(stats[stat_name_to_idx["ndgs"]].clone())
                        window_episode_dg.append(stats[stat_name_to_idx["dgs"]].clone())

                        stats = torch.tensor(
                            [value_loss, action_loss, dist_entropy, count_steps_delta], device=self.device,
                        )
                        distrib.all_reduce(stats)
                        count_steps += stats[3].item()

                        if self.world_rank == 0:
                            num_rollouts_done_store.set("num_done", "0")
                            value_loss = stats[0].item() / self.world_size
                            action_loss = stats[1].item() / self.world_size
                            dist_entropy = stats[2].item() / self.world_size
                    else:
                        window_episode_reward.append(episode_rewards.clone())
                        window_episode_counts.append(episode_counts.clone())
                        window_episode_step.append(episode_steps.clone())
                        window_episode_dist_probs.append(episode_dist_probs.clone())
                        window_episode_bin_losses_allSteps.append(episode_bin_losses_allSteps.clone())
                        window_episode_mono_losses_lastStep.append(episode_mono_losses_lastStep.clone())
                        window_episode_mono_losses_allSteps.append(episode_mono_losses_allSteps.clone())
                        window_episode_monoFromMem_losses_lastStep.append(episode_monoFromMem_losses_lastStep.clone())
                        window_episode_monoFromMem_losses_allSteps.append(episode_monoFromMem_losses_allSteps.clone())
                        window_episode_ndg.append(episode_ndgs.clone())
                        window_episode_dg.append(episode_dgs.clone())

                    if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
                        stats_keys = ["count", "reward", "step", 'dist_probs', 'avg_bin_loss_allSteps',
                                      'mono_loss_lastStep', 'mono_loss_allSteps', 'monoFromMem_loss_lastStep',
                                      'monoFromMem_loss_allSteps', 'normalized_geo_distance_to_target_audio_source', "geo_distance_to_target_audio_source"]
                        stats_vals = [window_episode_counts, window_episode_reward, window_episode_step, window_episode_dist_probs,
                                      window_episode_bin_losses_allSteps, window_episode_mono_losses_lastStep,
                                      window_episode_mono_losses_allSteps, window_episode_monoFromMem_losses_lastStep,
                                      window_episode_monoFromMem_losses_allSteps, window_episode_ndg, window_episode_dg]
                        stats = zip(stats_keys, stats_vals)

                        deltas = {}
                        for k, v in stats:
                            if len(v) > 1:
                                deltas[k] = (v[-1] - v[0]).sum(dim=0)\
                                    if (k == "dist_probs")\
                                    else (v[-1] - v[0]).sum().item()
                            else:
                                deltas[k] = v[0].sum(dim=0) if (k == "dist_probs")\
                                    else v[0].sum().item()

                        deltas["count"] = max(deltas["count"], 1.0)
                        count_steps_lst.append(count_steps)

                        # this reward is averaged over all the episodes happened during window_size updates
                        # approximately number of steps is window_size * num_steps
                        writer.add_scalar(
                            "Environment/Reward", deltas["reward"] / deltas["count"], count_steps
                        )
                        logging.debug('Number of steps: {}'.format(deltas["step"] / deltas["count"]))
                        writer.add_scalar(
                            "Environment/Episode_length", deltas["step"] / deltas["count"], count_steps
                        )
                        for i in range(self.envs.action_spaces[0].n):
                            if not isinstance(deltas['dist_probs'] / deltas["count"], float):
                                writer.add_scalar(
                                    "Policy/Action_prob_{}".format(i), (deltas['dist_probs'] / deltas["count"])[i].item(),
                                    count_steps
                                )
                            else:
                                writer.add_scalar(
                                    "Policy/Action_prob_{}".format(i), deltas['dist_probs'] / deltas["count"], count_steps
                                )
                        writer.add_scalar(
                            "Environment/STFT_L2_loss/mono_lastStep", deltas['mono_loss_lastStep'] / deltas["count"],
                            count_steps
                        )
                        writer.add_scalar(
                            "Environment/STFT_L2_loss/mono_avgAllSteps", deltas['mono_loss_allSteps'] / deltas["count"],
                            count_steps
                        )
                        writer.add_scalar(
                            "Environment/STFT_L2_loss/monoFromMem_lastStep", deltas['monoFromMem_loss_lastStep'] / deltas["count"],
                            count_steps
                        )
                        writer.add_scalar(
                            "Environment/STFT_L2_loss/monoFromMem_avgAllSteps", deltas['monoFromMem_loss_allSteps'] / deltas["count"],
                            count_steps
                        )
                        writer.add_scalar(
                            "Environment/Normalized_geo_distance_to_target_audio_source", deltas["normalized_geo_distance_to_target_audio_source"] / deltas["count"], count_steps
                        )

                        writer.add_scalar(
                            "Environment/Geo_distance_to_target_audio_source", deltas["geo_distance_to_target_audio_source"] / deltas["count"], count_steps
                        )

                        writer.add_scalar(
                            'Policy/Value_Loss', value_loss, count_steps
                        )
                        writer.add_scalar(
                            'Policy/Action_Loss', action_loss, count_steps
                        )
                        writer.add_scalar(
                            'Policy/Entropy', dist_entropy, count_steps
                        )
                        writer.add_scalar(
                            'Policy/Learning_Rate', lr_scheduler_pol.get_lr()[0], count_steps
                        )

                        # log stats
                        if (actual_update_num > 0) and (actual_update_num % self.config.LOG_INTERVAL == 0):

                            window_rewards = (
                                window_episode_reward[-1] - window_episode_reward[0]
                            ).sum()
                            window_counts = (
                                window_episode_counts[-1] - window_episode_counts[0]
                            ).sum()

                            if window_counts > 0:
                                logger.info(
                                    "Average window size {} reward: {:3f}".format(
                                        len(window_episode_reward),
                                        (window_rewards / window_counts).item(),
                                    )
                                )
                            else:
                                logger.info("No episodes finish in current window")

                for sub_update in range(int(ppo_cfg.num_updates_per_cycle)):
                    actual_update_num = update * ppo_cfg.num_updates_per_cycle + sub_update
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler_sep.step()

                    delta_pth_time, bin_loss, mono_loss, monoFromMem_loss = self._update_sep(
                        rollouts_sep,
                    )
                    if ppo_cfg.use_ddppo:
                        sep_loss_stats = torch.tensor(
                            [bin_loss, mono_loss, monoFromMem_loss],
                        )
                    pth_time += delta_pth_time

                    if ppo_cfg.use_ddppo:
                        distrib.all_reduce(sep_loss_stats.to(self.device))

                    if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
                        if (actual_update_num > 0) and (actual_update_num % self.config.LOG_INTERVAL == 0):
                            logger.info(
                                "update: {}\tfps: {:.3f}\t".format(
                                    actual_update_num, count_steps_lst[sub_update] / (time.time() - t_start)
                                )
                            )
                            logger.info(
                                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                                "frames: {}".format(
                                    actual_update_num, env_time, pth_time, count_steps_lst[sub_update]
                                )
                            )
                        if actual_update_num % self.config.CHECKPOINT_INTERVAL == 0:
                            self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                            count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        # time switch options taken from eval_cfg; flag set so early because it decides config loading
        switch_policy_flag = self.config.RL.PPO.switch_policy

        # setting up config
        if switch_policy_flag:
            # using config from audio-navigation policy part of the checkpoint
            config = self._setup_eval_config(ckpt_dict["config_nav"])
        else:
            if self.config.EVAL.USE_CKPT_CONFIG:
                config = self._setup_eval_config(ckpt_dict["config"])
            else:
                config = self.config.clone()

        assert config.NUM_PROCESSES == 1, "TODO: multi-process eval"

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        # eval on those scenes only whose names are given in the eval config
        if len(config.EPS_SCENES) != 0:
            full_dataset_path = os.path.join(config.TASK_CONFIG.DATASET.DATA_PATH.split("{")[0],
                                             config.TASK_CONFIG.DATASET.VERSION,
                                             config.TASK_CONFIG.DATASET.SPLIT,
                                             f"{config.TASK_CONFIG.DATASET.SPLIT}.json.gz")
            with gzip.GzipFile(full_dataset_path, "rb") as fo:
                dataset = fo.read()
            dataset = dataset.decode("utf-8")
            dataset = json.loads(dataset)
            dataset_episodes = dataset["episodes"]

            eval_episode_count = 0
            for scene in config.EPS_SCENES:
                for episode in dataset_episodes:
                    if episode["scene_id"].split("/")[0] == scene:
                        eval_episode_count += 1

            if config.EVAL_EPISODE_COUNT > eval_episode_count:
                config.defrost()
                config.EVAL_EPISODE_COUNT = eval_episode_count
                config.freeze()

        logger.info(f"env config: {config}")

        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG

        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )

        self._setup_actor_critic_agent(ppo_cfg)

        ########### TEMPORARY VARIABLES / OBJECTS FOR EVAL ########
        # loading trained weights to policies,  creating empty tensors for eval and setting flags for policy switching in eval
        if switch_policy_flag:
            # navigation policy
            self.agent_nav.load_state_dict(ckpt_dict["state_dict_nav"])
            self.actor_critic_nav = self.agent_nav.actor_critic
            self.actor_critic_nav.eval()
            self.agent_nav.eval()
            self.agent_nav.actor_critic.eval()

            not_done_masks_nav = torch.ones(
                config.NUM_PROCESSES, 1, device=self.device
            )
            test_recurrent_hidden_states_nav_pol = torch.zeros(
                self.actor_critic_nav.pol_net.num_recurrent_layers,
                config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )

            # quality improvement policy
            self.agent_qualImprov.load_state_dict(ckpt_dict["state_dict_qualImprov"])
            self.actor_critic_qualImprov = self.agent_qualImprov.actor_critic
            self.actor_critic_qualImprov.eval()
            self.agent_qualImprov.eval()
            self.agent_qualImprov.actor_critic.eval()

            not_done_masks_qualImprov = torch.ones(
                config.NUM_PROCESSES, 1, device=self.device
            )
            test_recurrent_hidden_states_qualImprov_pol = torch.zeros(
                self.actor_critic_qualImprov.pol_net.num_recurrent_layers,
                config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )

            # flags needed for policy switch in eval
            time_thres_for_pol_switch = ppo_cfg.time_thres_for_pol_switch
        else:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic
            self.actor_critic.eval()
            self.agent.eval()
            self.agent.actor_critic.eval()

            not_done_masks = torch.ones(
                config.NUM_PROCESSES, 1, device=self.device
            )
            test_recurrent_hidden_states_pol = torch.zeros(
                self.actor_critic.pol_net.num_recurrent_layers,
                config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )

        prev_pred_monoFromMem =\
            torch.zeros(config.NUM_PROCESSES,
                        self.envs.observation_spaces[0].spaces["gt_mono_comps"].shape[0],
                        self.envs.observation_spaces[0].spaces["gt_mono_comps"].shape[1],
                        1,
                        device=self.device)

        t = tqdm(total=config.EVAL_EPISODE_COUNT)
        active_envs = list(range(self.envs.num_envs))
        # these are particularly useful for multi-process eval but code doesn't support it for now (feel free to start a PR
        # to include changes for multi-process eval)
        step_count_all_processes = torch.zeros(config.NUM_PROCESSES, 1)
        episode_count_all_processes = torch.zeros(config.NUM_PROCESSES, 1)
        num_episode_numbers_taken = config.NUM_PROCESSES - 1
        for episode_count_idx in range(config.NUM_PROCESSES):
            episode_count_all_processes[episode_count_idx] = episode_count_idx

        ########### EVAL METRICS ########
        self.metric_uuids = []
        # get name of metric ... these performance metrics are returned by the simulator
        for metric_name in task_cfg.TASK.MEASUREMENTS:
            metric_cfg = getattr(task_cfg.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())
        if "audio_comps_names" in self.metric_uuids:
            raise ValueError("remove this mofo from task_cfg, task.py, simulator_eval.py")
        # dict of dicts that stores stats of simulator-returned performance metrics per episode
        stats_episodes = dict()

        # eval metrics (STFT losses) for logging to log and comparing models without post-processing and
        # statistical-significance test
        mono_losses_last_step = []
        mono_losses_all_steps = []
        mono_loss_this_episode = 0.
        monoFromMem_losses_last_step = []
        monoFromMem_losses_all_steps = []
        monoFromMem_loss_this_episode = 0.

        # qualitative eval metrics and STFT losses per step per episode for dumping to storage, post-processing and
        # comparing models via statistical-significance test
        if config.COMPUTE_EVAL_METRICS:
            eval_metrics_toDump = {"mono": {}, "monoFromMem": {}}
            for metric in config.EVAL_METRICS_TO_COMPUTE:
                eval_metrics_toDump["mono"][metric] = {}
                eval_metrics_toDump["monoFromMem"][metric] = {}
            eval_metrics_toDump["mono"]["STFT_L2_loss"] = {}
            eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"] = {}

        # resetting environments for 1st step of eval
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        # looping over episodes
        while (
            len(stats_episodes) < config.EVAL_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            ### ALL CODE HERE ONWARDS ASSUME 1-PROCESS EVAL
            # scene and episode id
            current_scene = current_episodes[0].scene_id.split('/')[-2]
            current_episode_id = current_episodes[0].episode_id

            # episode and step count
            current_episode_count = int(episode_count_all_processes[0].item())
            current_step_count = int(step_count_all_processes[0].item())

            # particularly useful for multi-process eval
            # hack to not collect stats from environments which have finished
            active_envs_tmp = []
            for env_idx in active_envs:
                if env_idx > 0:
                    raise NotImplementedError
                if (current_episodes[env_idx].scene_id.split('/')[-2], current_episodes[env_idx].episode_id)\
                        not in stats_episodes:
                    active_envs_tmp.append(env_idx)
            active_envs = active_envs_tmp

            # passive-separate mono given target class
            with torch.no_grad():
                if switch_policy_flag:
                    # switch control  from nav to qualImprov policy after X stepss
                    if current_step_count < time_thres_for_pol_switch:
                        do_nav = True
                        do_qualImprov = False
                    else:
                        do_nav = False
                        do_qualImprov = True

                    if do_nav:
                        # passive-separate mono given target class
                        pred_binSepMasks =\
                            self.actor_critic_nav.get_binSepMasks(
                                batch,
                            )
                        pred_mono =\
                            self.actor_critic_nav.convert_bin2mono(pred_binSepMasks.detach(),
                                                                   mixed_audio=batch["mixed_bin_audio_mag"],
                                                                   )

                        # mask the previous memory_aggregated prediction with 0 only at episode reset
                        prev_pred_monoFromMem_masked = prev_pred_monoFromMem *\
                                                       not_done_masks_nav.unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                                           *pred_mono.size()[1:],
                                                                                                           )
                        pred_monoFromMem =\
                            self.actor_critic_nav.get_monoFromMem(
                                pred_mono.detach(),
                                prev_pred_monoFromMem_masked.detach()
                            )

                        # get actions
                        _, actions, _, test_recurrent_hidden_states_nav_pol, _ =\
                            self.actor_critic_nav.act(
                                batch,
                                test_recurrent_hidden_states_nav_pol,
                                not_done_masks_nav,
                                deterministic=ppo_cfg.deterministic_eval,
                                pred_binSepMasks=pred_binSepMasks,
                                pred_mono=pred_mono,
                                pred_monoFromMem=pred_monoFromMem,
                            )
                    elif do_qualImprov:
                        # passive-separate mono given target class
                        pred_binSepMasks =\
                            self.actor_critic_qualImprov.get_binSepMasks(
                                batch,
                            )
                        pred_mono =\
                            self.actor_critic_qualImprov.convert_bin2mono(pred_binSepMasks.detach(),
                                                                          mixed_audio=batch["mixed_bin_audio_mag"],
                                                                          )

                        # mask the previous memory_aggregated prediction with 0 only at episode reset..use not_done_masks_nav
                        # here too because that's always gets reset at the end of an episode
                        prev_pred_monoFromMem_masked = prev_pred_monoFromMem *\
                                                       not_done_masks_nav.unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                                           *pred_mono.size()[1:],
                                                                                                           )
                        pred_monoFromMem =\
                            self.actor_critic_qualImprov.get_monoFromMem(
                                pred_mono.detach(),
                                prev_pred_monoFromMem_masked.detach()
                            )

                        # get actions
                        _, actions, _, test_recurrent_hidden_states_qualImprov_pol, _ =\
                            self.actor_critic_qualImprov.act(
                                batch,
                                test_recurrent_hidden_states_qualImprov_pol,
                                not_done_masks_qualImprov,
                                deterministic=ppo_cfg.deterministic_eval,
                                pred_binSepMasks=pred_binSepMasks,
                                pred_mono=pred_mono,
                                pred_monoFromMem=pred_monoFromMem,
                            )
                else:
                    # passive-separate mono given target class
                    pred_binSepMasks =\
                        self.actor_critic.get_binSepMasks(
                            batch
                        )
                    pred_mono =\
                        self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                           mixed_audio=batch["mixed_bin_audio_mag"],
                                                           )

                    # mask the previous memory_aggregated prediction with 0 only at episode reset
                    prev_pred_monoFromMem_masked = prev_pred_monoFromMem *\
                                                   not_done_masks.unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                                   *pred_mono.size()[1:],
                                                                                                   )
                    pred_monoFromMem =\
                        self.actor_critic.get_monoFromMem(
                            pred_mono.detach(),
                            prev_pred_monoFromMem_masked.detach()
                        )

                    # get actions
                    _, actions, _, test_recurrent_hidden_states_pol, _ =\
                        self.actor_critic.act(
                            batch,
                            test_recurrent_hidden_states_pol,
                            not_done_masks,
                            deterministic=ppo_cfg.deterministic_eval,
                            pred_binSepMasks=pred_binSepMasks,
                            pred_mono=pred_mono,
                            pred_monoFromMem=pred_monoFromMem,
                        )

            prev_pred_monoFromMem = pred_monoFromMem

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if switch_policy_flag:
                not_done_masks_nav = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )
                if do_qualImprov:
                    not_done_masks_qualImprov = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                        device=self.device,
                    )
            else:
                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

            # compute STFT L2 losses
            _, monoFromMem_losses =\
                STFT_L2_distance(batch["mixed_bin_audio_mag"],
                                 pred_binSepMasks.detach(),
                                 batch["gt_bin_comps"].clone(),
                                 pred_monoFromMem,
                                 batch["gt_mono_comps"].clone(),
                                 )
            monoFromMem_loss_this_episode += monoFromMem_losses[0][0].item()

            bin_losses, mono_losses =\
                STFT_L2_distance(batch["mixed_bin_audio_mag"],
                                 pred_binSepMasks.detach(),
                                 batch["gt_bin_comps"].clone(),
                                 pred_mono,
                                 batch["gt_mono_comps"].clone(),
                                 )
            mono_loss_this_episode += mono_losses[0][0].item()

            if config.COMPUTE_EVAL_METRICS:
                # works only for 1 process, idx=0 used for infos
                pred_n_gt_spects =\
                    {
                        "mixed_bin_audio_mag": batch["mixed_bin_audio_mag"].cpu().numpy(),
                        "mixed_bin_audio_phase": batch["mixed_bin_audio_phase"].cpu().numpy(),
                        "gt_mono_mag": batch["gt_mono_comps"][..., :1].clone().cpu().numpy(),
                        "gt_mono_phase": batch["gt_mono_comps"][..., 1:2].clone().cpu().numpy(),
                        "pred_mono": pred_mono.detach().cpu().numpy(),
                        "pred_monoFromMem": pred_monoFromMem.detach().cpu().numpy(),
                    }

                if len(config.EVAL_METRICS_TO_COMPUTE) != 0:
                    metrics =\
                        compute_waveform_quality(
                            pred_n_gt_spects,
                            config.EVAL_METRICS_TO_COMPUTE,
                        )

                    mono_metric_name2vals = metrics["mono"]
                    monoFromMem_metric_name2vals = metrics["monoFromMem"]
                    for metric_name in mono_metric_name2vals:
                        if (current_episode_count + 1) not in eval_metrics_toDump["mono"][metric_name]:
                            assert (current_episode_count + 1) not in eval_metrics_toDump["monoFromMem"][metric_name]
                            eval_metrics_toDump["mono"][metric_name][current_episode_count + 1] = {}
                            eval_metrics_toDump["monoFromMem"][metric_name][current_episode_count + 1] = {}
                        eval_metrics_toDump["mono"][metric_name][current_episode_count + 1][current_step_count + 1] =\
                            mono_metric_name2vals[metric_name]
                        eval_metrics_toDump["monoFromMem"][metric_name][current_episode_count + 1][current_step_count + 1] =\
                            monoFromMem_metric_name2vals[metric_name]

                if (current_episode_count + 1) not in eval_metrics_toDump["mono"]["STFT_L2_loss"]:
                    assert (current_episode_count + 1) not in eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"]
                    eval_metrics_toDump["mono"]["STFT_L2_loss"][current_episode_count + 1] = {}
                    eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"][current_episode_count + 1] = {}
                eval_metrics_toDump["mono"]["STFT_L2_loss"][current_episode_count + 1][current_step_count + 1] =\
                    mono_losses[0][0].item()
                eval_metrics_toDump["monoFromMem"]["STFT_L2_loss"][current_episode_count + 1][current_step_count + 1] =\
                    monoFromMem_losses[0][0].item()

                if "episodeCount_to_sceneIdEpisodeId" not in eval_metrics_toDump:
                    eval_metrics_toDump["episodeCount_to_sceneIdEpisodeId"] =\
                        {current_episode_count + 1: (current_scene, current_episode_id)}
                else:
                    eval_metrics_toDump["episodeCount_to_sceneIdEpisodeId"][current_episode_count + 1] =\
                        (current_scene, current_episode_id)

            # batch being re-assigned here because current batch used in the computation of eval metrics
            batch = batch_obs(observations, self.device)
            step_count_all_processes += 1
            next_episodes = self.envs.current_episodes()
            next_scene = next_episodes[0].scene_id.split('/')[-2]
            next_episode_id = next_episodes[0].episode_id

            # particularly useful for multi-process eval
            for env_idx in active_envs:
                if env_idx > 0:
                    raise NotImplementedError

                # episode has ended
                if dones[env_idx]:
                    # stats of simulator-returned performance metrics
                    episode_stats = dict()

                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[env_idx][metric_uuid]
                    for goal_idx in range(len(current_episodes[env_idx].goals)):
                        episode_stats['geodesic_distance_' + str(goal_idx)] =\
                            current_episodes[env_idx].info[goal_idx]['geodesic_distance']
                        episode_stats['euclidean_distance_' + str(goal_idx)] =\
                            norm(np.array(current_episodes[env_idx].goals[goal_idx].position) -
                                 np.array(current_episodes[env_idx].start_position))

                    # use scene + episode_id as unique id for storing stats
                    assert (current_scene, current_episode_id) not in stats_episodes
                    stats_episodes[(current_scene, current_episode_id)] = episode_stats

                    # eval metrics (STFT losses) for logging to log
                    mono_losses_last_step.append(mono_losses[env_idx][0].item())
                    mono_losses_all_steps.append(mono_loss_this_episode / step_count_all_processes[env_idx].item())
                    mono_loss_this_episode = 0.
                    monoFromMem_losses_last_step.append(monoFromMem_losses[env_idx][0].item())
                    monoFromMem_losses_all_steps.append(monoFromMem_loss_this_episode / step_count_all_processes[env_idx].item())
                    monoFromMem_loss_this_episode = 0.

                    # update tqdm object
                    t.update()

                    # particularly useful for multi-process eval
                    if (next_scene, next_episode_id) not in stats_episodes:
                        episode_count_all_processes[env_idx] = num_episode_numbers_taken + 1
                        num_episode_numbers_taken += 1
                        step_count_all_processes[env_idx] = 0

        # closing the open environments after iterating over all episodes
        self.envs.close()

        # mean and std of simulator-returned metrics and STFT L2 losses
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = dict()
            aggregated_stats[stat_key]["mean"] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )
            aggregated_stats[stat_key]["std"] = np.std(
                [v[stat_key] for v in stats_episodes.values()]
            )
        aggregated_stats["mono_loss_last_step"] = dict()
        aggregated_stats["mono_loss_last_step"]["mean"] = np.mean(mono_losses_last_step)
        aggregated_stats["mono_loss_last_step"]["std"] = np.std(mono_losses_last_step)
        aggregated_stats["mono_loss_all_steps"] = dict()
        aggregated_stats["mono_loss_all_steps"]["mean"] = np.mean(mono_losses_all_steps)
        aggregated_stats["mono_loss_all_steps"]["std"] = np.std(mono_losses_all_steps)
        aggregated_stats["monoFromMem_loss_last_step"] = dict()
        aggregated_stats["monoFromMem_loss_last_step"]["mean"] = np.mean(monoFromMem_losses_last_step)
        aggregated_stats["monoFromMem_loss_last_step"]["std"] = np.std(monoFromMem_losses_last_step)
        aggregated_stats["monoFromMem_loss_all_steps"] = dict()
        aggregated_stats["monoFromMem_loss_all_steps"]["mean"] = np.mean(monoFromMem_losses_all_steps)
        aggregated_stats["monoFromMem_loss_all_steps"]["std"] = np.std(monoFromMem_losses_all_steps)

        # dump stats file to disk
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT,
                                                            config.SEED)
                                  )
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        # dump eval metrics to disk
        if config.COMPUTE_EVAL_METRICS:
            with open(os.path.join(config.MODEL_DIR, "eval_metrics.pkl"), "wb") as fo:
                pickle.dump(eval_metrics_toDump, fo, protocol=pickle.HIGHEST_PROTOCOL)

        # dump eval metrics to train.log
        result = {}
        # writing metrics to train.log and/or terminal and tb
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid]["mean"]
            result['episode_{}_mean'.format(metric_uuid)] = aggregated_stats[metric_uuid]["mean"]
        for metric_uuid in episode_metrics_mean.keys():
            if metric_uuid not in ["mono_loss_last_step", "mono_loss_all_steps", "monoFromMem_loss_last_step", "monoFromMem_loss_all_steps"]:
                logger.info(
                    f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
                )
            writer.add_scalar(
                f"{metric_uuid}/{config.EVAL.SPLIT}/mean",
                episode_metrics_mean[metric_uuid],
                checkpoint_index,
            )

        logger.info("Mono STFT L2 loss at last step --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["mono_loss_last_step"]["mean"],
                            std=aggregated_stats["mono_loss_last_step"]["std"]))
        logger.info("Mono STFT L2 loss over all steps --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["mono_loss_all_steps"]["mean"],
                            std=aggregated_stats["mono_loss_all_steps"]["std"]))
        logger.info("MonoFromMem STFT L2 loss at last step --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["monoFromMem_loss_last_step"]["mean"],
                            std=aggregated_stats["monoFromMem_loss_last_step"]["std"]))
        logger.info("MonoFromMem STFT L2 loss over all steps --- mean: {mean:.6f}, std: {std:.6f}"\
                    .format(mean=aggregated_stats["monoFromMem_loss_all_steps"]["mean"],
                            std=aggregated_stats["monoFromMem_loss_all_steps"]["std"]))

        return result
