import os
import logging
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from habitat import logger

from audio_separation.common.base_trainer import BaseRLTrainer
from audio_separation.common.baseline_registry import baseline_registry
from audio_separation.common.env_utils import construct_envs
from audio_separation.common.environments import get_env_class
from audio_separation.common.tensorboard_utils import TensorboardWriter
from audio_separation.pretrain.passive.policy import Move2HearPassiveWoMemoryPolicy
from audio_separation.pretrain.passive.passive import Passive
from audio_separation.pretrain.datasets.dataset import PassiveDataset
from habitat_audio.utils import load_points_data


SCENE_SPLITS = {
    "mp3d":
        {
            'train':
                ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
                 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
                 'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va',  '5LpN3gDmAk7',
                 'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
                 'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
                 'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
                 '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                 'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
                 'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
                 '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
            'val':
                ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
                 'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
        },
}


EPS = 1e-7


@baseline_registry.register_trainer(name="passive")
class PassiveTrainer(BaseRLTrainer):
    r"""Trainer class for pretraining passive separators in a supervised fashion
    """
    # supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

    def _setup_passive_agent(self,) -> None:
        r"""Sets up agent for passive pretraining.

        Args:
            None
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        passive_cfg = self.config.Pretrain.Passive

        self.actor_critic = Move2HearPassiveWoMemoryPolicy(
            observation_space=self.envs.observation_spaces[0],
        )

        self.actor_critic.to(self.device)
        self.actor_critic.train()

        self.agent = Passive(
            actor_critic=self.actor_critic,
        )

    def save_checkpoint(self, file_name: str,) -> None:
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

    def get_dataloaders(self):
        r"""
        build datasets and dataloaders
        :return:
            dataloaders: PyTorch dataloaders for training and validation
            dataset_sizes: sizes of train and val datasets
        """
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        audio_cfg = sim_cfg.AUDIO

        scene_splits = {"train": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["train"],
                        "val": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["val"],
                        "nonoverlapping_val": SCENE_SPLITS[sim_cfg.SCENE_DATASET]["val"]}
        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in scene_splits:
            scenes = scene_splits[split]
            scene_graphs = dict()
            for scene in scenes:
                _, graph = load_points_data(
                    os.path.join(audio_cfg.META_DIR, scene),
                    audio_cfg.GRAPH_FILE,
                    transform=True,
                    scene_dataset=sim_cfg.SCENE_DATASET)
                scene_graphs[scene] = graph

            datasets[split] = PassiveDataset(
                split=split,
                scene_graphs=scene_graphs,
                sim_cfg=sim_cfg,
            )

            dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=audio_cfg.BATCH_SIZE,
                                            shuffle=(split == 'train'),
                                            pin_memory=True,
                                            num_workers=audio_cfg.NUM_WORKER,
                                            )
            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))
        return dataloaders, dataset_sizes

    def train(self) -> None:
        r"""Main method for training passive separators using supervised learning.

        Returns:
            None
        """
        passive_cfg = self.config.Pretrain.Passive
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        audio_cfg = sim_cfg.AUDIO

        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # just needed to get observation_spaces
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_passive_agent()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
                                     lr=passive_cfg.lr, eps=passive_cfg.eps)

        # build datasets and dataloaders
        dataloaders, dataset_sizes = self.get_dataloaders()

        best_mono_loss = float('inf')
        best_nonoverlapping_mono_loss = float('inf')
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for epoch in range(passive_cfg.NUM_EPOCHS):
                logging.info('-' * 10)
                logging.info('Epoch {}/{}'.format(epoch, passive_cfg.NUM_EPOCHS - 1))

                for split in dataloaders.keys():
                    # set forward pass mode
                    if split == "train":
                        self.actor_critic.train()
                    else:
                        self.actor_critic.eval()

                    bin_loss_epoch = 0.
                    mono_loss_epoch = 0.
                    for i, data in enumerate(tqdm(dataloaders[split])):
                        mixed_audio = data[0].to(self.device)
                        gt_bin_mag = data[1].to(self.device)[..., 0:2]
                        gt_mono_mag = data[2].to(self.device)[..., :1]
                        target_class = data[3].to(self.device)
                        bs = target_class.size(0)

                        obs_batch = {"mixed_bin_audio_mag": mixed_audio, "target_class": target_class}

                        if split == "train":
                            pred_binSepMasks = self.actor_critic.get_binSepMasks(obs_batch)
                            pred_mono =\
                                self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(), mixed_audio=mixed_audio)
                        else:
                            with torch.no_grad():
                                pred_binSepMasks = self.actor_critic.get_binSepMasks(obs_batch)
                                pred_mono =\
                                    self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                                       mixed_audio=mixed_audio)

                        bin_loss, mono_loss, optimizer =\
                            self.optimize_supervised_loss(optimizer=optimizer,
                                                          mixed_audio=mixed_audio,
                                                          pred_binSepMasks=pred_binSepMasks,
                                                          gt_bin_mag=gt_bin_mag,
                                                          pred_mono=pred_mono,
                                                          gt_mono_mag=gt_mono_mag,
                                                          split=split,
                                                          )

                        bin_loss_epoch += bin_loss.item() * bs
                        mono_loss_epoch += mono_loss.item() * bs

                    bin_loss_epoch /= dataset_sizes[split]
                    mono_loss_epoch /= dataset_sizes[split]

                    writer.add_scalar('bin_loss/{}'.format(split), bin_loss_epoch, epoch)
                    writer.add_scalar('mono_loss/{}'.format(split), mono_loss_epoch, epoch)
                    logging.info('{} -- bin loss: {:.4f}, mono loss: {:.4f}'.format(split.upper(),
                                                                                    bin_loss_epoch,
                                                                                    mono_loss_epoch))
                    if split == "val":
                        if mono_loss_epoch < best_mono_loss:
                            best_mono_loss = mono_loss_epoch
                            self.save_checkpoint(f"best_ckpt_val.pth")
                    elif split == "nonoverlapping_val":
                        if mono_loss_epoch < best_nonoverlapping_mono_loss:
                            best_nonoverlapping_mono_loss = mono_loss_epoch
                            self.save_checkpoint(f"best_ckpt_nonoverlapping_val.pth")
        self.envs.close()

    def optimize_supervised_loss(self, optimizer, mixed_audio, pred_binSepMasks, gt_bin_mag, pred_mono, gt_mono_mag,
                                 split='train',):
        mixed_audio = torch.exp(mixed_audio) - 1
        pred_bin = pred_binSepMasks * mixed_audio
        bin_loss = F.l1_loss(pred_bin, gt_bin_mag)

        mono_loss = F.l1_loss(pred_mono, gt_mono_mag)

        if split == "train":
            optimizer.zero_grad()
            loss = bin_loss + mono_loss
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.Pretrain.Passive.max_grad_norm
            )
            loss.backward()
            optimizer.step()

        return bin_loss, mono_loss, optimizer
