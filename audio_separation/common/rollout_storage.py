from collections import defaultdict

import torch


class RolloutStoragePol:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        r"""
        Class for storing rollout information for RL policy trainer.
        :param num_steps: number of steps before PPO update
        :param num_envs: number of training environments
        :param observation_space: simulator observation space
        :param recurrent_hidden_state_size: hidden state size for policy GRU
        :param num_recurrent_layers: number of hidden layers in policy GRU
        """
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states_pol = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        assert "gt_mono_comps" in observation_space.spaces

        self.pred_binSepMasks =\
            torch.zeros(
                num_steps,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                2,
            )

        self.pred_mono =\
            torch.zeros(
                num_steps,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                1,
            )

        self.prev_pred_monoFromMem =\
            torch.zeros(
                num_steps + 1,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                1,
            )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)

        action_shape = 1

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)
        self.recurrent_hidden_states_pol = self.recurrent_hidden_states_pol.to(device)

        self.pred_binSepMasks = self.pred_binSepMasks.to(device)
        self.pred_mono = self.pred_mono.to(device)
        self.prev_pred_monoFromMem = self.prev_pred_monoFromMem.to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)

        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states_pol,
        actions,
        action_log_probs,
        values,
        rewards,
        masks,
        pred_binSepMasks=None,
        pred_mono=None,
        pred_monoFromMem=None,
    ):
        r"""
        Method for inserting useful scalars and tensors from the current step into the storage
        :param observations: current observations from the simulator
        :param recurrent_hidden_states_pol: current policy GRU hidden states
        :param actions: current actions
        :param action_log_probs: current action log probabilities
        :param values: current values
        :param rewards: current rewards
        :param masks: current not-done masks
        :param pred_binSepMasks: current binaural separation masks
        :param pred_mono: current monaural predictions
        :param pred_monoFromMem: current monaural predictions from acoustic memory refiner
        """
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states_pol[self.step + 1].copy_(
            recurrent_hidden_states_pol
        )

        self.pred_binSepMasks[self.step].copy_(pred_binSepMasks)
        self.pred_mono[self.step].copy_(pred_mono)
        self.prev_pred_monoFromMem[self.step + 1].copy_(pred_monoFromMem)

        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step].copy_(values)

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)

        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])
        self.recurrent_hidden_states_pol[0].copy_(self.recurrent_hidden_states_pol[-1])

        self.prev_pred_monoFromMem[0].copy_(self.prev_pred_monoFromMem[-1])

        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        r"""
        compute returns with or without GAE
        """
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        r"""
        Recurrent batch generator for PPO update
        :param advantages: advantage values needed for PPO update
        :param num_mini_batch: number of mini batches to split all processes across all environments into
        :return: current batch for doing forward and backward passes for PPO update
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            recurrent_hidden_states_pol_batch = []

            pred_binSepMasks_batch = []
            pred_mono_batch = []
            pred_monoFromMem_batch = []

            value_preds_batch = []
            return_batch = []
            adv_targ = []

            actions_batch = []
            old_action_log_probs_batch = []

            masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )
                recurrent_hidden_states_pol_batch.append(
                    self.recurrent_hidden_states_pol[0, :, ind]
                )

                pred_binSepMasks_batch.append(self.pred_binSepMasks[:, ind])
                pred_mono_batch.append(self.pred_mono[:, ind])
                pred_monoFromMem_batch.append(self.prev_pred_monoFromMem[1:, ind])

                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                adv_targ.append(advantages[:, ind])

                actions_batch.append(self.actions[:, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                masks_batch.append(self.masks[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_pol_batch = torch.stack(
                recurrent_hidden_states_pol_batch, 1
            )

            pred_binSepMasks_batch = torch.stack(pred_binSepMasks_batch, 1)
            pred_mono_batch = torch.stack(pred_mono_batch, 1)
            pred_monoFromMem_batch = torch.stack(pred_monoFromMem_batch, 1)

            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            actions_batch = torch.stack(actions_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )

            masks_batch = torch.stack(masks_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            pred_binSepMasks_batch = self._flatten_helper(T, N, pred_binSepMasks_batch)
            pred_mono_batch = self._flatten_helper(T, N, pred_mono_batch)
            pred_monoFromMem_batch = self._flatten_helper(T, N, pred_monoFromMem_batch)

            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            adv_targ = self._flatten_helper(T, N, adv_targ)

            actions_batch = self._flatten_helper(T, N, actions_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )

            masks_batch = self._flatten_helper(T, N, masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_pol_batch,
                pred_binSepMasks_batch,
                pred_mono_batch,
                pred_monoFromMem_batch,
                value_preds_batch,
                return_batch,
                adv_targ,
                actions_batch,
                old_action_log_probs_batch,
                masks_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorageSep:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
    ):
        r"""
        Class for storing rollout information for audio separator trainer.
        :param num_steps: number of steps before audio separator update
        :param num_envs: number of training environments
        :param observation_space: simulator observation space
        """
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        assert "gt_mono_comps" in observation_space.spaces

        self.prev_pred_monoFromMem =\
            torch.zeros(
                num_steps + 1,
                num_envs,
                observation_space.spaces["gt_mono_comps"].shape[0],
                observation_space.spaces["gt_mono_comps"].shape[1],
                1
                )

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.prev_pred_monoFromMem = self.prev_pred_monoFromMem.to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        masks,
        pred_monoFromMem=None,
    ):
        r"""
        Method for inserting useful scalars and tensors from the current step into the storage
        :param observations: current observations from the simulator
        :param masks: current not-done masks
        :param pred_monoFromMem: current monaural predictions from acoustic memory refiner
        """
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )

        self.prev_pred_monoFromMem[self.step + 1].copy_(pred_monoFromMem)

        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.prev_pred_monoFromMem[0].copy_(self.prev_pred_monoFromMem[-1])

        self.masks[0].copy_(self.masks[-1])

    def recurrent_generator(self, num_mini_batch):
        r"""
        Recurrent batch generator for audio separator training
        :param num_mini_batch: number of mini batches to split all processes across all environments into
        :return: current batch for doing forward and backward passes for audio separator training
        """
        num_processes = self.masks.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            pred_monoFromMem_batch = []
            prev_pred_monoFromMem_batch = []

            masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                pred_monoFromMem_batch.append(self.prev_pred_monoFromMem[1:, ind])
                prev_pred_monoFromMem_batch.append(self.prev_pred_monoFromMem[:-1, ind])

                masks_batch.append(self.masks[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            pred_monoFromMem_batch = torch.stack(pred_monoFromMem_batch, 1)
            prev_pred_monoFromMem_batch = torch.stack(prev_pred_monoFromMem_batch, 1)

            masks_batch = torch.stack(masks_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            pred_monoFromMem_batch = self._flatten_helper(T, N, pred_monoFromMem_batch)
            prev_pred_monoFromMem_batch = self._flatten_helper(T, N, prev_pred_monoFromMem_batch)

            masks_batch = self._flatten_helper(T, N, masks_batch)

            yield (
                observations_batch,
                pred_monoFromMem_batch,
                prev_pred_monoFromMem_batch,
                masks_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
