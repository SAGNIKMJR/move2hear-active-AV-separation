import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from audio_separation.rl.ppo.ddppo_utils import distributed_mean_and_var

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        bin_separation_loss_coef,
        mono_conversion_loss_coef,
        entropy_coef,
        lr_pol=None,
        lr_sep=None,
        eps=None,
        max_grad_norm=None,
        freeze_passive_separators=False,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):
        super().__init__()
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.bin_separation_loss_coef = bin_separation_loss_coef
        self.mono_conversion_loss_coef = mono_conversion_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_normalized_advantage = use_normalized_advantage

        self.freeze_passive_separators=freeze_passive_separators

        pol_params = list(actor_critic.pol_net.parameters()) + list(actor_critic.action_dist.parameters()) +\
                     list(actor_critic.critic.parameters())
        self.optimizer_pol = optim.Adam(pol_params, lr=lr_pol, eps=eps)

        sep_params = list(actor_critic.binSep_enc.parameters()) + list(actor_critic.binSep_dec.parameters()) +\
                     list(actor_critic.bin2mono_enc.parameters()) + list(actor_critic.bin2mono_dec.parameters()) +\
                     list(actor_critic.acoustic_mem.parameters())
        self.optimizer_sep = optim.Adam(sep_params, lr=lr_sep, eps=eps)

        self.device = next(actor_critic.parameters()).device

    def load_pretrained_passive_separators(self, state_dict):
        # loading pretrained weights from passive binaural separator
        for name in self.actor_critic.binSep_enc.state_dict():
            self.actor_critic.binSep_enc.state_dict()[name].copy_(state_dict["actor_critic.binSep_enc." + name])
        for name in self.actor_critic.binSep_dec.state_dict():
            self.actor_critic.binSep_dec.state_dict()[name].copy_(state_dict["actor_critic.binSep_dec." + name])

        # loading pretrained weights from passive bin2mono separator
        for name in self.actor_critic.bin2mono_enc.state_dict():
            self.actor_critic.bin2mono_enc.state_dict()[name].copy_(state_dict["actor_critic.bin2mono_enc." + name])
        for name in self.actor_critic.bin2mono_dec.state_dict():
            self.actor_critic.bin2mono_dec.state_dict()[name].copy_(state_dict["actor_critic.bin2mono_dec." + name])

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts_pol):
        advantages = rollouts_pol.returns[:-1] - rollouts_pol.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update_pol(self, rollouts_pol):
        advantages = self.get_advantages(rollouts_pol)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts_pol.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
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
                ) = sample


                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_pol_batch,
                    masks_batch,
                    actions_batch,
                    pred_binSepMasks=pred_binSepMasks_batch,
                    pred_mono=pred_mono_batch,
                    pred_monoFromMem=pred_monoFromMem_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_pol.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step_pol()
                self.optimizer_pol.step()
                self.after_step()

                action_loss_epoch += action_loss.item()
                value_loss_epoch += value_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_sep(self, rollouts_sep):
        bin_loss_epoch = 0.
        mono_loss_epoch = 0.
        monoFromMem_loss_epoch = 0.

        for e in range(self.ppo_epoch):
            data_generator = rollouts_sep.recurrent_generator(self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    pred_monoFromMem_batch,
                    prev_pred_monoFromMem_batch,
                    masks_batch
                ) = sample

                # use torch.no_grad since passive separators are frozen
                with torch.no_grad():
                    pred_binSepMasks =\
                        self.actor_critic.get_binSepMasks(
                            obs_batch,
                        )
                    pred_mono =\
                        self.actor_critic.convert_bin2mono(pred_binSepMasks.detach(),
                                                           mixed_audio=obs_batch["mixed_bin_audio_mag"],
                                                           )

                prev_pred_monoFromMem_masked = prev_pred_monoFromMem_batch *\
                                               masks_batch.unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                            *pred_mono.size()[1:]
                                                                                            )
                pred_monoFromMem =\
                    self.actor_critic.get_monoFromMem(pred_mono, prev_pred_monoFromMem_masked)

                # get monoFromMem loss
                gt_mono_mag = obs_batch["gt_mono_comps"][..., 0::2].clone()[..., :1]
                monoFromMem_loss = F.l1_loss(pred_monoFromMem, gt_mono_mag)

                # get bin2mono loss
                mono_loss = F.l1_loss(pred_mono, gt_mono_mag)

                # get bin-sep loss
                gt_bin_mag = obs_batch["gt_bin_comps"][..., 0::2].clone()[..., :2]
                pred_bin = (torch.exp(obs_batch["mixed_bin_audio_mag"]) - 1) * pred_binSepMasks
                bin_loss = F.l1_loss(pred_bin, gt_bin_mag)

                self.optimizer_sep.zero_grad()
                total_loss = monoFromMem_loss

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step_sep()
                self.optimizer_sep.step()
                self.after_step()

                bin_loss_epoch += bin_loss.item()
                mono_loss_epoch += mono_loss.item()
                monoFromMem_loss_epoch += monoFromMem_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        bin_loss_epoch /= num_updates
        mono_loss_epoch /= num_updates
        monoFromMem_loss_epoch /= num_updates

        return bin_loss_epoch, mono_loss_epoch, monoFromMem_loss_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step_pol(self):
        pol_params = list(self.actor_critic.pol_net.parameters()) +\
                     list(self.actor_critic.action_dist.parameters()) +\
                     list(self.actor_critic.critic.parameters())
        nn.utils.clip_grad_norm_(
            pol_params, self.max_grad_norm
        )

    def before_step_sep(self):
        sep_params = list(self.actor_critic.binSep_enc.parameters()) + list(self.actor_critic.binSep_dec.parameters()) +\
                     list(self.actor_critic.bin2mono_enc.parameters()) + list(self.actor_critic.bin2mono_dec.parameters()) +\
                     list(self.actor_critic.acoustic_mem.parameters())
        nn.utils.clip_grad_norm_(
            sep_params, self.max_grad_norm
        )

    def after_step(self):
        pass


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(
        self, rollouts_nav
    ) -> torch.Tensor:
        advantages = rollouts_nav.returns[:-1] - rollouts_nav.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)

        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        super().before_backward(loss)

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
