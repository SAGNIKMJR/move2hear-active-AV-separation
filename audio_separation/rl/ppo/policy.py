import abc

import torch
import torch.nn as nn
from torchsummary import summary

from audio_separation.common.utils import CategoricalNet
from audio_separation.rl.models.rnn_state_encoder import RNNStateEncoder
from audio_separation.rl.models.visual_cnn import VisualCNN
from audio_separation.rl.models.audio_cnn import AudioCNN
from audio_separation.rl.models.separator_cnn import PassiveSepEncCNN, PassiveSepDecCNN
from audio_separation.rl.models.memory_nets import AcousticMem


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PolicyNet(Net):
    r"""Network which passes the observations and separated audio outputs through CNNs and concatenates
    them into a single vector before passing that through RNN.
    """
    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, extra_depth=False,
                 world_rank=0,):
        super().__init__()
        assert 'mixed_bin_audio_mag' in observation_space.spaces
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb, extra_depth)
        self.bin_encoder = AudioCNN(observation_space, hidden_size,)
        self.monoNmonoFromMem_encoder = AudioCNN(observation_space, hidden_size, encode_monoNmonoFromMem=True,)

        rnn_input_size = 3 * self._hidden_size
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        # printing out the network layers
        if world_rank == 0:
            if (('rgb' in observation_space.spaces) and (not extra_rgb)) and\
                    (('depth' in observation_space.spaces) and (not extra_depth)):
                assert observation_space.spaces['rgb'].shape[:-1] == observation_space.spaces['depth'].shape[:-1]
                # hardcoding needed since rgb is of size (720, 720, 3) when rendering videos and this call throws an error
                rgb_shape = (128, 128, 3)   # observation_space.spaces['rgb'].shape
                depth_shape = observation_space.spaces['depth'].shape
                summary(self.visual_encoder.cnn, (rgb_shape[2] + depth_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
            elif ('rgb' in observation_space.spaces) and (not extra_rgb):
                """hardcoding needed since rgb is of size (720, 720, 3) when rendering videos and this call throws an error"""
                rgb_shape = (128, 128, 3)   # observation_space.spaces['rgb'].shape
                summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
            elif ('depth' in observation_space.spaces) and (not extra_depth):
                depth_shape = observation_space.spaces['depth'].shape
                summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')

            audio_shape = observation_space.spaces['mixed_bin_audio_mag'].shape
            summary(self.bin_encoder.cnn, (2 * 16, audio_shape[0] // 16, audio_shape[1]), device='cpu')
            summary(self.monoNmonoFromMem_encoder.cnn, (2 * 16, audio_shape[0] // 16, audio_shape[1]), device='cpu')

    @property
    def is_blind(self):
        return False

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, masks, pred_binSepMasks=None, pred_mono=None,
                pred_monoFromMem=None,):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.bin_encoder(observations, pred_binSepMasks=pred_binSepMasks))
        x.append(self.monoNmonoFromMem_encoder(observations, pred_monoNmonoFromMem=torch.cat((pred_mono, pred_monoFromMem), dim=3)))

        try:
            x1 = torch.cat(x, dim=1)
        except AssertionError as error:
            for data in x:
                print(data.size())

        try:
            x2, rnn_hidden_states_new = self.state_encoder(x1, rnn_hidden_states, masks)
        except AssertionError as error:
            print(x1.size(), rnn_hidden_states.size(), masks.size(), x2.size(), rnn_hidden_states_new.size())

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states_new


class PassiveSepEnc(nn.Module):
    r"""Network which encodes separated bin or mono outputs
    """
    def __init__(self, observation_space, world_rank=0, convert_bin2mono=False,):
        super().__init__()
        assert 'mixed_bin_audio_mag' in observation_space.spaces

        self.passive_sep_encoder = PassiveSepEncCNN(convert_bin2mono=convert_bin2mono,)

        if world_rank == 0:
            audio_shape = observation_space.spaces['mixed_bin_audio_mag'].shape

            if not convert_bin2mono:
                summary(self.passive_sep_encoder.cnn,
                        (audio_shape[2] * 16 + 1, audio_shape[0] // 16, audio_shape[1]),
                        device='cpu')
            else:
                summary(self.passive_sep_encoder.cnn,
                        (audio_shape[2] * 16, audio_shape[0] // 16, audio_shape[1]),
                        device='cpu')

    def forward(self, observations, mixed_audio=None):
        bottleneck_feats, lst_skip_feats = self.passive_sep_encoder(observations, mixed_audio=mixed_audio,)

        return bottleneck_feats, lst_skip_feats


class PassiveSepDec(nn.Module):
    r"""Network which decodes separated bin or mono outputs feature embeddings
    """
    def __init__(self, convert_bin2mono=False,):
        super().__init__()
        self.passive_sep_decoder = PassiveSepDecCNN(convert_bin2mono=convert_bin2mono,)

    def forward(self, bottleneck_feats, lst_skip_feats):
        return self.passive_sep_decoder(bottleneck_feats, lst_skip_feats)


class Policy(nn.Module):
    r"""
    Network for the full Move2Hear policy, including separation and action-making
    """
    def __init__(self, pol_net, dim_actions, binSep_enc, binSep_dec, bin2mono_enc, bin2mono_dec, acoustic_mem,):
        super().__init__()
        self.dim_actions = dim_actions

        # full policy with actor and critic
        self.pol_net = pol_net
        self.action_dist = CategoricalNet(
            self.pol_net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.pol_net.output_size)

        self.binSep_enc = binSep_enc
        self.binSep_dec = binSep_dec
        self.bin2mono_enc = bin2mono_enc
        self.bin2mono_dec = bin2mono_dec
        self.acoustic_mem = acoustic_mem

    def forward(self):
        raise NotImplementedError

    def get_binSepMasks(self, observations):
        bottleneck_feats,  lst_skip_feats = self.binSep_enc(
            observations,
        )
        return self.binSep_dec(bottleneck_feats, lst_skip_feats)

    def convert_bin2mono(self, pred_binSepMasks, mixed_audio=None):
        bottleneck_feats,  lst_skip_feats = self.bin2mono_enc(
            pred_binSepMasks, mixed_audio=mixed_audio
        )
        return self.bin2mono_dec(bottleneck_feats, lst_skip_feats)

    def get_monoFromMem(self, pred_mono, prev_pred_monoFromMem_masked):
        return self.acoustic_mem(pred_mono, prev_pred_monoFromMem_masked)

    def act(
        self,
        observations,
        rnn_hidden_states_pol,
        masks,
        deterministic=False,
        pred_binSepMasks=None,
        pred_mono=None,
        pred_monoFromMem=None,
    ):
        feats_pol, rnn_hidden_states_pol = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks=pred_binSepMasks.detach(),
            pred_mono=pred_mono.detach(),
            pred_monoFromMem=pred_monoFromMem.detach(),
        )

        dist = self.action_dist(feats_pol)
        value = self.critic(feats_pol)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states_pol, dist.get_probs()

    def get_value(
            self,
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks=None,
            pred_mono=None,
            pred_monoFromMem=None,
    ):

        feats_pol, _ = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks=pred_binSepMasks.detach(),
            pred_mono=pred_mono.detach(),
            pred_monoFromMem=pred_monoFromMem.detach(),
        )

        return self.critic(feats_pol)

    def evaluate_actions(
            self,
            observations,
            rnn_hidden_states_pol,
            masks,
            action,
            pred_binSepMasks=None,
            pred_mono=None,
            pred_monoFromMem=None,
    ):
        feats_pol, rnn_hidden_states_pol = self.pol_net(
            observations,
            rnn_hidden_states_pol,
            masks,
            pred_binSepMasks=pred_binSepMasks,
            pred_mono=pred_mono,
            pred_monoFromMem=pred_monoFromMem,
        )

        dist = self.action_dist(feats_pol)
        value = self.critic(feats_pol)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hidden_states_pol


class Move2HearPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        extra_depth=False,
        use_ddppo=False,
        world_rank=0,
        use_smartnav_for_eval_pol_mix=False,
    ):
        pol_net = PolicyNet(
            observation_space=observation_space,
            hidden_size=hidden_size,
            goal_sensor_uuid=goal_sensor_uuid,
            extra_rgb=extra_rgb,
            extra_depth=extra_depth,
            world_rank=world_rank,
        )

        binSep_enc = PassiveSepEnc(
            observation_space=observation_space,
            world_rank=world_rank,
        )
        binSep_dec = PassiveSepDec()

        bin2mono_enc = PassiveSepEnc(
            observation_space=observation_space,
            world_rank=world_rank,
            convert_bin2mono=True,
        )
        bin2mono_dec = PassiveSepDec(
            convert_bin2mono=True,
        )

        acoustic_mem = AcousticMem(
            use_ddppo=use_ddppo,
        )

        super().__init__(
            pol_net,
            # action_space.n - 1 if use_smartnav_for_eval_pol_mix else action_space.n,
            action_space.n,
            binSep_enc,
            binSep_dec,
            bin2mono_enc,
            bin2mono_dec,
            acoustic_mem,
        )

