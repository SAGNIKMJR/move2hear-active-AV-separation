import torch.nn as nn
from torchsummary import summary

from audio_separation.rl.models.separator_cnn import PassiveSepEncCNN, PassiveSepDecCNN


class PassiveSepEnc(nn.Module):
    r"""Network which encodes separated bin or mono outputs
    """
    def __init__(self, observation_space, world_rank=0, convert_bin2mono=False,):
        super().__init__()
        print(observation_space.spaces)
        assert 'mixed_bin_audio_mag' in observation_space.spaces

        print(convert_bin2mono)
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
    Network for the passive separation in Move2Hear pretraining
    """
    def __init__(self, binSep_enc, binSep_dec, bin2mono_enc, bin2mono_dec,):
        super().__init__()
        self.binSep_enc = binSep_enc
        self.binSep_dec = binSep_dec
        self.bin2mono_enc = bin2mono_enc
        self.bin2mono_dec = bin2mono_dec

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


class Move2HearPassiveWoMemoryPolicy(Policy):
    def __init__(
        self,
        observation_space,
    ):
        binSep_enc = PassiveSepEnc(
            observation_space=observation_space,
        )
        binSep_dec = PassiveSepDec()

        bin2mono_enc = PassiveSepEnc(
            observation_space=observation_space,
            convert_bin2mono=True,
        )
        bin2mono_dec = PassiveSepDec(
            convert_bin2mono=True,
        )

        super().__init__(
            binSep_enc,
            binSep_dec,
            bin2mono_enc,
            bin2mono_dec,
        )
