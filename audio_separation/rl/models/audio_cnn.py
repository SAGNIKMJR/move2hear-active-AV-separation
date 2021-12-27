import numpy as np
import torch
import torch.nn as nn

from audio_separation.common.utils import Flatten


class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer for high res spec.

    Takes in separated audio outputs (bin/monos) and produces an embedding

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        encode_monoNmonoFromMem: creates CNN for encoding predicted monaurals (concatenation of passive and acoustic
                                 memory outputs) if set to True
    """
    def __init__(self, observation_space, output_size, encode_monoNmonoFromMem=False,):
        super().__init__()
        self.encode_monoNmonoFromMem = encode_monoNmonoFromMem
        # originally 2 channels for binaural or concatenation of monos but spec. sliced up into 16 chunks along the frequency
        # dimension (this makes the high-res. specs. easier to deal with)
        self._slice_factor = 16
        self._n_input_audio = 2 * self._slice_factor

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (2, 2)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        cnn_dims = np.array(
            [observation_space.spaces["mixed_bin_audio_mag"].shape[0] // 16,
             observation_space.spaces["mixed_bin_audio_mag"].shape[1]],
            dtype=np.float32
        )

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations, pred_binSepMasks=None, pred_monoNmonoFromMem=None,):
        cnn_input = []

        if self.encode_monoNmonoFromMem:
            assert pred_monoNmonoFromMem is not None
            x = torch.log1p(torch.clamp(pred_monoNmonoFromMem, min=0))
        else:
            assert pred_binSepMasks is not None
            x = observations["mixed_bin_audio_mag"]
            x = torch.exp(x) - 1
            x = x * pred_binSepMasks
            x = torch.log1p(torch.clamp(x, min=0))

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = x.permute(0, 3, 1, 2)

        # slice along freq dimension into 16 chunks
        x = x.view(x.size(0), x.size(1), self._slice_factor, -1, x.size(3))
        x = x.reshape(x.size(0), -1, x.size(3),  x.size(4))

        cnn_input.append(x)
        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)
