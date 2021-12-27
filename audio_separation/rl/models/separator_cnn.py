import torch.nn as nn
import torch


def unet_conv(input_nc, output_nc, kernel_size=(4, 4), norm_layer=nn.BatchNorm2d, padding=(1, 1), stride=(2, 2), bias=False):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,)
    downrelu = nn.LeakyReLU(0.2, True)
    if norm_layer is not None:
        downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv, downrelu])


def unet_upconv(input_nc, output_nc, kernel_size=(4, 4), outermost=False, norm_layer=nn.BatchNorm2d, stride=(2, 2),
                padding=(1, 1), output_padding=(0, 0), bias=False,):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                                output_padding=output_padding, bias=bias)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


class PassiveSepEncCNN(nn.Module):
    r"""A U-net encoder for passive separation.

    Takes in mixed binaural audio or predicted clean binaural and produces an clean binaural or clean monaural embeddings
    and skip-connection feature list, respectively.

    Args:
        convert_bin2mono: creates encoder for converting binaural to monaural if set to True
    """
    def __init__(self, convert_bin2mono=False):
        super().__init__()
        self._convert_bin2mono = convert_bin2mono
        # originally 2 channels for binaural or concatenation of monos but spec. sliced up into 16 chunks along the frequency
        # dimension (this makes the high-res. specs. easier to deal with)
        self._slice_factor = 16
        self._n_input_audio = 2 * self._slice_factor
        if not convert_bin2mono:
            self._n_input_audio += 1

        self.cnn = nn.Sequential(
            unet_conv(self._n_input_audio, 64,),
            unet_conv(64, 64 * 2,),
            unet_conv(64 * 2, 64 * 4,),
            unet_conv(64 * 4, 64 * 8,),
            unet_conv(64 * 8, 64 * 8,)
        )

        self.layer_init()

    def layer_init(self):
        for module in self.cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("leaky_relu", 0.2)
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, observations, mixed_audio=None,):
        cnn_input = []

        if self._convert_bin2mono:
            assert mixed_audio is not None
            # observations has pred_binSepMasks
            x = observations
            mixed_audio = torch.exp(mixed_audio) - 1
            x = x * mixed_audio
            x = torch.log1p(torch.clamp(x, min=0))
        else:
            # observations has all sensor readings
            x = observations["mixed_bin_audio_mag"]

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = x.permute(0, 3, 1, 2)

        # slice along freq dimension into 16 chunks
        x = x.view(x.size(0), x.size(1), self._slice_factor, -1, x.size(3))
        x = x.reshape(x.size(0), -1, x.size(3),  x.size(4))

        # append target class for passive bin. extraction
        if not self._convert_bin2mono:
            target_class = observations["target_class"]
            # adding 1 to the target_class sensor value (probably not necessary)
            target_class = target_class.unsqueeze(1).unsqueeze(1).repeat(1, 1, x.size(2), x.size(3)).float() + 1
            x = torch.cat((x, target_class), dim=1)

        cnn_input.append(x)
        cnn_input = torch.cat(cnn_input, dim=1)

        lst_skip_feats = []
        out = cnn_input
        for module in self.cnn:
            out = module(out)
            lst_skip_feats.append(out)
        # return the first N - 1 features (last feature is the bottleneck feature) and invert for convenience during
        # upsampling forward pass
        return out.reshape(cnn_input.size(0), -1), lst_skip_feats[:-1][::-1]


class PassiveSepDecCNN(nn.Module):
    r"""A U-net decoder for passive separation.

    Takes in feature embeddings and skip-connection feature list and produces an clean binaural or clean monaural, respectively.

    Args:
        convert_bin2mono: creates encoder for converting binaural to monaural if set to True
    """
    def __init__(self, convert_bin2mono=False,):
        super().__init__()
        # originally 2 channels for binaural or concatenation of monos but spec. sliced up into 16 chunks along the frequency
        # dimension (this makes the high-res. specs. easier to deal with)
        self._slice_factor = 16
        self._n_out_audio = self._slice_factor
        if not convert_bin2mono:
            self._n_out_audio *= 2

        self.cnn = nn.Sequential(
            unet_upconv(64 * 8, 64 * 8),
            unet_upconv(64 * 16, 64 * 4,),
            unet_upconv(64 * 8, 64 * 2,),
            unet_upconv(64 * 4, 64 * 1),
            unet_upconv(64 * 2, self._n_out_audio, padding=(1, 1)),
            nn.Sequential(nn.Conv2d(self._n_out_audio, self._n_out_audio, kernel_size=(1, 1),)),
        )

        self.layer_init()

    def layer_init(self):
        for module in self.cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, bottleneck_feats, lst_skip_feats):
        out = bottleneck_feats.view(bottleneck_feats.size(0), -1, 1, 1)

        for idx, module in enumerate(self.cnn):
            if (idx == 0) or (idx == len(self.cnn) - 1):
                out = module(out)
            else:
                skip_feats = lst_skip_feats[idx - 1]
                out = module(torch.cat((out, skip_feats), dim=1))

        # deslice
        out = out.view(out.size(0), -1, self._slice_factor, out.size(2), out.size(3))
        out = out.reshape(out.size(0), out.size(1), -1, out.size(4))

        # permute tensor to dimension [BATCH x HEIGHT X WIDTH x CHANNEL]
        out = out.permute(0, 2, 3, 1)

        return out
