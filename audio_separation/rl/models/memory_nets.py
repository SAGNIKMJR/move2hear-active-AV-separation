import torch.nn as nn
import torch


class AcousticMem(nn.Module):
    def __init__(self, use_ddppo=False,):
        super().__init__()
        self._slice_factor = 16
        _n_out_audio = self._slice_factor

        if use_ddppo:
            self.cnn = nn.Sequential(
                nn.Conv2d(_n_out_audio * 2, 32, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, _n_out_audio, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(_n_out_audio * 2, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, _n_out_audio, kernel_size=3, padding=1, bias=False),
            )

        self.layer_init()

    def layer_init(self):
        for layer in self.cnn:
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

    def forward(self, pred_mono, prev_pred_monoFromMem_masked):
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        pred_mono = pred_mono.permute(0, 3, 1, 2)
        prev_pred_monoFromMem_masked = prev_pred_monoFromMem_masked.permute(0, 3, 1, 2)

        # slice along freq dimension into 16 chunks
        pred_mono = pred_mono.view(pred_mono.size(0), pred_mono.size(1), self._slice_factor, -1, pred_mono.size(3))
        pred_mono = pred_mono.reshape(pred_mono.size(0), -1, pred_mono.size(3), pred_mono.size(4))

        prev_pred_monoFromMem_masked = prev_pred_monoFromMem_masked.view(prev_pred_monoFromMem_masked.size(0),
                                                                         prev_pred_monoFromMem_masked.size(1),
                                                                         self._slice_factor,
                                                                         -1,
                                                                         prev_pred_monoFromMem_masked.size(3))
        prev_pred_monoFromMem_masked = prev_pred_monoFromMem_masked.reshape(prev_pred_monoFromMem_masked.size(0),
                                                                            -1,
                                                                            prev_pred_monoFromMem_masked.size(3),
                                                                            prev_pred_monoFromMem_masked.size(4))

        out = torch.cat((pred_mono, prev_pred_monoFromMem_masked), dim=1)
        out = self.cnn(out)

        # deslice
        out = out.view(out.size(0), -1, self._slice_factor, out.size(2), out.size(3))
        out = out.reshape(out.size(0), out.size(1), -1, out.size(4))

        # permute tensor to dimension [BATCH x HEIGHT X WIDTH x CHANNEL]
        out = out.permute(0, 2, 3, 1)

        return out
