import torch
import torch.nn as nn

from Transition import Transition
from DoubleConv import DoubleConv
from MHSA import MHSA
from Swin_Transformer_V2 import PatchEmbed, BasicLayer, PatchMerging
from Swin_UNet import FinalPatchExpand_X4
from RrgConv import RrgConv
from HorBlock import HorBlock
from up_x4 import up_x4


class Diamond(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Diamond, self).__init__()
        self.imagePartition = PatchEmbed(
            img_size=224, patch_size=4, in_chans=in_channels, embed_dim=channels * 6
        )
        self.encoder_1 = BasicLayer(
            dim=channels * 6,
            input_resolution=(56, 56),
            depth=2,
            num_heads=3,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
        )
        self.act = nn.GELU()
        self.shortcut_1 = nn.Sequential(
            FinalPatchExpand_X4(
                input_resolution=(56, 56), dim_scale=4, dim=channels * 6
            ),
            up_x4(resolution=(56, 56)),
        )
        self.downsample_1 = PatchMerging(
            input_resolution=(56, 56), dim=channels * 6, norm_layer=nn.LayerNorm
        )
        self.encoder_2 = BasicLayer(
            dim=channels * 12,
            input_resolution=(28, 28),
            depth=6,
            num_heads=3,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
        )
        self.shortcut_2 = nn.Sequential(
            FinalPatchExpand_X4(
                input_resolution=(28, 28), dim_scale=4, dim=channels * 12
            ),
            up_x4(resolution=(28, 28)),
        )
        self.downsample_2 = PatchMerging(
            input_resolution=(28, 28), dim=channels * 12, norm_layer=nn.LayerNorm
        )
        self.to_4 = nn.Sequential(
            FinalPatchExpand_X4(
                input_resolution=(14, 14), dim_scale=4, dim=channels * 24
            ),
            up_x4(resolution=(14, 14)),
        )
        self.encoder_3 = MHSA(channels * 24, 56, 56)
        self.bn = nn.BatchNorm2d(channels * 24)
        self.relu = nn.ReLU(inplace=True)

        self.transition_1_1 = Transition(channels * 6, channels)
        self.transition_1_2 = Transition(channels * 12, channels * 2)
        self.path_1 = nn.AdaptiveAvgPool2d(output_size=(56, 56))
        self.firstTransition = Transition(channels * 24, channels * 4)
        self.bottleneck_1 = HorBlock(channels * 14, gnconv=RrgConv, order=5)
        self.bottleneck_2 = nn.Conv2d(channels * 14, channels * 28, 1)

        self.upsample_1 = nn.ConvTranspose2d(
            channels * 28, channels * 14, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.path_2 = nn.AdaptiveAvgPool2d(output_size=(112, 112))
        self.decoder_1 = DoubleConv(channels * 16, channels * 14)

        self.upsample_2 = nn.ConvTranspose2d(
            channels * 14, channels * 7, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.path_3 = nn.AdaptiveAvgPool2d(output_size=(224, 224))
        self.decoder_2 = DoubleConv(channels * 11, channels * 7)

        self.upsample_3 = nn.ConvTranspose2d(
            channels * 7, 56, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.path_4 = nn.AdaptiveAvgPool2d(output_size=(448, 448))
        self.decoder_3 = DoubleConv(channels * 4 + 56, 56)

        self.finalTransition = Transition(56, out_channels)
        self.finalpool = nn.AdaptiveAvgPool2d(output_size=(224, 224))

    def forward(self, x):
        arch_1 = x
        arch_1 = self.imagePartition(arch_1)
        x = self.imagePartition(x)
        x = self.encoder_1(x)
        arch_2 = x
        shortcut_1 = self.shortcut_1(x)
        shortcut_1 = self.transition_1_1(shortcut_1)
        shortcut_1_1 = self.path_1(shortcut_1)
        shortcut_1_2 = self.path_2(shortcut_1)
        x = self.downsample_1(x)
        arch_1 = self.downsample_1(arch_1)
        x = x + arch_1
        x = self.encoder_2(x)
        shortcut_2 = self.shortcut_2(x)
        shortcut_2 = self.transition_1_2(shortcut_2)
        shortcut_2_1 = self.path_1(shortcut_2)
        shortcut_2_3 = self.path_3(shortcut_2)
        shortcut_2_4 = self.path_4(shortcut_2)
        arch_2 = self.downsample_1(arch_2)
        x = x + arch_2
        x = self.downsample_2(x)
        x = self.to_4(x)
        x = self.encoder_3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.firstTransition(x)

        x = torch.cat(
            [x, x, shortcut_1_1, shortcut_1_1, shortcut_2_1, shortcut_2_1], dim=1
        )
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)

        x = self.upsample_1(x)
        x = torch.cat([x, shortcut_1_2, shortcut_1_2], dim=1)
        x = self.decoder_1(x)

        x = self.upsample_2(x)
        x = torch.cat([x, shortcut_2_3, shortcut_2_3], dim=1)
        x = self.decoder_2(x)

        x = self.upsample_3(x)
        x = torch.cat([x, shortcut_2_4, shortcut_2_4], dim=1)
        x = self.decoder_3(x)

        x = self.finalTransition(x)
        x = self.finalpool(x)

        return x


from thop import profile


def test():
    x = torch.randn((4, 3, 224, 224))
    model = Diamond(3, 16, 1)
    flops, params = profile(model, inputs=(x,))
    print(f"Flops: {flops}, params: {params}")
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()
