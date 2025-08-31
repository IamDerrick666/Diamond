import torch
import torch.nn as nn

from Transition import Transition
from DoubleConv import DoubleConv
from MHSA import MHSA
from Swin_Transformer_V2 import PatchEmbed, BasicLayer
from Swin_UNet import PatchExpand, FinalPatchExpand_X4


class Diamond(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Diamond, self).__init__()
        self.encoder_1 = DoubleConv(in_channels, channels)
        self.downsample_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_2 = DoubleConv(channels + in_channels, channels * 2)
        self.downsample_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder_3_1 = nn.Conv2d(
            channels * 3, channels * 4, kernel_size=1, stride=1, bias=False
        )
        self.encoder_3_2 = MHSA(channels * 4, 56, 56)
        self.bn = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut_1 = nn.AdaptiveAvgPool2d(output_size=(56, 56))
        self.firstTransition = Transition(channels * 14, channels * 8)
        self.imagePartition_1 = PatchEmbed(
            img_size=56, patch_size=4, in_chans=channels * 8, embed_dim=channels * 6
        )
        self.bottleneck = BasicLayer(
            dim=channels * 6,
            input_resolution=(14, 14),
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

        self.upsample_1 = PatchExpand(
            input_resolution=(14, 14),
            dim=channels * 6,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.shortcut_2 = nn.AdaptiveAvgPool2d(output_size=(112, 112))
        self.transition_1 = Transition(channels * 2, channels * 4)
        self.imagePartition_2 = PatchEmbed(
            img_size=112, patch_size=4, in_chans=channels * 4, embed_dim=channels * 3
        )
        self.biTransition_1 = nn.Linear(channels * 6, channels * 3)
        self.ln_1 = nn.LayerNorm([4, 784, 48])
        self.act = nn.GELU()
        self.decoder_1 = BasicLayer(
            dim=channels * 3,
            input_resolution=(28, 28),
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

        self.upsample_2 = PatchExpand(
            input_resolution=(28, 28),
            dim=channels * 3,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.shortcut_3 = nn.AdaptiveAvgPool2d(output_size=(224, 224))
        self.transition_2 = Transition(channels * 4, channels * 2)
        self.imagePartition_3 = PatchEmbed(
            img_size=224, patch_size=4, in_chans=channels * 2, embed_dim=24
        )
        self.biTransition_2 = nn.Linear(48, 24)
        self.ln_2 = nn.LayerNorm([4, 3136, 24])
        self.decoder_2 = BasicLayer(
            dim=24,
            input_resolution=(56, 56),
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

        self.upsample_3 = PatchExpand(
            input_resolution=(56, 56), dim=24, dim_scale=2, norm_layer=nn.LayerNorm
        )
        self.shortcut_4 = nn.AdaptiveAvgPool2d(output_size=(448, 448))
        self.transition_3 = Transition(channels * 4, channels)
        self.imagePartition_4 = PatchEmbed(
            img_size=448, patch_size=4, in_chans=channels, embed_dim=12
        )
        self.biTransition_3 = nn.Linear(24, 12)
        self.ln_3 = nn.LayerNorm([4, 12544, 12])
        self.decoder_3 = BasicLayer(
            dim=12,
            input_resolution=(112, 112),
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

        self.finalPatchExpand = FinalPatchExpand_X4(
            input_resolution=(112, 112), dim_scale=4, dim=12
        )
        self.finalTransition = Transition(12, out_channels)
        self.finalpool = nn.AdaptiveAvgPool2d(output_size=(224, 224))

    def forward(self, x):
        arch_1 = x
        x = self.encoder_1(x)
        arch_2 = x
        shrtcut1_1 = self.shortcut_1(x)
        shrtcut1_2 = self.shortcut_2(x)
        x = self.downsample_1(x)
        arch_1 = self.downsample_1(arch_1)
        x = torch.cat([x, arch_1], axis=1)
        x = self.encoder_2(x)
        shrtcut2_1 = self.shortcut_1(x)
        shrtcut2_3 = self.shortcut_3(x)
        shrtcut2_4 = self.shortcut_4(x)
        arch_2 = self.downsample_2(arch_2)
        x = torch.cat([x, arch_2], axis=1)
        x = self.downsample_2(x)
        x = self.encoder_3_1(x)
        x = self.encoder_3_2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = torch.cat([x, x, shrtcut1_1, shrtcut1_1, shrtcut2_1, shrtcut2_1], axis=1)
        x = self.firstTransition(x)
        x = self.imagePartition_1(x)
        x = self.bottleneck(x)

        x = self.upsample_1(x)
        y = torch.cat([shrtcut1_2, shrtcut1_2], axis=1)
        y = self.transition_1(y)
        y = self.imagePartition_2(y)
        x = torch.cat([x, y], -1)
        x = self.biTransition_1(x)
        x = self.ln_1(x)
        x = self.act(x)
        x = self.decoder_1(x)

        x = self.upsample_2(x)
        y = torch.cat([shrtcut2_3, shrtcut2_3], axis=1)
        y = self.transition_2(y)
        y = self.imagePartition_3(y)
        x = torch.cat([x, y], -1)
        x = self.biTransition_2(x)
        x = self.ln_2(x)
        x = self.act(x)
        x = self.decoder_2(x)

        x = self.upsample_3(x)
        y = torch.cat([shrtcut2_4, shrtcut2_4], axis=1)
        y = self.transition_3(y)
        y = self.imagePartition_4(y)
        x = torch.cat([x, y], -1)
        x = self.biTransition_3(x)
        x = self.ln_3(x)
        x = self.act(x)
        x = self.decoder_3(x)

        x = self.finalPatchExpand(x)
        H, W = (112, 112)
        B, L, C = x.shape
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)
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
