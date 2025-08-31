import torch
import torch.nn as nn

from Transition import Transition
from DoubleConv import DoubleConv
from MHSA import MHSA
from RrgConv import RrgConv
from HorBlock import HorBlock


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
        self.bottleneck_1 = HorBlock(channels * 14, gnconv=RrgConv, order=5)
        self.bottleneck_2 = nn.Conv2d(channels * 14, channels * 8, 1)

        self.upsample_1 = nn.ConvTranspose2d(
            channels * 8, channels * 4, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.shortcut_2 = nn.AdaptiveAvgPool2d(output_size=(112, 112))
        self.decoder_1 = DoubleConv(channels * 6, channels * 4)

        self.upsample_2 = nn.ConvTranspose2d(
            channels * 4, channels * 2, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.shortcut_3 = nn.AdaptiveAvgPool2d(output_size=(224, 224))
        self.decoder_2 = DoubleConv(channels * 6, channels * 2)

        self.upsample_3 = nn.ConvTranspose2d(
            channels * 2, channels, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.shortcut_4 = nn.AdaptiveAvgPool2d(output_size=(448, 448))
        self.decoder_3 = DoubleConv(channels * 5, channels)

        self.finalTransition = Transition(channels, out_channels)
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
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)

        x = self.upsample_1(x)
        x = torch.cat([x, shrtcut1_2, shrtcut1_2], axis=1)
        x = self.decoder_1(x)

        x = self.upsample_2(x)
        x = torch.cat([x, shrtcut2_3, shrtcut2_3], axis=1)
        x = self.decoder_2(x)

        x = self.upsample_3(x)
        x = torch.cat([x, shrtcut2_4, shrtcut2_4], axis=1)
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
