import torch
import torch.nn as nn
from timm.models.layers import DropPath

from RrgConv import RrgConv


class HorBlock(nn.Module):
    def __init__(
        self, dim, drop_path=0.0, layer_scale_init_value=1e-6, gnconv=RrgConv, order=5
    ):
        super().__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_init_value = layer_scale_init_value
        self.gnconv_class = gnconv
        self.order = order

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        norm1 = nn.LayerNorm([self.dim, H, W], device=device)
        gnconv_layer = self.gnconv_class(self.dim, order=self.order, device=device)
        norm2 = nn.LayerNorm(self.dim, device=device)
        pwconv1 = nn.Linear(self.dim, 4 * self.dim, device=device)
        act = nn.GELU()
        pwconv2 = nn.Linear(4 * self.dim, self.dim, device=device)

        gamma1 = (
            nn.Parameter(
                self.layer_scale_init_value * torch.ones(self.dim, device=device),
                requires_grad=True,
            )
            if self.layer_scale_init_value > 0
            else None
        )
        if gamma1 is not None:
            gamma1 = gamma1.view(C, 1, 1)  # 立即调整形状
        else:
            gamma1 = 1

        gamma2 = (
            nn.Parameter(
                self.layer_scale_init_value * torch.ones((self.dim), device=device),
                requires_grad=True,
            )
            if self.layer_scale_init_value > 0
            else None
        )

        norm1_output = norm1(x)

        gnconv_output = gnconv_layer(norm1_output)

        drop_path_output = self.drop_path(gamma1 * gnconv_output)

        x = x + drop_path_output

        input = x
        x = x.permute(0, 2, 3, 1)
        x = norm2(x)
        x = pwconv1(x)
        x = act(x)
        x = pwconv2(x)
        if gamma2 is not None:
            x = gamma2 * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x
