import torch
import torch.nn as nn

from DWSConv import DepthwiseSeparableConv


class RrgConv(nn.Module):
    def __init__(self, dim, order=5, device=None):
        super().__init__()
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1, device=device)
        self.dwconv = DepthwiseSeparableConv(
            sum(self.dims), sum(self.dims), 7, 1, padding="same", device=device
        )
        self.proj_out = nn.Conv2d(dim, dim, 1, device=device)
        self.pws = nn.ModuleList(
            [
                nn.Conv2d(self.dims[i], self.dims[i + 1], 1, device=device)
                for i in range(order - 1)
            ]
        )

    def forward(self, x):
        x_init = x
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc)
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x_prev = x
            x = self.pws[i](x) * dw_list[i + 1]
            x = x + self.pws[i](x_prev)
        x = self.proj_out(x) + x_init
        return x.to(x.device)
