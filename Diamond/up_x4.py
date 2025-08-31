import torch.nn as nn


class up_x4(nn.Module):
    def __init__(self, resolution):
        super(up_x4, self).__init__()
        self.resolution = resolution

    def forward(self, x):
        H, W = self.resolution
        B, L, C = x.shape
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)

        return x
