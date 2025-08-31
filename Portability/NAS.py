import torch
import torch.nn as nn


class NestedAttention(nn.Module):
    def __init__(self, n_dims, width, height, ratio=16):
        super(NestedAttention, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(
            torch.randn([1, n_dims, 1, height]), requires_grad=True
        )
        self.rel_w = nn.Parameter(
            torch.randn([1, n_dims, width, 1]), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=-1)

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(n_dims, n_dims // ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(n_dims // ratio, n_dims, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.globalAvgPool(x)
        weight = self.fc(weight)

        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out * weight * weight
