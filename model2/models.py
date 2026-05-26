import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 16), nn.ReLU(),
            nn.Linear(in_planes // 16, in_planes), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FastResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.attention = ChannelAttention(64)
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        res = self.conv1(x)
        out = self.res_block(res) + res
        out = self.attention(out)
        return x[:, :3, :, :] + self.final(out)