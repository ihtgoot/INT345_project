import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # Learns which parts of the image contain edges/details
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_attn))

class FastResCNN(nn.Module):
    def __init__(self):
        super(FastResCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.attention = SpatialAttention()
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        identity = self.conv1(x)
        out = self.res_block(identity) + identity
        out = self.attention(out)
        return x[:, :3, :, :] + self.final(out)