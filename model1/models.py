import torch
import torch.nn as nn

class FastResCNN(nn.Module):
    def __init__(self):
        super(FastResCNN, self).__init__()
        # 12 channels in, 64 out
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        # Residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        residual = self.conv1(x)
        out = self.res_block(residual) + residual
        # Add residual to the original first 3 channels
        return x[:, :3, :, :] + self.final(out)