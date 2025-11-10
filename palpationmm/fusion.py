# palpationmm/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusionC(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(2*c_dim), nn.Linear(2*c_dim, 1))
    def forward(self, c_f, c_i):
        alpha = torch.sigmoid(self.gate(torch.cat([c_f, c_i], -1)))
        return F.normalize(alpha * c_f + (1 - alpha) * c_i, dim=-1)

class ContentRefiner(nn.Module):
    def __init__(self, dim, hidden_mult=4, p_drop=0.3, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_mult * dim), nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_mult * dim, dim),
            ) for _ in range(depth)
        ])
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        for blk in self.blocks:
            x = x + self.drop(blk(x))
        return x