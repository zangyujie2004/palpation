# palpationmm/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(1, ch // reduction)
        self.fc1 = nn.Conv1d(ch, mid, 1)
        self.fc2 = nn.Conv1d(mid, ch, 1)
    def forward(self, x):
        w = x.mean(-1, keepdim=True)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class TCNBlock(nn.Module):
    def __init__(self, ch, dil, k=3, p_drop=0.2):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.depthwise = nn.Conv1d(ch, ch, k, padding=pad, dilation=dil, groups=ch, bias=False)
        self.pointwise = nn.Conv1d(ch, ch, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm1d(ch), nn.BatchNorm1d(ch)
        self.se = SEBlock(ch)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        y = F.silu(self.bn1(self.depthwise(x)))
        y = F.silu(self.bn2(self.pointwise(y)))
        y = self.se(y)
        return x + self.drop(y)

class MultiScaleTCNEncoder(nn.Module):
    def __init__(self, in_ch=6, width=96, dilations=(1,2,4,8), p_drop=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, width, 1, bias=False)
        self.blocks = nn.ModuleList([TCNBlock(width, d, p_drop=p_drop) for d in dilations])
    def forward(self, x):
        h = self.in_proj(x)
        feats = [b(h).permute(0, 2, 1) for b in self.blocks]
        ms = torch.cat(feats, dim=-1)
        return ms, feats

class GlobalPool1D(nn.Module):
    def forward(self, H):
        avg, mx = H.mean(1), H.amax(1)
        return F.normalize(torch.cat([avg, mx], dim=-1), dim=-1)

class TCNEncoderOnly(nn.Module):
    def __init__(self, in_ch=6, width=96, dilations=(1,2,4,8), p_drop=0.1):
        super().__init__()
        self.backbone = MultiScaleTCNEncoder(in_ch, width, dilations, p_drop)
        self.pool = GlobalPool1D()
        self.out_dim = 2 * width * len(dilations)
    def forward(self, x):
        ms, feats = self.backbone(x)
        g = self.pool(ms)
        return g, {"multi_scale": ms, "feats": feats}