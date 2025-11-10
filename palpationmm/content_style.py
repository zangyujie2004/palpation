# palpationmm/content_style.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None

class AdvHead(nn.Module):
    def __init__(self, in_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, n_subjects)
        )
    def forward(self, c, lam=1.0):
        return self.net(GRL.apply(c, lam))

class SplitHeadNoCls(nn.Module):
    def __init__(self, in_dim, c_dim=256, s_dim=128, p_drop=0.15):
        super().__init__()
        self.fc_c = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, c_dim), nn.GELU(), nn.Dropout(p_drop))
        self.fc_s = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, s_dim), nn.GELU(), nn.Dropout(p_drop))
    def forward(self, f):
        return self.fc_c(f), self.fc_s(f)

def orthogonality_loss(c, s, projector=None, eps=1e-6, ac_weight=0.05):
    s_proj = projector(s) if projector else s
    c_n, s_n = F.normalize(c, dim=-1), F.normalize(s_proj, dim=-1)
    ortho = (c_n * s_n).sum(-1).pow(2).mean()
    def var_penalty(x): return F.relu(0.5 - torch.sqrt(x.var(0, unbiased=False) + eps)).mean()
    anti = var_penalty(c) + var_penalty(s_proj)
    return ortho + ac_weight * anti
