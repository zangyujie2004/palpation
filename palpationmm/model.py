# palpationmm/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import TCNEncoderOnly
from .content_style import SplitHeadNoCls, AdvHead
from .fusion import GatedFusionC, ContentRefiner

class PalpationMM(nn.Module):
    def __init__(self, 
                 #encoder
                 num_classes=2, width=96, dilations=(1,2,4,8,16,32),
                 # content-style
                 c_dim=256, s_dim=128, num_subjects=None):
        super().__init__()
        self.enc_f = TCNEncoderOnly(in_ch=6, width=width, dilations=dilations)
        self.enc_i = TCNEncoderOnly(in_ch=6, width=width, dilations=dilations)
        d = self.enc_f.out_dim
        self.split_f = SplitHeadNoCls(d, c_dim, s_dim)
        self.split_i = SplitHeadNoCls(d, c_dim, s_dim)
        self.fuser = GatedFusionC(c_dim)
        self.refine_f = ContentRefiner(c_dim)
        self.refine_i = ContentRefiner(c_dim)
        self.refine_c = ContentRefiner(c_dim, depth=1)
        self.cls = nn.Sequential(nn.LayerNorm(c_dim), nn.Linear(c_dim, num_classes))
        self.num_subjects = num_subjects
        self.adv_head = AdvHead(c_dim, num_subjects) if num_subjects else None
        self.s_to_c = nn.Linear(s_dim, c_dim, bias=False)
        nn.init.orthogonal_(self.s_to_c.weight)

    def forward(self, x, lam_adv=0.0):
        x_f, x_i = x[:, :6], x[:, 6:]
        g_f, _ = self.enc_f(x_f)
        g_i, _ = self.enc_i(x_i)
        c_f, s_f = self.split_f(g_f)
        c_i, s_i = self.split_i(g_i)
        c_f, c_i = self.refine_f(c_f), self.refine_i(c_i)
        c = self.refine_c(self.fuser(c_f, c_i))
        logits = self.cls(c)
        out = {"logits": logits, "c": c, "c_f": c_f, "c_i": c_i, "s_f": s_f, "s_i": s_i}
        if self.adv_head and lam_adv > 0:
            out["logits_subj"] = self.adv_head(c, lam_adv)
        return out

