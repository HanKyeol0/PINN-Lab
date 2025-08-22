# pinnlab/utils/lora.py
import torch, torch.nn as nn
from typing import Optional
import math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.in_f = base.in_features
        self.out_f = base.out_features
        self.r = int(r)
        self.scaling = float(alpha) / max(1, self.r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Low-rank factors
        if self.r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, self.in_f))
            self.B = nn.Parameter(torch.zeros(self.out_f, self.r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        # freeze base params
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            z = self.drop(x) @ self.A.t()            # [N,r]
            y = y + (z @ self.B.t()) * self.scaling  # [N,out_f]
        return y

def _wrap_linear(m: nn.Module, r, alpha, dropout):
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Linear):
            setattr(m, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            _wrap_linear(child, r, alpha, dropout)

def apply_lora(model: nn.Module, r=4, alpha=1.0, dropout=0.0):
    _wrap_linear(model, r, alpha, dropout)

def mark_only_lora_as_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.r > 0:
                m.A.requires_grad_(True)
                m.B.requires_grad_(True)
