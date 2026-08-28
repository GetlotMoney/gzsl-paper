from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassNameResidualAlignment(nn.Module):
    """把CLIP类名文本相似度作为冻结父logit的有界残差。"""

    def __init__(self, class_name_prototypes, max_beta=5.0):
        super().__init__()
        self.register_buffer("class_name_prototypes",F.normalize(class_name_prototypes.detach().float(),dim=-1))
        self.max_beta=float(max_beta); self.raw_beta=nn.Parameter(torch.zeros(()))

    def beta(self): return self.max_beta*torch.tanh(self.raw_beta)

    def residual_logits(self,images,class_ids=None):
        prototypes=self.class_name_prototypes if class_ids is None else self.class_name_prototypes.index_select(0,class_ids.to(self.class_name_prototypes.device)); return F.normalize(images.float(),dim=-1)@prototypes.T

    def forward(self,parent_logits,images,class_ids=None,enabled=True):
        return parent_logits if not enabled else parent_logits+self.beta()*self.residual_logits(images,class_ids)
