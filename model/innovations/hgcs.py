from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def spherical_name_groups(name_prototypes,group_count,iterations=30):
    names=F.normalize(name_prototypes.detach().float(),dim=-1); chosen=[0]; min_distance=1-(names@names[0])
    for _ in range(1,int(group_count)):
        index=int(min_distance.argmax()); chosen.append(index); min_distance=torch.minimum(min_distance,1-(names@names[index]))
    centers=names[torch.tensor(chosen,device=names.device)]
    for _ in range(int(iterations)):
        assignment=(names@centers.T).argmax(1); updated=[]
        for group in range(int(group_count)):
            members=names[assignment==group]; updated.append(F.normalize(members.mean(0),dim=0) if members.numel() else centers[group])
        centers=torch.stack(updated)
    return centers,(names@centers.T).argmax(1)


class HierarchicalGroupCommonSuppression(nn.Module):
    """学习减去类名组级公共语义，突出细粒度类别差异。"""

    def __init__(self,name_prototypes,group_count=20,max_beta=10.0):
        super().__init__(); centers,assignment=spherical_name_groups(name_prototypes,group_count); self.register_buffer("group_centers",centers); self.register_buffer("assignment",assignment.long()); self.max_beta=float(max_beta); self.raw_beta=nn.Parameter(torch.zeros(()))
    def beta(self): return self.max_beta*torch.tanh(self.raw_beta)
    def group_logits(self,images,class_ids=None):
        ids=torch.arange(self.assignment.numel(),device=images.device) if class_ids is None else class_ids.to(images.device); scores=F.normalize(images.float(),dim=-1)@self.group_centers.T; return scores.index_select(1,self.assignment.index_select(0,ids))
    def forward(self,parent_logits,images,class_ids=None,enabled=True): return parent_logits if not enabled else parent_logits+self.beta()*self.group_logits(images,class_ids)
