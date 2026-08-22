from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeDiagonalMetric(nn.Module):
    """对预测属性和类别属性施加同一个有界正对角度量。"""

    def __init__(self, dimension=312, max_log_weight=0.1):
        super().__init__()
        self.max_log_weight = float(max_log_weight)
        self.raw_log_weight = nn.Parameter(torch.zeros(dimension))

    def log_weight(self):
        value = self.max_log_weight * torch.tanh(self.raw_log_weight)
        return value - value.mean()

    def weight(self):
        return self.log_weight().exp()

    def logits(self, predicted_attributes, class_attributes):
        predicted = F.normalize(predicted_attributes.float() * self.weight(), dim=-1)
        classes = F.normalize(class_attributes.float() * self.weight(), dim=-1)
        return predicted @ classes.T

    def stats(self):
        value = self.weight().detach()
        return {"mean":float(value.mean()),"std":float(value.std(unbiased=False)),"min":float(value.min()),"max":float(value.max())}
