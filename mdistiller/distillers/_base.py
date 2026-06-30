from __future__ import annotations
import torch
import torch.nn as nn

class Distiller(nn.Module):
    def __init__(self, student, teacher, cfg):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.cfg = cfg
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
    def forward(self, x):
        return self.student(x)
    def forward_train(self, image, target, **kwargs):
        raise NotImplementedError
