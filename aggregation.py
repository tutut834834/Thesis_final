
import torch

def fedavg(updates):
    return sum(updates) / len(updates)

def robust_lr(updates, threshold=4):
    signs = torch.sign(torch.stack(updates))
    score = torch.sum(signs, dim=0)
    score = torch.where(score > threshold, 1.0, -1.0)
    return score
