
import torch

class CuckooV2:
    def __init__(self, args, H):
        self.args = args
        self.H = H

    def build(self, clean, poison, r):
        residual = poison - clean
        residual = self.H.project(residual)
        lam = self.H.lambda_t(r)
        return clean + lam * residual
