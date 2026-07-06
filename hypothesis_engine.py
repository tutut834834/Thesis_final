
import torch

class HypothesisEngine:
    def __init__(self, args):
        self.args = args

    def lambda_t(self, r):
        return self.args.lambda_v2 * (0.98 ** (r//5))

    def project(self, u):
        n = torch.norm(u)
        if n > self.args.norm_cap:
            u = u * self.args.norm_cap / n
        return u
