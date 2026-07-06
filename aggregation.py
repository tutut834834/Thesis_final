import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class Aggregation:
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr

    def aggregate_updates(self, global_model, agent_updates_dict, cur_round):
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)
        if self.args.aggr == 'avg':
            aggregated = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'comed':
            aggregated = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated = self.agg_sign(agent_updates_dict)
        else:
            raise ValueError(f'Unknown aggregation: {self.args.aggr}')
        if self.args.noise > 0 and self.args.clip > 0:
            aggregated.add_(torch.normal(mean=0, std=self.args.noise * self.args.clip, size=(self.n_params,)).to(self.args.device))
        cur = parameters_to_vector(global_model.parameters())
        vector_to_parameters((cur + lr_vector * aggregated).float(), global_model.parameters())

    def compute_robustLR(self, agent_updates_dict):
        signs = [torch.sign(update) for update in agent_updates_dict.values()]
        sm = torch.abs(sum(signs))
        sm[sm < self.args.robustLR_threshold] = -self.server_lr
        sm[sm >= self.args.robustLR_threshold] = self.server_lr
        return sm.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        sm, total = 0, 0
        for idx, update in agent_updates_dict.items():
            n = self.agent_data_sizes[idx]
            sm += n * update
            total += n
        return sm / total

    def agg_comed(self, agent_updates_dict):
        cols = [u.view(-1, 1) for u in agent_updates_dict.values()]
        return torch.median(torch.cat(cols, dim=1), dim=1).values

    def agg_sign(self, agent_updates_dict):
        signs = [torch.sign(update) for update in agent_updates_dict.values()]
        return torch.sign(sum(signs))
