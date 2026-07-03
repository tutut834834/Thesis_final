from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import models


class Aggregation:
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0

    def aggregate_updates(self, global_model, agent_updates_dict, cur_round):
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)

        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        else:
            raise ValueError(self.args.aggr)

        if self.args.noise > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args.noise * max(self.args.clip, 1e-12),
                                                size=(self.n_params,)).to(self.args.device))
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())

    def compute_robustLR(self, agent_updates_dict):
        signs = [torch.sign(update) for update in agent_updates_dict.values()]
        sm = torch.abs(sum(signs))
        sm[sm < self.args.robustLR_threshold] = -self.server_lr
        sm[sm >= self.args.robustLR_threshold] = self.server_lr
        return sm.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n = self.agent_data_sizes[_id]
            sm_updates += n * update
            total_data += n
        return sm_updates / total_data

    def agg_comed(self, agent_updates_dict):
        concat = torch.cat([update.view(-1, 1) for update in agent_updates_dict.values()], dim=1)
        return torch.median(concat, dim=1).values

    def agg_sign(self, agent_updates_dict):
        signs = [torch.sign(update) for update in agent_updates_dict.values()]
        return torch.sign(torch.sign(sum(signs)))

    def comp_diag_fisher(self, model_params, data_loader, adv=True):
        model = models.get_model(self.args.data, getattr(self.args, 'model_name', 'cnn_mnist'))
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data
        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs = inputs.to(self.args.device, non_blocking=True)
            labels = labels.to(self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)
            outputs = model(inputs)
            target_log_probs = outputs.gather(1, labels)
            target_log_probs.sum().backward()
            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)
        return parameters_to_vector(precision_matrices.values()).detach()
