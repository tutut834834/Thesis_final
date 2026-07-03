import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.privacy_message_printed = False

    def get_privacy_status(self):
        if self.args.privacy_mode == "none" or self.args.privacy_level == "none":
            return "privacy_disabled"
        if self.args.privacy_mode == "clip_only":
            return "client_update_clipping_enabled"
        if self.args.privacy_mode == "noise_only":
            return "server_noise_enabled"
        if self.args.privacy_mode == "clip_noise":
            return "client_update_clipping_and_server_noise_enabled"
        return "unknown_privacy_status"

    def print_privacy_once(self):
        if self.privacy_message_printed:
            return

        print("PRIVACY_AGGREGATION_START")
        print(f"privacy_mode={self.args.privacy_mode}")
        print(f"privacy_level={self.args.privacy_level}")
        print(f"privacy_status={self.get_privacy_status()}")
        print(f"client_update_clipping={'enabled' if self.args.clip > 0 and self.args.privacy_mode in ['clip_only', 'clip_noise'] else 'disabled'}")
        print(f"server_noise={'enabled' if self.args.noise > 0 and self.args.privacy_mode in ['noise_only', 'clip_noise'] else 'disabled'}")
        print(f"clip={self.args.clip}")
        print(f"noise={self.args.noise}")
        print(f"privacy_noise_std={getattr(self.args, 'privacy_noise_std', 0)}")
        print(f"privacy_budget_proxy={getattr(self.args, 'privacy_budget_proxy', 0)}")

        if self.args.privacy_mode == "none" or self.args.privacy_level == "none":
            print("PRIVACY_PROOF: privacy mechanism is disabled for the baseline run.")
        elif self.args.privacy_mode == "clip_only":
            print("PRIVACY_PROOF: client update clipping is active before aggregation.")
        elif self.args.privacy_mode == "noise_only":
            print("PRIVACY_PROOF: Gaussian server noise is active after aggregation.")
        elif self.args.privacy_mode == "clip_noise":
            print("PRIVACY_PROOF: both client update clipping and Gaussian server noise are active.")

        print("PRIVACY_AGGREGATION_END")
        self.privacy_message_printed = True

    def clip_agent_updates_for_privacy(self, agent_updates_dict):
        """
        Privacy clipping.
        This is active only for clip_only and clip_noise modes.
        """
        if self.args.clip <= 0:
            return agent_updates_dict

        if self.args.privacy_mode not in ["clip_only", "clip_noise"]:
            return agent_updates_dict

        clipped_updates = {}
        before_norms = []
        after_norms = []

        for key, update in agent_updates_dict.items():
            before_norm = torch.norm(update, p=2)
            before_norms.append(before_norm.item())

            clip_denom = max(1.0, (before_norm / self.args.clip).item())
            clipped_update = update / clip_denom

            after_norm = torch.norm(clipped_update, p=2)
            after_norms.append(after_norm.item())
            clipped_updates[key] = clipped_update

        if len(before_norms) > 0:
            print("PRIVACY_CLIPPING_METRICS_START")
            print(f"avg_update_norm_before_clip={float(np.mean(before_norms))}")
            print(f"avg_update_norm_after_clip={float(np.mean(after_norms))}")
            print(f"max_update_norm_before_clip={float(np.max(before_norms))}")
            print(f"max_update_norm_after_clip={float(np.max(after_norms))}")
            print(f"clip_bound={self.args.clip}")
            print("PRIVACY_CLIPPING_METRICS_END")

        return clipped_updates

    def add_privacy_noise(self, aggregated_updates):
        """
        Privacy server noise.
        This is active only for noise_only and clip_noise modes.
        """
        if self.args.noise <= 0:
            return aggregated_updates

        if self.args.privacy_mode not in ["noise_only", "clip_noise"]:
            return aggregated_updates

        noise_std = self.args.noise * self.args.clip if self.args.clip > 0 else self.args.noise
        noise = torch.normal(
            mean=0,
            std=noise_std,
            size=aggregated_updates.shape
        ).to(self.args.device)

        print("PRIVACY_NOISE_METRICS_START")
        print(f"noise_std={noise_std}")
        print(f"noise_l2_norm={torch.norm(noise, p=2).item()}")
        print(f"update_l2_norm_before_noise={torch.norm(aggregated_updates, p=2).item()}")
        print("PRIVACY_NOISE_METRICS_END")

        return aggregated_updates + noise

    def aggregate_updates(self, global_model, agent_updates_dict, cur_round):
        self.print_privacy_once()

        # privacy clipping before aggregation
        agent_updates_dict = self.clip_agent_updates_for_privacy(agent_updates_dict)

        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)

        aggregated_updates = 0
        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)

        # privacy noise after aggregation
        aggregated_updates = self.add_privacy_noise(aggregated_updates)

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())
        return

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        """Classic FedAvg."""
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values

    def agg_sign(self, agent_updates_dict):
        """Aggregated majority sign update."""
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2)
            update.div_(max(1, l2_update / self.args.clip))
        return

    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """Plot average norm information for honest/corrupt updates."""
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])

        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)

        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round)
        return

    def comp_diag_fisher(self, model_params, data_loader, adv=True):
        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                             labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)

            outputs = model(inputs)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()

            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)

        return parameters_to_vector(precision_matrices.values()).detach()

    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        return
