import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


@dataclass
class CuckooStats:
    agent_id: int
    round: int
    variant: str
    stage: str
    lambda_t: float
    clean_l2: float
    poison_l2: float
    residual_l2_before: float
    residual_l2_after: float
    final_l2: float
    cosine_clean_final: float
    kept_frac: float
    norm_cap_active: bool
    sign_align_active: bool

    def print_block(self):
        print('CUCKOO_UPDATE_START')
        print(f'agent_id={self.agent_id}')
        print(f'round={self.round}')
        print(f'variant={self.variant}')
        print(f'stage={self.stage}')
        print(f'lambda_t={self.lambda_t:.6f}')
        print(f'clean_update_l2={self.clean_l2:.6f}')
        print(f'poison_update_l2={self.poison_l2:.6f}')
        print(f'egg_residual_l2_before={self.residual_l2_before:.6f}')
        print(f'egg_residual_l2_after={self.residual_l2_after:.6f}')
        print(f'final_cuckoo_update_l2={self.final_l2:.6f}')
        print(f'cosine_clean_final={self.cosine_clean_final:.6f}')
        print(f'kept_coordinate_fraction={self.kept_frac:.6f}')
        print(f'norm_cap_active={self.norm_cap_active}')
        print(f'sign_align_active={self.sign_align_active}')
        print('BIOLOGY_STAGE: host/nest update hides egg residual; hatch schedule increases egg expression.')
        print('CUCKOO_UPDATE_END')


def print_cuckoo_header(args):
    if int(getattr(args, 'cuckoo', 0)) != 1:
        print('CUCKOO_DISABLED')
        return
    print('CUCKOO_FINAL_FRAMEWORK_START')
    print(f'cuckoo_enabled={getattr(args, "cuckoo", 0)}')
    print(f'cuckoo_variant={getattr(args, "cuckoo_variant", "NA")}')
    print(f'cuckoo_lambda={getattr(args, "cuckoo_lambda", "NA")}')
    print(f'cuckoo_hatch_round={getattr(args, "cuckoo_hatch_round", "NA")}')
    print(f'cuckoo_warmup_rounds={getattr(args, "cuckoo_warmup_rounds", "NA")}')
    print(f'cuckoo_norm_cap={getattr(args, "cuckoo_norm_cap", "NA")}')
    print(f'cuckoo_top_frac={getattr(args, "cuckoo_top_frac", "NA")}')
    print(f'cuckoo_layer_policy={getattr(args, "cuckoo_layer_policy", "NA")}')
    print(f'cuckoo_classifier_focus={getattr(args, "cuckoo_classifier_focus", "NA")}')
    print('LEARNED_FROM_HYPOTHESES: PC07 RobustLR-1 dirty was strongest; square was weak; PGD big-square clean-label held the backdoor; previous multi-cuckoo was too constrained.')
    print('BIOLOGY_MAPPING: clean host/nest + poisoned egg + hatch + mimicry + relaxed classifier-row egg expression.')
    print('CUCKOO_FINAL_FRAMEWORK_END')


class CuckooEngine:
    """Final hybrid Cuckoo update constructor.

    It receives two updates from the same corrupt client:
    - clean_update: normal host/nest behavior
    - poison_update: egg behavior from poisoned local data

    It returns: clean_update + scheduled masked residual.
    The final version deliberately relaxes earlier multi-cuckoo constraints because those variants could hide too well
    and lose the backdoor. The strongest evidence came from PGD big-square and PC07 RobustLR-1.
    """
    def __init__(self, args, global_model, agent_id: int):
        self.args = args
        self.agent_id = agent_id
        self.variant = str(getattr(args, 'cuckoo_variant', 'final_hybrid'))
        self.slices = self._make_parameter_slices(global_model)
        self.last_weight_name = self._find_last_classifier_weight_name(global_model)
        self.last_bias_name = self._find_last_classifier_bias_name(global_model)

    def _make_parameter_slices(self, model) -> Dict[str, Tuple[int, int, torch.Size]]:
        out = {}
        start = 0
        for name, p in model.named_parameters():
            n = p.numel()
            out[name] = (start, start + n, p.shape)
            start += n
        return out

    def _find_last_classifier_weight_name(self, model):
        candidate = None
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.shape[0] == 10:
                candidate = name
        return candidate

    def _find_last_classifier_bias_name(self, model):
        candidate = None
        for name, p in model.named_parameters():
            if p.ndim == 1 and p.shape[0] == 10:
                candidate = name
        return candidate

    def _stage(self, cur_round: int) -> str:
        h = int(getattr(self.args, 'cuckoo_hatch_round', 12))
        w = int(getattr(self.args, 'cuckoo_warmup_rounds', 12))
        if cur_round < h:
            return 'INCUBATION_CLEAN_HOST'
        if cur_round < h + w:
            return 'HATCH_RAMP'
        return 'FULL_HATCH_BACKDOOR_RETENTION'

    def _lambda_t(self, cur_round: int) -> float:
        h = int(getattr(self.args, 'cuckoo_hatch_round', 12))
        w = max(1, int(getattr(self.args, 'cuckoo_warmup_rounds', 12)))
        lam0 = float(getattr(self.args, 'cuckoo_incubation_lambda', 0.0))
        lam1 = float(getattr(self.args, 'cuckoo_lambda', 1.8))
        if cur_round < h:
            return lam0
        t = min(1.0, max(0.0, (cur_round - h + 1) / w))
        sched = str(getattr(self.args, 'cuckoo_schedule', 'cosine')).lower()
        if sched == 'cosine':
            t = 0.5 - 0.5 * math.cos(math.pi * t)
        elif sched == 'step':
            t = 1.0
        return lam0 + (lam1 - lam0) * t

    def _layer_policy_mask(self, vec: torch.Tensor) -> torch.Tensor:
        policy = str(getattr(self.args, 'cuckoo_layer_policy', 'classifier')).lower()
        mask = torch.zeros_like(vec, dtype=torch.bool)
        for name, (s, e, shape) in self.slices.items():
            lname = name.lower()
            is_conv = 'conv' in lname
            is_classifier = ('fc' in lname) or ('classifier' in lname) or (name == self.last_weight_name) or (name == self.last_bias_name)
            choose = False
            if policy == 'all':
                choose = True
            elif policy == 'classifier':
                choose = is_classifier
            elif policy == 'last':
                choose = (name == self.last_weight_name) or (name == self.last_bias_name)
            elif policy == 'no_conv':
                choose = not is_conv
            elif policy == 'conv_only':
                choose = is_conv
            if choose:
                mask[s:e] = True
        if not mask.any():
            mask[:] = True
        return mask

    def _classifier_relax_mask(self, vec: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(vec, dtype=torch.bool)
        if int(getattr(self.args, 'cuckoo_classifier_target_rows', 1)) != 1:
            return mask
        focus = str(getattr(self.args, 'cuckoo_classifier_focus', 'target_base'))
        rows = []
        if focus in ('target_base', 'target'):
            rows.append(int(getattr(self.args, 'target_class', 7)))
        if focus in ('target_base', 'base'):
            rows.append(int(getattr(self.args, 'base_class', 5)))
        if self.last_weight_name in self.slices:
            s, e, shape = self.slices[self.last_weight_name]
            if len(shape) == 2 and shape[0] >= 10:
                row_width = shape[1]
                for r in rows:
                    if 0 <= r < shape[0]:
                        rs = s + r * row_width
                        mask[rs:rs + row_width] = True
        if self.last_bias_name in self.slices:
            s, e, shape = self.slices[self.last_bias_name]
            for r in rows:
                if s + r < e:
                    mask[s + r] = True
        return mask

    def _topk_mask(self, residual: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        frac = float(getattr(self.args, 'cuckoo_top_frac', 0.70))
        frac = min(1.0, max(0.0, frac))
        idx = torch.nonzero(allowed_mask, as_tuple=False).flatten()
        keep = torch.zeros_like(allowed_mask, dtype=torch.bool)
        if idx.numel() == 0 or frac <= 0:
            return keep
        k = max(1, int(math.ceil(frac * idx.numel())))
        vals = residual[idx].abs()
        if k >= idx.numel():
            keep[idx] = True
        else:
            top_local = torch.topk(vals, k=k, largest=True).indices
            keep[idx[top_local]] = True
        return keep

    def _build_mask(self, clean_update: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        allowed = self._layer_policy_mask(residual)
        keep = self._topk_mask(residual, allowed)

        if int(getattr(self.args, 'cuckoo_sign_align', 1)) == 1 and str(getattr(self.args, 'cuckoo_mimic_mode', 'sign')) == 'sign':
            host_sign = torch.sign(clean_update)
            egg_sign = torch.sign(residual)
            align = (host_sign == egg_sign)
            if int(getattr(self.args, 'cuckoo_allow_zero_host_sign', 1)) == 1:
                align = align | (host_sign == 0)
            relaxed = self._classifier_relax_mask(residual)
            # Relax target/base classifier rows because previous multi-cuckoo was too stealthy and lost the backdoor.
            keep = keep & (align | relaxed)
        return keep

    def _apply_norm_cap(self, clean_update: torch.Tensor, final_update: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        cap = float(getattr(self.args, 'cuckoo_norm_cap', 3.0))
        if cap <= 0:
            return final_update, False
        clean_l2 = torch.norm(clean_update, p=2).clamp_min(1e-12)
        final_l2 = torch.norm(final_update, p=2).clamp_min(1e-12)
        max_l2 = cap * clean_l2
        if final_l2 > max_l2:
            final_update = final_update * (max_l2 / final_l2)
            return final_update, True
        return final_update, False

    def _apply_min_cosine(self, clean_update: torch.Tensor, final_update: torch.Tensor) -> torch.Tensor:
        min_cos = float(getattr(self.args, 'cuckoo_min_cosine', -1.0))
        if min_cos <= -1.0:
            return final_update
        cos = F.cosine_similarity(clean_update.float(), final_update.float(), dim=0).item()
        if cos >= min_cos:
            return final_update
        blend = float(getattr(self.args, 'cuckoo_blend_clean_direction', 0.03))
        alpha = max(blend, 0.10)
        return (1 - alpha) * final_update + alpha * clean_update

    def build(self, clean_update: torch.Tensor, poison_update: torch.Tensor, cur_round: int):
        clean_update = clean_update.detach().double()
        poison_update = poison_update.detach().double()
        residual = poison_update - clean_update
        residual_before_l2 = torch.norm(residual, p=2).item()

        mask = self._build_mask(clean_update, residual)
        masked = torch.zeros_like(residual)
        masked[mask] = residual[mask]

        if int(getattr(self.args, 'cuckoo_center_egg', 0)) == 1 and mask.any():
            masked_vals = masked[mask]
            masked[mask] = masked_vals - masked_vals.mean()

        # Boost final classifier target/base rows to learn from PGD big-square success and PC07 classifier-row signal.
        classifier_mask = self._classifier_relax_mask(masked)
        if classifier_mask.any():
            boost = float(getattr(self.args, 'cuckoo_classifier_row_boost', 4.0))
            masked[classifier_mask] *= boost

        if int(getattr(self.args, 'clean_label_adv', 0)) == 1 or str(getattr(self.args, 'cuckoo_variant', '')).lower() == 'clean_label_egg':
            masked *= float(getattr(self.args, 'cuckoo_pgd_bias', 1.25))

        lam = self._lambda_t(cur_round)
        final_update = clean_update + lam * masked
        final_update, capped = self._apply_norm_cap(clean_update, final_update)
        final_update = self._apply_min_cosine(clean_update, final_update)

        cos = F.cosine_similarity(clean_update.float(), final_update.float(), dim=0).item()
        stats = CuckooStats(
            agent_id=self.agent_id,
            round=cur_round,
            variant=self.variant,
            stage=self._stage(cur_round),
            lambda_t=lam,
            clean_l2=torch.norm(clean_update, p=2).item(),
            poison_l2=torch.norm(poison_update, p=2).item(),
            residual_l2_before=residual_before_l2,
            residual_l2_after=torch.norm(masked, p=2).item(),
            final_l2=torch.norm(final_update, p=2).item(),
            cosine_clean_final=cos,
            kept_frac=float(mask.float().mean().item()),
            norm_cap_active=capped,
            sign_align_active=bool(int(getattr(self.args, 'cuckoo_sign_align', 1)) == 1),
        )
        return final_update.detach(), stats
