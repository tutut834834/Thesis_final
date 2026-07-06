"""
cuckoo_v1_biology_nest.py

Cuckoo Version 1: Biology Nest Egg
Research-only federated-learning thesis module.

Core idea:
    clean_update  = host/nest update h_t
    poison_update = poisoned/egg update p_t
    egg_residual  = e_t = p_t - h_t
    cuckoo_update = h_t + lambda_t * budget(mask(e_t))

This version uses:
    - hatch schedule
    - top coordinate egg selection
    - classifier/base/target row emphasis
    - sign-alignment shell
    - norm budget shell
    - cosine repair toward the clean host
    - residual memory
"""

import math
import torch


def _safe_norm(x):
    return torch.norm(x.detach().float().view(-1), p=2).item()


def _cosine(a, b):
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    denom = torch.norm(a, p=2) * torch.norm(b, p=2)
    if denom.item() <= 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def _sign_agreement(a, b, mask=None):
    a = a.detach().view(-1)
    b = b.detach().view(-1)
    if mask is None:
        mask = torch.ones_like(a, dtype=torch.bool)
    else:
        mask = mask.bool().view(-1)
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((torch.sign(a[mask]) == torch.sign(b[mask])).float().mean().item())


class CuckooV1Stats:
    def __init__(self, rows):
        self.rows = rows

    def print_block(self):
        print("CUCKOO_V1_BIOLOGY_NEST_START")
        for k, v in self.rows.items():
            print(f"{k}={v}")
        print("CUCKOO_V1_BIOLOGY_NEST_PROOF: clean host/nest update is used; poisoned egg residual is hidden under norm, sign, and cosine shells.")
        print("CUCKOO_V1_BIOLOGY_NEST_END")


class CuckooV1BiologyNestEngine:
    def __init__(self, args, global_model, agent_id=0):
        self.args = args
        self.agent_id = agent_id
        self.memory = None
        self.host_ema = None
        self.param_slices = []
        start = 0
        for name, p in global_model.named_parameters():
            n = p.numel()
            self.param_slices.append((name, start, start + n, tuple(p.shape)))
            start += n
        self.n_params = start
        self._build_masks()

    def _build_masks(self):
        self.classifier_mask = torch.zeros(self.n_params, dtype=torch.bool)
        self.target_row_mask = torch.zeros(self.n_params, dtype=torch.bool)
        self.base_row_mask = torch.zeros(self.n_params, dtype=torch.bool)
        self.bias_mask = torch.zeros(self.n_params, dtype=torch.bool)
        base_class = int(getattr(self.args, "base_class", 5))
        target_class = int(getattr(self.args, "target_class", 7))
        for name, s, e, shape in self.param_slices:
            lower = name.lower()
            is_weight = lower.endswith("weight") and len(shape) == 2 and shape[0] >= 10 and ("fc2" in lower or "fc3" in lower or "classifier" in lower)
            is_bias = lower.endswith("bias") and len(shape) == 1 and shape[0] >= 10 and ("fc2" in lower or "fc3" in lower or "classifier" in lower)
            if is_weight:
                self.classifier_mask[s:e] = True
                row_w = shape[1]
                if 0 <= target_class < shape[0]:
                    self.target_row_mask[s + target_class * row_w:s + (target_class + 1) * row_w] = True
                if 0 <= base_class < shape[0]:
                    self.base_row_mask[s + base_class * row_w:s + (base_class + 1) * row_w] = True
            if is_bias:
                self.classifier_mask[s:e] = True
                self.bias_mask[s:e] = True
                if 0 <= target_class < shape[0]:
                    self.target_row_mask[s + target_class] = True
                if 0 <= base_class < shape[0]:
                    self.base_row_mask[s + base_class] = True

    def _schedule(self, cur_round):
        hatch = int(getattr(self.args, "cuckoo_v1_hatch_round", 8))
        warm = max(1, int(getattr(self.args, "cuckoo_v1_warmup_rounds", 20)))
        lam_max = float(getattr(self.args, "cuckoo_v1_lambda", 1.2))
        lam_inc = float(getattr(self.args, "cuckoo_v1_incubation_lambda", 0.0))
        schedule = str(getattr(self.args, "cuckoo_v1_schedule", "cosine")).lower()
        if cur_round < hatch:
            return lam_inc
        x = min(1.0, max(0.0, (cur_round - hatch + 1) / float(warm)))
        if schedule == "linear":
            s = x
        elif schedule == "sqrt":
            s = math.sqrt(x)
        elif schedule == "step":
            s = 1.0
        else:
            s = 0.5 - 0.5 * math.cos(math.pi * x)
        return lam_inc + (lam_max - lam_inc) * s

    def _topk_mask(self, score, frac):
        score = score.detach().float().view(-1)
        frac = float(frac)
        if frac <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        if frac >= 1:
            return torch.ones_like(score, dtype=torch.bool)
        k = max(1, int(frac * score.numel()))
        _, idx = torch.topk(score, k=k, largest=True)
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask[idx] = True
        return mask

    def build(self, clean_update, poison_update, cur_round):
        host = clean_update.detach().double().view(-1)
        poison = poison_update.detach().double().view(-1)

        ema = float(getattr(self.args, "cuckoo_v1_host_ema", 0.0))
        if ema > 0:
            if self.host_ema is None:
                self.host_ema = host.clone()
            else:
                self.host_ema = ema * self.host_ema + (1.0 - ema) * host
            host_used = self.host_ema.clone()
        else:
            host_used = host

        egg_raw = poison - host_used

        use_memory = int(getattr(self.args, "cuckoo_v1_memory", 1)) == 1
        mem_decay = float(getattr(self.args, "cuckoo_v1_memory_decay", 0.85))
        mem_carry = float(getattr(self.args, "cuckoo_v1_memory_carry", 0.20))
        if use_memory:
            if self.memory is None:
                self.memory = torch.zeros_like(egg_raw)
            egg_work = egg_raw + mem_carry * self.memory
        else:
            egg_work = egg_raw

        score = torch.abs(egg_work).float()
        host_abs = torch.abs(host_used).float()
        if host_abs.max().item() > 0:
            score += float(getattr(self.args, "cuckoo_v1_host_mag_bonus", 0.20)) * host_abs / (host_abs.max() + 1e-12)

        sign_same = torch.sign(egg_work) == torch.sign(host_used)
        sign_zero = torch.sign(host_used) == 0
        score += float(getattr(self.args, "cuckoo_v1_sign_bonus", 0.40)) * sign_same.float()

        classifier_boost = float(getattr(self.args, "cuckoo_v1_classifier_boost", 1.5))
        target_boost = float(getattr(self.args, "cuckoo_v1_target_row_boost", 3.0))
        base_boost = float(getattr(self.args, "cuckoo_v1_base_row_boost", 1.8))
        bias_boost = float(getattr(self.args, "cuckoo_v1_bias_boost", 1.25))
        row_focus = str(getattr(self.args, "cuckoo_v1_row_focus", "target_base")).lower()

        score[self.classifier_mask.to(score.device)] *= classifier_boost
        if row_focus in ("target", "target_base", "both"):
            score[self.target_row_mask.to(score.device)] *= target_boost
        if row_focus in ("base", "target_base", "both"):
            score[self.base_row_mask.to(score.device)] *= base_boost
        if int(getattr(self.args, "cuckoo_v1_include_bias", 1)) == 1:
            score[self.bias_mask.to(score.device)] *= bias_boost

        if int(getattr(self.args, "cuckoo_v1_sign_align", 1)) == 1:
            shell = sign_same | sign_zero
        else:
            shell = torch.ones_like(score, dtype=torch.bool)
        relax = float(getattr(self.args, "cuckoo_v1_sign_relax", 0.05))
        if relax > 0:
            disagree = score.clone()
            disagree[shell] = -1
            shell = shell | self._topk_mask(disagree, relax)
        score = score.masked_fill(~shell, -1.0)

        top_frac = float(getattr(self.args, "cuckoo_v1_top_frac", 0.22))
        keep = self._topk_mask(score, top_frac) & (score >= 0)
        egg = torch.zeros_like(egg_work)
        egg[keep] = egg_work[keep]

        if int(getattr(self.args, "cuckoo_v1_center_egg", 1)) == 1 and keep.sum().item() > 0:
            egg[keep] -= egg[keep].mean()

        blend = float(getattr(self.args, "cuckoo_v1_preserve_blend", 0.03))
        hnorm = torch.norm(host_used, p=2)
        enorm = torch.norm(egg, p=2)
        if blend > 0 and hnorm.item() > 1e-12 and enorm.item() > 1e-12:
            egg = egg + blend * enorm * host_used / hnorm

        budget_mult = float(getattr(self.args, "cuckoo_v1_egg_budget_mult", 0.85))
        budget = max(1e-8, budget_mult * hnorm.item())
        enorm = torch.norm(egg, p=2)
        if enorm.item() > budget:
            egg = egg * (budget / (enorm.item() + 1e-12))

        if use_memory:
            unused = egg_work.clone()
            unused[keep] = 0.0
            self.memory = mem_decay * unused.detach()

        lam = self._schedule(cur_round)
        cuckoo = host_used + lam * egg

        norm_cap = float(getattr(self.args, "cuckoo_v1_norm_cap", 1.40))
        norm_cap_applied = False
        if norm_cap > 0 and hnorm.item() > 1e-12:
            cap = norm_cap * hnorm.item()
            cnorm = torch.norm(cuckoo, p=2).item()
            if cnorm > cap:
                cuckoo = cuckoo * (cap / (cnorm + 1e-12))
                norm_cap_applied = True

        min_cos = float(getattr(self.args, "cuckoo_v1_min_cosine", 0.15))
        repair = 0
        cos = _cosine(host_used, cuckoo)
        while min_cos > -1 and cos < min_cos and repair < 8:
            cuckoo = 0.75 * cuckoo + 0.25 * host_used
            cos = _cosine(host_used, cuckoo)
            repair += 1

        stats = CuckooV1Stats({
            "framework": "Cuckoo Version 1 - Biology Nest Egg",
            "agent_id": self.agent_id,
            "round": cur_round,
            "lambda_t": round(float(lam), 6),
            "host_norm": round(_safe_norm(host_used), 6),
            "poison_norm": round(_safe_norm(poison), 6),
            "egg_raw_norm": round(_safe_norm(egg_raw), 6),
            "egg_selected_norm": round(_safe_norm(egg), 6),
            "cuckoo_norm": round(_safe_norm(cuckoo), 6),
            "cosine_host_cuckoo": round(_cosine(host_used, cuckoo), 6),
            "cosine_host_poison": round(_cosine(host_used, poison), 6),
            "sign_agreement_host_cuckoo": round(_sign_agreement(host_used, cuckoo), 6),
            "sign_agreement_host_egg_kept": round(_sign_agreement(host_used, egg, keep), 6),
            "kept_coordinates": int(keep.sum().item()),
            "kept_fraction": round(float(keep.float().mean().item()), 6),
            "classifier_kept_fraction": round(float((keep & self.classifier_mask.to(keep.device)).float().sum().item() / max(1, int(self.classifier_mask.sum().item()))), 6),
            "target_row_kept_fraction": round(float((keep & self.target_row_mask.to(keep.device)).float().sum().item() / max(1, int(self.target_row_mask.sum().item()))), 6),
            "base_row_kept_fraction": round(float((keep & self.base_row_mask.to(keep.device)).float().sum().item() / max(1, int(self.base_row_mask.sum().item()))), 6),
            "norm_cap": norm_cap,
            "norm_cap_applied": norm_cap_applied,
            "min_cosine": min_cos,
            "cosine_repair_steps": repair,
            "memory_enabled": use_memory,
            "memory_norm": round(_safe_norm(self.memory), 6) if self.memory is not None else 0.0,
            "top_frac": top_frac,
            "egg_budget": round(float(budget), 6),
            "row_focus": row_focus,
        })
        return cuckoo.detach(), stats
