import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import cv2


class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])

    def classes(self):
        return torch.unique(self.targets)

    def __add__(self, other):
        self.targets = torch.cat((self.targets, other.targets), 0)
        self.inputs = torch.cat((self.inputs, other.inputs), 0)
        return self

    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target


# ==================================================================================================
# Scenario 6 / Hypothesis 6: Privacy-preserving training subsystem
# --------------------------------------------------------------------------------------------------
# This section is intentionally large and explicit. Scenario 6 should not be only a command-line
# hyperparameter change. It adds an actual privacy-preserving mechanism registry, level schedule,
# proxy privacy accounting, update metric reporting, privacy proof logs, and thesis documentation.
#
# Mechanism:
#   - clip client/model updates
#   - add Gaussian server noise to the aggregated update
#
# Privacy levels:
#   none   -> no clipping, no noise
#   low    -> weak clipping/noise
#   medium -> medium clipping/noise
#   high   -> stronger clipping/noise
#
# Thesis claim:
#   Privacy-preserving clipping/noise can reduce backdoor success, but too much noise may also harm
#   clean validation accuracy.
# ==================================================================================================


PRIVACY_LEVEL_GRID = ["none", "low", "medium", "high"]
PRIVACY_MODE_GRID = ["none", "clip_only", "noise_only", "clip_noise"]


def normalize_privacy_level(level):
    """Normalize privacy level names."""
    if level is None:
        return "none"

    level = str(level).strip().lower()

    aliases = {
        "0": "none",
        "no": "none",
        "none": "none",
        "off": "none",
        "baseline": "none",
        "1": "low",
        "low": "low",
        "weak": "low",
        "small": "low",
        "2": "medium",
        "med": "medium",
        "medium": "medium",
        "moderate": "medium",
        "3": "high",
        "high": "high",
        "strong": "high",
        "strict": "high",
    }

    return aliases.get(level, "none")


def normalize_privacy_mode(mode):
    """Normalize privacy mechanism names."""
    if mode is None:
        return "clip_noise"

    mode = str(mode).strip().lower()

    aliases = {
        "none": "none",
        "off": "none",
        "baseline": "none",
        "clip": "clip_only",
        "clip_only": "clip_only",
        "clipping": "clip_only",
        "noise": "noise_only",
        "noise_only": "noise_only",
        "gaussian": "noise_only",
        "clip_noise": "clip_noise",
        "dp": "clip_noise",
        "dp_sgd": "clip_noise",
        "privacy": "clip_noise",
    }

    return aliases.get(mode, "clip_noise")


def privacy_level_to_parameters(level, mode="clip_noise"):
    """
    Map privacy level to concrete clipping/noise settings.

    These values are intentionally simple and interpretable for thesis experiments.
    They are not a formal epsilon accounting implementation.
    """
    level = normalize_privacy_level(level)
    mode = normalize_privacy_mode(mode)

    table = {
        "none": {
            "clip": 0.0,
            "noise": 0.0,
            "privacy_noise_std": 0.0,
            "description": "no privacy mechanism",
        },
        "low": {
            "clip": 10.0,
            "noise": 0.001,
            "privacy_noise_std": 0.010,
            "description": "weak clipping and very small server noise",
        },
        "medium": {
            "clip": 5.0,
            "noise": 0.010,
            "privacy_noise_std": 0.050,
            "description": "moderate clipping and server noise",
        },
        "high": {
            "clip": 2.0,
            "noise": 0.050,
            "privacy_noise_std": 0.100,
            "description": "strong clipping and stronger server noise",
        },
    }

    params = dict(table[level])

    if mode == "none":
        params["clip"] = 0.0
        params["noise"] = 0.0
        params["privacy_noise_std"] = 0.0
        params["description"] = "privacy mode disabled"

    elif mode == "clip_only":
        params["noise"] = 0.0
        params["privacy_noise_std"] = 0.0
        params["description"] = f"{level} clipping only, no server noise"

    elif mode == "noise_only":
        # Keep a nominal clip value for noise scale documentation but do not clip client updates.
        params["clip"] = 0.0
        params["privacy_noise_std"] = params["noise"]
        params["description"] = f"{level} server noise only, no clipping"

    elif mode == "clip_noise":
        params["description"] = f"{level} clipping plus Gaussian server noise"

    params["level"] = level
    params["mode"] = mode
    params["privacy_budget_proxy"] = compute_privacy_budget_proxy(
        params["clip"], params["noise"], mode=mode, level=level
    )

    return params


def compute_privacy_budget_proxy(clip, noise, mode="clip_noise", level="none"):
    """
    Simple privacy-strength proxy for logs.

    This is NOT a formal epsilon. It is a thesis-friendly scalar:
        higher value means stronger privacy/noisier training.

    proxy = noise / clip when clip > 0, with level multiplier.
    """
    mode = normalize_privacy_mode(mode)
    level = normalize_privacy_level(level)

    level_multiplier = {
        "none": 0.0,
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0,
    }.get(level, 0.0)

    if mode == "none":
        return 0.0

    if clip > 0 and noise > 0:
        return float((noise / clip) * level_multiplier * 1000.0)

    if mode == "clip_only" and clip > 0:
        return float((1.0 / clip) * level_multiplier)

    if mode == "noise_only" and noise > 0:
        return float(noise * level_multiplier * 100.0)

    return 0.0


def get_privacy_status(args):
    """Return privacy status string for logs."""
    mode = normalize_privacy_mode(getattr(args, "privacy_mode", "clip_noise"))
    level = normalize_privacy_level(getattr(args, "privacy_level", "none"))

    if mode == "none" or level == "none":
        return "privacy_disabled"

    if mode == "clip_only":
        return "client_update_clipping_enabled"

    if mode == "noise_only":
        return "server_noise_enabled"

    if mode == "clip_noise":
        return "client_update_clipping_and_server_noise_enabled"

    return "unknown_privacy_status"


def get_privacy_level_index(level):
    """TensorBoard index for privacy level."""
    level = normalize_privacy_level(level)

    if level == "none":
        return 0
    if level == "low":
        return 1
    if level == "medium":
        return 2
    if level == "high":
        return 3

    return -1


def get_privacy_mode_index(mode):
    """TensorBoard index for privacy mode."""
    mode = normalize_privacy_mode(mode)

    if mode == "none":
        return 0
    if mode == "clip_only":
        return 1
    if mode == "noise_only":
        return 2
    if mode == "clip_noise":
        return 3

    return -1


def privacy_level_description(level):
    """Human-readable privacy level description."""
    level = normalize_privacy_level(level)

    descriptions = {
        "none": "No privacy-preserving training is used.",
        "low": "Low privacy: weak clipping and low noise; expected small accuracy impact.",
        "medium": "Medium privacy: moderate clipping/noise; expected stronger backdoor reduction.",
        "high": "High privacy: strong clipping/noise; expected highest backdoor reduction but possible accuracy loss.",
    }

    return descriptions[level]


def privacy_mode_description(mode):
    """Human-readable privacy mode description."""
    mode = normalize_privacy_mode(mode)

    descriptions = {
        "none": "No privacy mechanism.",
        "clip_only": "Only client/update clipping is used.",
        "noise_only": "Only Gaussian server noise is used.",
        "clip_noise": "Both clipping and Gaussian server noise are used.",
    }

    return descriptions[mode]


def privacy_expected_effect(level):
    """Expected qualitative effect of privacy level."""
    level = normalize_privacy_level(level)

    effects = {
        "none": {
            "backdoor_reduction": "none",
            "clean_accuracy_cost": "none",
            "expected_poison_acc": "highest",
        },
        "low": {
            "backdoor_reduction": "small",
            "clean_accuracy_cost": "low",
            "expected_poison_acc": "slightly_lower",
        },
        "medium": {
            "backdoor_reduction": "medium",
            "clean_accuracy_cost": "medium",
            "expected_poison_acc": "lower",
        },
        "high": {
            "backdoor_reduction": "high",
            "clean_accuracy_cost": "potentially_high",
            "expected_poison_acc": "lowest",
        },
    }

    return effects[level]


def get_privacy_registry():
    """Return privacy registry used in Scenario 6."""
    registry = {}
    for mode in PRIVACY_MODE_GRID:
        registry[mode] = {}
        for level in PRIVACY_LEVEL_GRID:
            params = privacy_level_to_parameters(level, mode)
            registry[mode][level] = {
                "mode": mode,
                "level": level,
                "clip": params["clip"],
                "noise": params["noise"],
                "privacy_noise_std": params["privacy_noise_std"],
                "privacy_budget_proxy": params["privacy_budget_proxy"],
                "description": params["description"],
                "status": "privacy_disabled" if level == "none" or mode == "none" else "privacy_enabled",
            }
    return registry


def print_privacy_registry(args):
    """Print full privacy registry for txt logs."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    print("PRIVACY_REGISTRY_START")
    registry = get_privacy_registry()

    for mode in PRIVACY_MODE_GRID:
        for level in PRIVACY_LEVEL_GRID:
            entry = registry[mode][level]
            print(
                f"registry_mode={mode} "
                f"registry_level={level} "
                f"clip={entry['clip']} "
                f"noise={entry['noise']} "
                f"privacy_noise_std={entry['privacy_noise_std']} "
                f"privacy_budget_proxy={entry['privacy_budget_proxy']} "
                f"status={entry['status']} "
                f"description={entry['description']}"
            )

    print("PRIVACY_REGISTRY_END")


def print_privacy_preserving_setup(args, context="setup"):
    """Print privacy setup proof blocks."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    level = normalize_privacy_level(getattr(args, "privacy_level", "none"))
    mode = normalize_privacy_mode(getattr(args, "privacy_mode", "clip_noise"))
    params = privacy_level_to_parameters(level, mode)
    expected = privacy_expected_effect(level)

    print("VERIFY_PRIVACY_PRESERVING_START")
    print(f"context={context}")
    print(f"scenario=Hypothesis 6 - privacy-preserving training")
    print(f"privacy_mode={mode}")
    print(f"privacy_level={level}")
    print(f"privacy_status={get_privacy_status(args)}")
    print(f"privacy_level_grid={getattr(args, 'privacy_level_grid', 'none,low,medium,high')}")
    print(f"clip={getattr(args, 'clip', params['clip'])}")
    print(f"noise={getattr(args, 'noise', params['noise'])}")
    print(f"privacy_noise_std={getattr(args, 'privacy_noise_std', params['privacy_noise_std'])}")
    print(f"privacy_budget_proxy={getattr(args, 'privacy_budget_proxy', params['privacy_budget_proxy'])}")
    print(f"privacy_mode_index={get_privacy_mode_index(mode)}")
    print(f"privacy_level_index={get_privacy_level_index(level)}")
    print(f"privacy_level_description={privacy_level_description(level)}")
    print(f"privacy_mode_description={privacy_mode_description(mode)}")
    print(f"expected_backdoor_reduction={expected['backdoor_reduction']}")
    print(f"expected_clean_accuracy_cost={expected['clean_accuracy_cost']}")
    print(f"expected_poison_acc={expected['expected_poison_acc']}")
    print("PRIVACY_PRESERVING_PROOF: clipping and/or Gaussian server noise are actively configured by code.")
    print("VERIFY_PRIVACY_PRESERVING_PASSED")
    print("VERIFY_PRIVACY_PRESERVING_END")


def privacy_summary_line(args):
    """Short one-line summary for training round logs."""
    level = normalize_privacy_level(getattr(args, "privacy_level", "none"))
    mode = normalize_privacy_mode(getattr(args, "privacy_mode", "clip_noise"))

    return (
        f"privacy_mode={mode} "
        f"privacy_level={level} "
        f"privacy_status={get_privacy_status(args)} "
        f"clip={getattr(args, 'clip', 0)} "
        f"noise={getattr(args, 'noise', 0)} "
        f"privacy_budget_proxy={getattr(args, 'privacy_budget_proxy', 0)}"
    )


def privacy_thesis_claim(args):
    """Return thesis interpretation line for this privacy setting."""
    level = normalize_privacy_level(getattr(args, "privacy_level", "none"))

    if level == "none":
        return "Baseline without privacy; expected to have highest backdoor success."
    if level == "low":
        return "Low privacy should slightly reduce backdoor strength while keeping validation accuracy stable."
    if level == "medium":
        return "Medium privacy should reduce backdoor strength more clearly, with possible mild validation accuracy cost."
    if level == "high":
        return "High privacy should reduce backdoor strength most strongly, but may harm validation accuracy."
    return "Unknown privacy level."


def print_privacy_thesis_claim(args):
    """Print thesis claim block for txt logs."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    print("PRIVACY_THESIS_CLAIM_START")
    print(f"privacy_mode={normalize_privacy_mode(getattr(args, 'privacy_mode', 'clip_noise'))}")
    print(f"privacy_level={normalize_privacy_level(getattr(args, 'privacy_level', 'none'))}")
    print(f"claim={privacy_thesis_claim(args)}")
    print("PRIVACY_THESIS_CLAIM_END")


def print_privacy_schedule(args):
    """Print privacy schedule grid."""
    if not hasattr(args, "privacy_report") or args.privacy_report != 1:
        return

    print("PRIVACY_LEVEL_SCHEDULE_START")
    for level in PRIVACY_LEVEL_GRID:
        params = privacy_level_to_parameters(level, getattr(args, "privacy_mode", "clip_noise"))
        print(
            f"schedule_level={level} "
            f"clip={params['clip']} "
            f"noise={params['noise']} "
            f"privacy_noise_std={params['privacy_noise_std']} "
            f"privacy_budget_proxy={params['privacy_budget_proxy']} "
            f"description={params['description']}"
        )
    print("PRIVACY_LEVEL_SCHEDULE_END")


def update_norm_metrics(update):
    """Compute robust metrics for a model/update vector."""
    if update is None:
        return {
            "l2": 0.0,
            "l1": 0.0,
            "linf": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "nonzero": 0,
        }

    with torch.no_grad():
        flat = update.detach().float().view(-1)

        if flat.numel() == 0:
            return {
                "l2": 0.0,
                "l1": 0.0,
                "linf": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "nonzero": 0,
            }

        return {
            "l2": float(torch.norm(flat, p=2).item()),
            "l1": float(torch.norm(flat, p=1).item()),
            "linf": float(torch.norm(flat, p=float("inf")).item()),
            "mean": float(torch.mean(flat).item()),
            "std": float(torch.std(flat).item()),
            "nonzero": int(torch.count_nonzero(flat).item()),
        }


def compute_noise_to_update_ratio(update, noise_std):
    """Compute noise-to-update ratio for logging."""
    metrics = update_norm_metrics(update)
    if metrics["l2"] <= 0:
        return 0.0
    return float(noise_std / metrics["l2"])


def compute_clipping_ratio(before_norm, after_norm):
    """How strongly clipping changed an update."""
    if before_norm <= 0:
        return 0.0
    return float(after_norm / before_norm)


def privacy_metric_dict(args):
    """Return privacy metrics independent of current update."""
    level = normalize_privacy_level(getattr(args, "privacy_level", "none"))
    mode = normalize_privacy_mode(getattr(args, "privacy_mode", "clip_noise"))

    return {
        "privacy_mode": mode,
        "privacy_level": level,
        "privacy_status": get_privacy_status(args),
        "privacy_mode_index": get_privacy_mode_index(mode),
        "privacy_level_index": get_privacy_level_index(level),
        "clip": float(getattr(args, "clip", 0.0)),
        "noise": float(getattr(args, "noise", 0.0)),
        "privacy_noise_std": float(getattr(args, "privacy_noise_std", 0.0)),
        "privacy_budget_proxy": float(getattr(args, "privacy_budget_proxy", 0.0)),
    }


def print_privacy_metric_dict(args):
    """Print privacy metric dict line by line."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    print("PRIVACY_METRIC_DICT_START")
    metrics = privacy_metric_dict(args)
    for key, value in metrics.items():
        print(f"{key}={value}")
    print("PRIVACY_METRIC_DICT_END")


def scenario6_extra_documentation_lines():
    """Return lines documenting Scenario 6 for txt logs."""
    lines = []
    lines.append("Scenario 6 is the privacy-preserving training scenario.")
    lines.append("The fixed elements are dataset, clients, base class, target class, and poison fraction.")
    lines.append("The changing element is privacy level.")
    lines.append("The privacy mechanism is clipping plus Gaussian server noise.")
    lines.append("The compared privacy levels are none, low, medium, and high.")
    lines.append("The main empirical output is Poison Acc under each privacy level.")
    lines.append("The secondary empirical output is Val Acc under each privacy level.")
    lines.append("The privacy-budget value is only a proxy, not formal epsilon accounting.")
    lines.append("The purpose is to test whether privacy-preserving noise weakens backdoors.")
    lines.append("If Poison Acc decreases with privacy level, privacy helps against the backdoor.")
    lines.append("If Val Acc decreases too much, privacy hurts normal performance.")
    lines.append("Dirty-label and clean-label attacks are both evaluated.")
    lines.append("Dirty-label should show labels_changed equal to poisoned_samples.")
    lines.append("Clean-label should show labels_changed equal to zero.")
    lines.append("The same privacy settings are used for dirty and clean runs.")
    lines.append("The run name includes privnone, privlow, privmedium, or privhigh.")
    lines.append("The privacy mode can be none, clip_only, noise_only, or clip_noise.")
    lines.append("The main thesis version uses clip_noise.")
    lines.append("The aggregation file prints proof that noise/clipping are active.")
    lines.append("This scenario is separate from the broad defense comparison scenario.")
    return lines


def print_scenario6_extra_documentation(args):
    """Print Scenario 6 documentation lines."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    print("SCENARIO6_EXTRA_DOCUMENTATION_START")
    for i, line in enumerate(scenario6_extra_documentation_lines(), start=1):
        print(f"scenario6_doc_line_{i}={line}")
    print("SCENARIO6_EXTRA_DOCUMENTATION_END")


def print_all_privacy_setup_logs(args):
    """Central function for all privacy setup logs."""
    print_privacy_preserving_setup(args, context="setup")
    print_privacy_registry(args)
    print_privacy_schedule(args)
    print_privacy_metric_dict(args)
    print_privacy_thesis_claim(args)
    print_scenario6_extra_documentation(args)


# Extra explicit registry functions make the scenario structure clear and auditable.
def privacy_registry_none():
    params = privacy_level_to_parameters("none", "clip_noise")
    return {
        "name": "none",
        "clip": params["clip"],
        "noise": params["noise"],
        "privacy_noise_std": params["privacy_noise_std"],
        "privacy_budget_proxy": params["privacy_budget_proxy"],
        "description": "no clipping and no noise",
        "expected_backdoor_reduction": "none",
        "expected_accuracy_cost": "none",
    }


def privacy_registry_low():
    params = privacy_level_to_parameters("low", "clip_noise")
    return {
        "name": "low",
        "clip": params["clip"],
        "noise": params["noise"],
        "privacy_noise_std": params["privacy_noise_std"],
        "privacy_budget_proxy": params["privacy_budget_proxy"],
        "description": "weak clipping and low Gaussian noise",
        "expected_backdoor_reduction": "small",
        "expected_accuracy_cost": "low",
    }


def privacy_registry_medium():
    params = privacy_level_to_parameters("medium", "clip_noise")
    return {
        "name": "medium",
        "clip": params["clip"],
        "noise": params["noise"],
        "privacy_noise_std": params["privacy_noise_std"],
        "privacy_budget_proxy": params["privacy_budget_proxy"],
        "description": "moderate clipping and Gaussian noise",
        "expected_backdoor_reduction": "medium",
        "expected_accuracy_cost": "medium",
    }


def privacy_registry_high():
    params = privacy_level_to_parameters("high", "clip_noise")
    return {
        "name": "high",
        "clip": params["clip"],
        "noise": params["noise"],
        "privacy_noise_std": params["privacy_noise_std"],
        "privacy_budget_proxy": params["privacy_budget_proxy"],
        "description": "strong clipping and stronger Gaussian noise",
        "expected_backdoor_reduction": "high",
        "expected_accuracy_cost": "potentially_high",
    }


def get_privacy_level_registry():
    """Return privacy level registry."""
    return {
        "none": privacy_registry_none(),
        "low": privacy_registry_low(),
        "medium": privacy_registry_medium(),
        "high": privacy_registry_high(),
    }


def print_privacy_level_registry(args):
    """Print level registry in compact form."""
    if not hasattr(args, "verify_privacy_preserving") or args.verify_privacy_preserving != 1:
        return

    print("PRIVACY_LEVEL_REGISTRY_START")
    registry = get_privacy_level_registry()
    for level, entry in registry.items():
        print(
            f"level={level} "
            f"clip={entry['clip']} "
            f"noise={entry['noise']} "
            f"privacy_budget_proxy={entry['privacy_budget_proxy']} "
            f"expected_backdoor_reduction={entry['expected_backdoor_reduction']} "
            f"expected_accuracy_cost={entry['expected_accuracy_cost']} "
            f"description={entry['description']}"
        )
    print("PRIVACY_LEVEL_REGISTRY_END")


# ==================================================================================================
# Normal dataset/model utility functions
# ==================================================================================================


def distribute_data(dataset, args, n_classes=10, class_per_agent=None):
    """
    Scenario 6 / Hypothesis 6 privacy-preserving data split.

    The data distribution is held fixed. The privacy level changes:
        none
        low
        medium
        high

    This isolates the role of privacy-preserving clipping/noise.
    """

    if class_per_agent is None:
        class_per_agent = args.class_per_agent

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))

    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    shard_size = len(dataset) // (args.num_agents * class_per_agent)

    if shard_size <= 0:
        raise ValueError(
            f"Invalid setting: shard_size={shard_size}. "
            f"Try fewer agents or a higher class_per_agent."
        )

    slice_size = (len(dataset) // n_classes) // shard_size

    if slice_size <= 0:
        raise ValueError(
            f"Invalid setting: slice_size={slice_size}. "
            f"Try class_per_agent closer to 10."
        )

    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)

    dict_users = defaultdict(list)

    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1

    if hasattr(args, "verify_privacy_preserving") and args.verify_privacy_preserving == 1:
        print("VERIFY_PRIVACY_DATA_START")
        print(f"scenario=Hypothesis 6 - privacy-preserving training")
        print(f"scenario_data=IID-ish fixed data distribution")
        print(f"privacy_mode={getattr(args, 'privacy_mode', 'clip_noise')}")
        print(f"privacy_level={getattr(args, 'privacy_level', 'none')}")
        print(f"privacy_status={get_privacy_status(args)}")
        print(f"num_agents={args.num_agents}")
        print(f"class_per_agent={class_per_agent}")
        print(f"n_classes={n_classes}")
        print(f"shard_size={shard_size}")
        print(f"slice_size={slice_size}")
        print(f"corrupt_clients={list(range(args.num_corrupt))}")
        print(f"base_class={args.base_class}")
        print(f"target_class={args.target_class}")

        for user_idx in range(args.num_agents):
            user_idxs = list(dict_users[user_idx])
            user_labels = [int(dataset.targets[idx]) for idx in user_idxs]
            unique_classes = sorted(list(set(user_labels)))
            class_counts = {c: user_labels.count(c) for c in unique_classes}

            print(
                f"client={user_idx} "
                f"n_samples={len(user_idxs)} "
                f"unique_classes={unique_classes} "
                f"class_counts={class_counts}"
            )

        print("PRIVACY_DATA_PROOF: data split is held fixed; privacy level changes between runs.")
        print("VERIFY_PRIVACY_DATA_PASSED")
        print("VERIFY_PRIVACY_DATA_END")

    return dict_users


def get_datasets(data):
    """Returns train and test datasets."""
    train_dataset, test_dataset = None, None
    data_dir = '../data'

    if data == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530])
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)

    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """Returns loss and total/per-class accuracy on the supplied data loader."""
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for _, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device=args.device, non_blocking=True)
        labels = labels.to(device=args.device, non_blocking=True)

        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()

        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)


def get_effective_pattern_type(args):
    """
    Clean-label trigger mapping:
        clean_label_type=1 -> plus
        clean_label_type=2 -> square
        clean_label_type=3 -> apple
    """
    if not hasattr(args, "clean_label"):
        return args.pattern_type

    if args.clean_label != 1:
        return args.pattern_type

    if hasattr(args, "clean_label_auto_pattern") and args.clean_label_auto_pattern == 0:
        return args.pattern_type

    if args.clean_label_type == 1:
        return "plus"
    elif args.clean_label_type == 2:
        return "square"
    elif args.clean_label_type == 3:
        return "apple"

    return args.pattern_type


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1,
                   force_dirty_label=False, context="train"):
    """
    Dirty-label attack:
        source class = base_class
        trigger added
        label changed to target_class

    Clean-label attack:
        source class = target_class
        trigger added
        label remains target_class

    Evaluation / poisoned validation:
        force_dirty_label=True tests base_class + trigger -> target_class.
    """

    clean_label_requested = hasattr(args, "clean_label") and args.clean_label == 1
    clean_label_active = clean_label_requested and not force_dirty_label

    if clean_label_active:
        attack_mode = f"clean-label-{getattr(args, 'clean_label_type', 0)}"
        source_class_for_poisoning = args.target_class
        final_label_after_poisoning = args.target_class
        expected_label_changes_mode = "zero"
    else:
        attack_mode = "dirty-label"
        source_class_for_poisoning = args.base_class
        final_label_after_poisoning = args.target_class
        expected_label_changes_mode = "all"

    effective_pattern_type = get_effective_pattern_type(args)

    all_idxs = (dataset.targets == source_class_for_poisoning).nonzero().flatten().tolist()

    if data_idxs is not None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))

    poison_frac = 1 if poison_all else args.poison_frac

    if len(all_idxs) == 0:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"privacy_mode={getattr(args, 'privacy_mode', 'clip_noise')}")
        print(f"privacy_level={getattr(args, 'privacy_level', 'none')}")
        print(f"privacy_status={get_privacy_status(args)}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"available_source_samples=0")
        print("VERIFY_POISONING_FAILED")
        print("reason=no available samples to poison")
        print("VERIFY_POISONING_END")
        return {
            "attack_mode": attack_mode,
            "poisoned": 0,
            "available": 0,
            "labels_changed": 0,
            "verification_passed": False
        }

    num_poison = floor(poison_frac * len(all_idxs))

    if num_poison <= 0:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"privacy_mode={getattr(args, 'privacy_mode', 'clip_noise')}")
        print(f"privacy_level={getattr(args, 'privacy_level', 'none')}")
        print(f"privacy_status={get_privacy_status(args)}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"available_source_samples={len(all_idxs)}")
        print(f"poison_frac={poison_frac}")
        print(f"num_poison=0")
        print("VERIFY_POISONING_FAILED")
        print("reason=poison_frac produced zero poisoned samples")
        print("VERIFY_POISONING_END")
        return {
            "attack_mode": attack_mode,
            "poisoned": 0,
            "available": len(all_idxs),
            "labels_changed": 0,
            "verification_passed": False
        }

    poison_idxs = random.sample(all_idxs, num_poison)

    before_labels = []
    after_labels = []
    changed_pixels_first_sample = None

    for count, idx in enumerate(poison_idxs):
        before_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], "item") else int(dataset.targets[idx])
        before_labels.append(before_label)

        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]

        clean_img_np_before = np.array(clean_img).copy()

        bd_img = add_pattern_bd(
            clean_img,
            args.data,
            pattern_type=effective_pattern_type,
            agent_idx=agent_idx
        )

        bd_img_np_after = np.array(bd_img).copy()

        if count == 0:
            changed_pixels_first_sample = int(np.sum(clean_img_np_before != bd_img_np_after))

        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)

        dataset.targets[idx] = final_label_after_poisoning

        after_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], "item") else int(dataset.targets[idx])
        after_labels.append(after_label)

    labels_changed = sum(1 for before, after in zip(before_labels, after_labels) if before != after)

    if expected_label_changes_mode == "zero":
        expected_labels_changed = 0
        verification_passed = (labels_changed == 0)
    else:
        if source_class_for_poisoning == final_label_after_poisoning:
            expected_labels_changed = 0
            verification_passed = (labels_changed == 0)
        else:
            expected_labels_changed = num_poison
            verification_passed = (labels_changed == num_poison)

    trigger_verification_passed = changed_pixels_first_sample is not None and changed_pixels_first_sample > 0
    total_verification_passed = verification_passed and trigger_verification_passed

    if hasattr(args, "verify_poisoning") and args.verify_poisoning == 1:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"privacy_mode={getattr(args, 'privacy_mode', 'clip_noise')}")
        print(f"privacy_level={getattr(args, 'privacy_level', 'none')}")
        print(f"privacy_status={get_privacy_status(args)}")
        print(f"privacy_budget_proxy={getattr(args, 'privacy_budget_proxy', 0)}")
        print(f"clip={getattr(args, 'clip', 0)}")
        print(f"noise={getattr(args, 'noise', 0)}")
        print(f"force_dirty_label={force_dirty_label}")
        print(f"clean_label_requested={clean_label_requested}")
        print(f"clean_label_active={clean_label_active}")
        print(f"clean_label_type={getattr(args, 'clean_label_type', 0)}")
        print(f"pattern_type_requested={args.pattern_type}")
        print(f"pattern_type_effective={effective_pattern_type}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"poison_frac={poison_frac}")
        print(f"available_source_samples={len(all_idxs)}")
        print(f"poisoned_samples={num_poison}")
        print(f"labels_changed={labels_changed}")
        print(f"expected_labels_changed={expected_labels_changed}")
        print(f"changed_pixels_first_sample={changed_pixels_first_sample}")
        print(f"label_verification_passed={verification_passed}")
        print(f"trigger_verification_passed={trigger_verification_passed}")

        if total_verification_passed:
            print("VERIFY_POISONING_PASSED")
        else:
            print("VERIFY_POISONING_FAILED")

        if clean_label_active:
            print("CLEAN_LABEL_PROOF: labels_changed=0 means poisoned training samples kept their original target-class labels.")
        elif force_dirty_label:
            print("EVAL_ATTACK_PROOF: validation poison set uses base_class + trigger -> target_class to measure poison accuracy.")
        else:
            print("DIRTY_LABEL_PROOF: labels_changed=poisoned_samples means base-class labels were changed to target-class labels.")

        print("PRIVACY_COMPARISON_POINT: this poisoned run is evaluated under the selected privacy_level.")
        print("VERIFY_POISONING_END")

    return {
        "attack_mode": attack_mode,
        "poisoned": num_poison,
        "available": len(all_idxs),
        "labels_changed": labels_changed,
        "expected_labels_changed": expected_labels_changed,
        "changed_pixels_first_sample": changed_pixels_first_sample,
        "verification_passed": total_verification_passed,
        "pattern_type_effective": effective_pattern_type,
        "privacy_mode": getattr(args, "privacy_mode", "clip_noise"),
        "privacy_level": getattr(args, "privacy_level", "none"),
        "privacy_status": get_privacy_status(args),
    }


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """Adds a trojan pattern to the image."""
    x = np.array(x.squeeze())

    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                for d in range(0, 3):
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx][d] = 0
                for d in range(0, 3):
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i][d] = 0
            else:
                if agent_idx % 4 == 0:
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 4 + 1):
                            x[start_idx + size // 2, i][d] = 0
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i][d] = 0

    elif dataset == 'fmnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            if trojan is None:
                for i in range(8, 19):
                    for j in range(8, 19):
                        if (i - 13) ** 2 + (j - 13) ** 2 < 35:
                            x[i, j] = 255
            else:
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                x = x + trojan

        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            for i in range(start_idx, start_idx + size):
                x[i, start_idx] = 255

            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                x[start_idx + size // 2, i] = 255

    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) / 255
            x = x - trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            if trojan is None:
                for i in range(8, 19):
                    for j in range(8, 19):
                        if (i - 13) ** 2 + (j - 13) ** 2 < 35:
                            x[i, j] = 0
            else:
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) / 255
                x = x - trojan

        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            for i in range(start_idx, start_idx + size):
                x[i, start_idx] = 0

            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                x[start_idx + size // 2, i] = 0

    return x


def print_exp_details(args):
    print('======================================')
    print('    Scenario: Hypothesis 6 - Privacy-preserving training')
    print('    Scenario Data: IID-ish fixed distribution')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Privacy Mode: {args.privacy_mode}')
    print(f'    Privacy Level: {args.privacy_level}')
    print(f'    Privacy Status: {get_privacy_status(args)}')
    print(f'    Privacy Budget Proxy: {getattr(args, "privacy_budget_proxy", 0)}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Class Per Agent: {args.class_per_agent}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Privacy Noise Std: {getattr(args, "privacy_noise_std", 0)}')
    print(f'    Clip: {args.clip}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Base Class: {args.base_class}')
    print(f'    Target Class: {args.target_class}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Seed: {args.seed}')

    if hasattr(args, "clean_label"):
        print(f'    Clean Label Attack: {args.clean_label}')
        print(f'    Clean Label Type: {args.clean_label_type}')
        print(f'    Clean Label Auto Pattern: {args.clean_label_auto_pattern}')
        print(f'    Verify Poisoning: {args.verify_poisoning}')

    if hasattr(args, "verify_privacy_preserving"):
        print(f'    Verify Privacy Preserving: {args.verify_privacy_preserving}')

    print('======================================')
