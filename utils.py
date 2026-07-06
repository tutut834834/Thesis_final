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
    def classes(self): return torch.unique(self.targets)
    def __len__(self): return self.targets.shape[0]
    def __getitem__(self, item): return self.inputs[item], self.targets[item]


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.LongTensor([int(self.dataset.targets[idx]) for idx in self.idxs])
    def classes(self): return torch.unique(self.targets)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item): return self.dataset[self.idxs[item]]


def distribute_data(dataset, args, n_classes=10):
    class_per_agent = int(getattr(args, 'class_per_agent', 10)) if hasattr(args, 'class_per_agent') else 10
    if args.num_agents == 1:
        return {0: range(len(dataset))}
    def chunker_list(seq, size): return [seq[i::size] for i in range(size)]
    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[int(k)].append(v)
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    users = defaultdict(list)
    for user_idx in range(args.num_agents):
        ctr = 0
        for j in range(n_classes):
            if ctr == class_per_agent:
                break
            if len(labels_dict[j]) > 0:
                users[user_idx] += labels_dict[j][0]
                del labels_dict[j][0]
                ctr += 1
    return users


def get_datasets(data):
    data_dir = '../data'
    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        return datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform), datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if data == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
        tr = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        te = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        tr.targets, te.targets = torch.LongTensor(tr.targets), torch.LongTensor(te.targets)
        return tr, te
    if data == 'fedemnist':
        return torch.load('../data/Fed_EMNIST/fed_emnist_all_trainset.pt'), torch.load('../data/Fed_EMNIST/fed_emnist_all_valset.pt')
    raise ValueError(data)


def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    model.eval()
    total_loss, correct = 0, 0
    confusion = torch.zeros(num_classes, num_classes)
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * outputs.shape[0]
        pred = torch.max(outputs, 1)[1].view(-1)
        correct += torch.sum(pred.eq(labels)).item()
        for t, p in zip(labels.view(-1), pred.view(-1)):
            confusion[t.long(), p.long()] += 1
    per_class = confusion.diag() / confusion.sum(1).clamp(min=1)
    return total_loss / len(data_loader.dataset), (correct / len(data_loader.dataset), per_class)


FMNIST_MEAN = 0.2860
FMNIST_STD = 0.3530


def get_effective_pattern_type(args):
    if getattr(args, 'clean_label', 0) != 1:
        return args.pattern_type
    if getattr(args, 'clean_label_auto_pattern', 1) == 0:
        return args.pattern_type
    return {1: 'big_plus', 2: 'big_square', 3: 'apple', 4: 'pgd_big_square'}.get(getattr(args, 'clean_label_type', 0), args.pattern_type)


def _clip_normalized_fmnist(x):
    lo = (0.0 - FMNIST_MEAN) / FMNIST_STD
    hi = (1.0 - FMNIST_MEAN) / FMNIST_STD
    return torch.clamp(x, lo, hi)


def add_pattern_tensor_bd(inputs, args, pattern_type=None):
    pattern_type = pattern_type or get_effective_pattern_type(args)
    x = inputs.clone()
    if args.data == 'fmnist':
        white = (1.0 - FMNIST_MEAN) / FMNIST_STD
        if pattern_type == 'square':
            x[:, :, 21:26, 21:26] = white
        elif pattern_type in ['big_square', 'pgd_big_square']:
            x[:, :, 16:27, 16:27] = white
        elif pattern_type == 'huge_square':
            x[:, :, 14:28, 14:28] = white
        elif pattern_type == 'plus':
            s, size = 5, 5
            x[:, :, s:s+size, s] = white
            x[:, :, s+size//2, s-size//2:s+size//2+1] = white
        elif pattern_type == 'big_plus':
            c, arm = 7, 5
            x[:, :, c-arm:c+arm+1, c] = white
            x[:, :, c, c-arm:c+arm+1] = white
        elif pattern_type == 'huge_plus':
            c, arm = 7, 7
            x[:, :, max(0,c-arm):min(28,c+arm+1), c] = white
            x[:, :, c, max(0,c-arm):min(28,c+arm+1)] = white
        return _clip_normalized_fmnist(x)
    return x


def apply_clean_label_pgd_batch(model, inputs, labels, args, criterion):
    if args.data != 'fmnist' or getattr(args, 'clean_label_pgd', 0) != 1:
        return inputs
    mask = (labels == args.target_class)
    if mask.sum().item() == 0:
        return inputs
    selected = inputs[mask].detach()
    selected_labels = labels[mask].detach()
    eps = float(getattr(args, 'pgd_epsilon', 16.0/255.0)) / FMNIST_STD
    alpha = float(getattr(args, 'pgd_alpha', 2.0/255.0)) / FMNIST_STD
    steps = int(getattr(args, 'pgd_steps', 5))
    original = selected.clone().detach()
    x_adv = original.clone().detach()
    was_training = model.training
    model.eval()
    for _ in range(max(1, steps)):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, selected_labels)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, original + eps), original - eps)
            x_adv = _clip_normalized_fmnist(x_adv)
    if was_training:
        model.train()
    x_adv = add_pattern_tensor_bd(x_adv.detach(), args, pattern_type=get_effective_pattern_type(args))
    out = inputs.clone()
    out[mask] = x_adv
    return out


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, force_dirty_label=False, context='train'):
    clean_req = getattr(args, 'clean_label', 0) == 1
    clean_active = clean_req and not force_dirty_label
    if clean_active:
        attack_mode = f"clean-label-{getattr(args, 'clean_label_type', 0)}"
        source_class = args.target_class
        final_label = args.target_class
    else:
        attack_mode = 'dirty-label'
        source_class = args.base_class
        final_label = args.target_class
    pattern = get_effective_pattern_type(args)
    all_idxs = (dataset.targets == source_class).nonzero().flatten().tolist()
    if data_idxs is not None:
        all_idxs = list(set(all_idxs).intersection(set(data_idxs)))
    frac = 1 if poison_all else args.poison_frac
    num_poison = floor(frac * len(all_idxs))
    if num_poison <= 0:
        print('VERIFY_POISONING_START')
        print(f'context={context}')
        print(f'attack_mode={attack_mode}')
        print('VERIFY_POISONING_FAILED')
        print('reason=no poison samples')
        print('VERIFY_POISONING_END')
        return {'verification_passed': False, 'poisoned': 0}
    poison_idxs = random.sample(all_idxs, num_poison)
    before, after = [], []
    changed_pixels_first_sample = None
    for count, idx in enumerate(poison_idxs):
        before_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], 'item') else int(dataset.targets[idx])
        before.append(before_label)
        clean_img = dataset.inputs[idx] if args.data == 'fedemnist' else dataset.data[idx]
        np_before = np.array(clean_img).copy()
        bd_img = add_pattern_bd(clean_img, args.data, pattern_type=pattern, agent_idx=agent_idx)
        np_after = np.array(bd_img).copy()
        if count == 0:
            changed_pixels_first_sample = int(np.sum(np_before != np_after))
        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = final_label
        after_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], 'item') else int(dataset.targets[idx])
        after.append(after_label)
    labels_changed = sum(1 for b, a in zip(before, after) if b != a)
    expected = 0 if clean_active or source_class == final_label else num_poison
    label_ok = labels_changed == expected
    trig_ok = changed_pixels_first_sample is not None and changed_pixels_first_sample > 0
    passed = label_ok and trig_ok
    if getattr(args, 'verify_poisoning', 1) == 1:
        print('VERIFY_POISONING_START')
        print(f'context={context}')
        print(f'agent_idx={agent_idx}')
        print(f'attack_mode={attack_mode}')
        print(f'force_dirty_label={force_dirty_label}')
        print(f'clean_label_requested={clean_req}')
        print(f'clean_label_active={clean_active}')
        print(f'clean_label_type={getattr(args, "clean_label_type", 0)}')
        print(f'pattern_type_requested={args.pattern_type}')
        print(f'pattern_type_effective={pattern}')
        print(f'source_class={source_class}')
        print(f'target_class={args.target_class}')
        print(f'poison_frac={frac}')
        print(f'available_source_samples={len(all_idxs)}')
        print(f'poisoned_samples={num_poison}')
        print(f'labels_changed={labels_changed}')
        print(f'expected_labels_changed={expected}')
        print(f'changed_pixels_first_sample={changed_pixels_first_sample}')
        print(f'label_verification_passed={label_ok}')
        print(f'trigger_verification_passed={trig_ok}')
        print('VERIFY_POISONING_PASSED' if passed else 'VERIFY_POISONING_FAILED')
        if clean_active:
            print('CLEAN_LABEL_PROOF: labels_changed=0 means poisoned training samples kept their original target-class labels.')
        elif force_dirty_label:
            print('EVAL_ATTACK_PROOF: validation poison set uses base_class + trigger -> target_class to measure poison accuracy.')
        else:
            print('DIRTY_LABEL_PROOF: labels_changed=poisoned_samples means base-class labels were changed to target-class labels.')
        print('VERIFY_POISONING_END')
    return {'attack_mode': attack_mode, 'poisoned': num_poison, 'labels_changed': labels_changed, 'verification_passed': passed}


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    x = np.array(x.squeeze()).copy()
    if dataset == 'fmnist':
        if pattern_type == 'square':
            x[21:26, 21:26] = 255
        elif pattern_type in ['big_square', 'pgd_big_square']:
            x[16:27, 16:27] = 255
        elif pattern_type == 'huge_square':
            x[14:28, 14:28] = 255
        elif pattern_type == 'plus':
            s, size = 5, 5
            x[s:s+size, s] = 255
            x[s+size//2, s-size//2:s+size//2+1] = 255
        elif pattern_type == 'big_plus':
            c, arm = 7, 5
            x[c-arm:c+arm+1, c] = 255
            x[c, c-arm:c+arm+1] = 255
        elif pattern_type == 'huge_plus':
            c, arm = 7, 7
            x[max(0,c-arm):min(28,c+arm+1), c] = 255
            x[c, max(0,c-arm):min(28,c+arm+1)] = 255
        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            if trojan is not None:
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                x = np.clip(x + trojan, 0, 255)
    elif dataset == 'cifar10' and pattern_type == 'plus':
        s, size = 5, 6
        if agent_idx == -1:
            x[s:s+size+1, s, :] = 0
            x[s+size//2, s-size//2:s+size//2+1, :] = 0
        else:
            if agent_idx % 4 == 0: x[s:s+size//2+1, s, :] = 0
            elif agent_idx % 4 == 1: x[s+size//2+1:s+size+1, s, :] = 0
            elif agent_idx % 4 == 2: x[s+size//2, s-size//2:s+size//4+1, :] = 0
            else: x[s+size//2, s-size//4+1:s+size//2+1, :] = 0
    return x


def print_exp_details(args):
    print('======================================')
    print('Scenario: Cuckoo V1 Biology Nest Egg')
    print(f'Dataset: {args.data}')
    print(f'Global Rounds: {args.rounds}')
    print(f'Aggregation Function: {args.aggr}')
    print(f'Number of agents: {args.num_agents}')
    print(f'Fraction of agents: {args.agent_frac}')
    print(f'Class Per Agent: {getattr(args, "class_per_agent", 10)}')
    print(f'Batch size: {args.bs}')
    print(f'Client_LR: {args.client_lr}')
    print(f'Server_LR: {args.server_lr}')
    print(f'Client_Momentum: {args.client_moment}')
    print(f'RobustLR_threshold: {args.robustLR_threshold}')
    print(f'Noise Ratio: {args.noise}')
    print(f'Number of corrupt agents: {args.num_corrupt}')
    print(f'Base Class: {args.base_class}')
    print(f'Target Class: {args.target_class}')
    print(f'Poison Frac: {args.poison_frac}')
    print(f'Clip: {args.clip}')
    print(f'Clean Label Attack: {args.clean_label}')
    print(f'Clean Label Type: {args.clean_label_type}')
    print(f'Clean Label Auto Pattern: {args.clean_label_auto_pattern}')
    print(f'Verify Poisoning: {args.verify_poisoning}')
    print(f'Clean Label PGD: {getattr(args, "clean_label_pgd", 0)}')
    print(f'Malicious Boost: {getattr(args, "malicious_boost", 1.0)}')
    print(f'Corrupt Local Epochs: {getattr(args, "corrupt_local_ep", 0)}')
    print(f'Cuckoo Enabled: {getattr(args, "cuckoo", 0)}')
    print(f'Cuckoo Variant: {getattr(args, "cuckoo_variant", "none")}')
    print('======================================')
