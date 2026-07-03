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


def distribute_data(dataset, args, n_classes=10, class_per_agent=None):
    """
    Scenario 1 / Hypothesis 1 non-IID data split.

    class_per_agent = 10:
        IID-ish setting. Each client may receive all 10 classes.

    class_per_agent = 2:
        Non-IID setting. Each client receives only 2 classes.
        With 10 clients on FMNIST, client 0 receives classes [0, 1].
        Therefore the recommended H1 pair is base_class=0 and target_class=1.

    This function prints a verification block into the txt log when --verify_noniid=1.
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

    if hasattr(args, "verify_noniid") and args.verify_noniid == 1:
        print("VERIFY_NONIID_START")
        print(f"scenario=Hypothesis 1 - non-IID dirty-label vs clean-label")
        print(f"num_agents={args.num_agents}")
        print(f"class_per_agent={class_per_agent}")
        print(f"n_classes={n_classes}")
        print(f"shard_size={shard_size}")
        print(f"slice_size={slice_size}")

        corrupt_clients = list(range(args.num_corrupt))
        print(f"corrupt_clients={corrupt_clients}")
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

        # Important warning for H1:
        # dirty-label needs base_class inside corrupt clients.
        # clean-label needs target_class inside corrupt clients.
        for corrupt_id in corrupt_clients:
            user_idxs = list(dict_users[corrupt_id])
            user_labels = [int(dataset.targets[idx]) for idx in user_idxs]
            unique_classes = sorted(list(set(user_labels)))

            has_base = args.base_class in unique_classes
            has_target = args.target_class in unique_classes

            print(
                f"corrupt_client_check client={corrupt_id} "
                f"unique_classes={unique_classes} "
                f"has_base_class={has_base} "
                f"has_target_class={has_target}"
            )

            if not has_base:
                print(
                    "WARNING_NONIID_DIRTY: corrupt client has no base_class samples. "
                    "Dirty-label training poisoning will fail unless you change base_class or num_corrupt."
                )

            if not has_target:
                print(
                    "WARNING_NONIID_CLEAN: corrupt client has no target_class samples. "
                    "Clean-label training poisoning will fail unless you change target_class or num_corrupt."
                )

        print("VERIFY_NONIID_PASSED")
        print("VERIFY_NONIID_END")

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

    If clean_label_auto_pattern=0, args.pattern_type is used exactly.
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

        print("VERIFY_POISONING_END")

    return {
        "attack_mode": attack_mode,
        "poisoned": num_poison,
        "available": len(all_idxs),
        "labels_changed": labels_changed,
        "expected_labels_changed": expected_labels_changed,
        "changed_pixels_first_sample": changed_pixels_first_sample,
        "verification_passed": total_verification_passed,
        "pattern_type_effective": effective_pattern_type
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
    print('    Scenario: Hypothesis 1 - Non-IID dirty-label vs clean-label')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Class Per Agent: {args.class_per_agent}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Base Class: {args.base_class}')
    print(f'    Target Class: {args.target_class}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print(f'    Seed: {args.seed}')

    if hasattr(args, "clean_label"):
        print(f'    Clean Label Attack: {args.clean_label}')
        print(f'    Clean Label Type: {args.clean_label_type}')
        print(f'    Clean Label Auto Pattern: {args.clean_label_auto_pattern}')
        print(f'    Verify Poisoning: {args.verify_poisoning}')

    if hasattr(args, "verify_noniid"):
        print(f'    Verify Non-IID: {args.verify_noniid}')

    print('======================================')
