import copy
import math
import os
import random
import re
import sys
from time import strftime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cuckoo_framework
import models
import utils
from agent import Agent
from aggregation import Aggregation
from options import args_parser


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_windows_safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name.replace(' ', '_')


class TeeLogger:
    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_file.write(message)
        self.terminal_stream.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal_stream.isatty()


if __name__ == '__main__':
    args = args_parser()
    set_all_seeds(getattr(args, 'seed', 1))
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0

    run_name = make_windows_safe_filename(
        f"finalcuckoo_{args.data}_model{args.model_name}_r{args.rounds}_cor{args.num_corrupt}"
        f"_pf{args.poison_frac}_rlr{args.robustLR_threshold}_cl{args.clean_label}"
        f"_cltype{args.clean_label_type}_ck{args.cuckoo_variant}_seed{args.seed}_{strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output_logs', exist_ok=True)
    txt_log_path = os.path.join('output_logs', f'{run_name}_console_output.txt')
    txt_log_file = open(txt_log_path, 'a', encoding='utf-8', buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeLogger(original_stdout, txt_log_file)
    sys.stderr = TeeLogger(original_stderr, txt_log_file)

    print(f'Console output will be saved to: {txt_log_path}')
    utils.print_exp_details(args)
    cuckoo_framework.print_cuckoo_header(args)
    print('FINAL_CUCKOO_THESIS_DECISION: dirty uses PC07 RobustLR-1 plus pf0.10; clean uses PGD big_square cltype4 because it previously held the backdoor.')

    log_dir = os.path.join('logs', run_name)
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard logs will be saved to: {log_dir}')

    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    user_groups = utils.distribute_data(train_dataset, args) if args.data != 'fedemnist' else None

    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True, force_dirty_label=True, context='poisoned_validation_eval')
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    global_model = models.get_model(args.data, args.model_name).to(args.device)
    surrogate_model = None
    if args.clean_label == 1 and args.clean_label_adv == 1 and args.num_corrupt > 0:
        if args.data == 'fmnist':
            print('Training clean-label PGD surrogate model on clean data...')
            surrogate_model = utils.train_clean_label_surrogate(train_dataset, args)
        else:
            print('WARNING: clean_label_adv PGD implemented for fmnist only; continuing without surrogate.')

    agents, agent_data_sizes = [], {}
    for _id in range(args.num_agents):
        if args.data == 'fedemnist':
            agent = Agent(_id, args, surrogate_model=surrogate_model)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id], surrogate_model=surrogate_model)
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    n_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)
    cum_poison_acc_mean = 0.0

    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        selected_agents = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        for agent_id in selected_agents:
            update = agents[agent_id].local_train(global_model, criterion, cur_round=rnd)
            agent_updates_dict[agent_id] = update
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        txt_log_file.flush()

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Round: {rnd} |')
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} |')
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                txt_log_file.flush()

    writer.close()
    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')
    sys.stdout, sys.stderr = original_stdout, original_stderr
    txt_log_file.close()
