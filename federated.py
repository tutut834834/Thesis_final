import torch
import utils
import models
import math
import copy
import numpy as np
import os
import re
import sys
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import strftime
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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


# =========================================================
# 🔥 FIX: force initialization of Lazy layers (CRITICAL)
# =========================================================
def warmup_model(model, device):
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 1, 28, 28).to(device)
        model(dummy)
    model.train()


if __name__ == '__main__':
    args = args_parser()

    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0

    run_name = make_windows_safe_filename(
        f"cuckooV1_{args.data}_r{args.rounds}_cor{args.num_corrupt}_pf{args.poison_frac}_rlr{args.robustLR_threshold}_cl{getattr(args,'clean_label',0)}_cltype{getattr(args,'clean_label_type',0)}_cuckoo{getattr(args,'cuckoo',0)}_seed{getattr(args,'seed',0) if hasattr(args,'seed') else 'NA'}_{strftime('%Y%m%d_%H%M%S')}"
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

    writer = SummaryWriter(os.path.join('logs', run_name))

    train_dataset, val_dataset = utils.get_datasets(args.data)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)

    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()

    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(
        poisoned_val_set.dataset,
        args,
        idxs,
        poison_all=True,
        force_dirty_label=True,
        context='poisoned_validation_eval'
    )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # =========================================================
    # MODEL INIT
    # =========================================================
    global_model = models.get_model(args.model_type).to(args.device)

    # 🔥 IMPORTANT FIX FOR CNN_WIDE / CNN_DEEP
    warmup_model(global_model, args.device)

    agents, agent_data_sizes = [], {}

    for _id in range(args.num_agents):
        agent = Agent(_id, args) if args.data == 'fedemnist' else Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    n_params = len(parameters_to_vector(global_model.parameters()))

    aggregator = Aggregation(agent_data_sizes, n_params, poisoned_val_loader, args, writer)

    criterion = nn.CrossEntropyLoss().to(args.device)

    cum_poison_acc_mean = 0.0

    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()

        agent_updates_dict = {}

        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion, cur_round=rnd)

            if agent_id < args.num_corrupt and getattr(args, 'malicious_boost', 1.0) != 1.0:
                update = update * float(args.malicious_boost)
                print(f'MALICIOUS_BOOST_APPLIED agent={agent_id} boost={args.malicious_boost}')

            agent_updates_dict[agent_id] = update

            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())

        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_pc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)

                cum_poison_acc_mean += poison_acc

                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)

                print(f'| Round: {rnd} |')
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_pc} |')
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                print(f'| Cumulative Poison Acc Mean: {cum_poison_acc_mean / rnd:.6f} |')

        txt_log_file.flush()

    writer.close()

    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    txt_log_file.close()