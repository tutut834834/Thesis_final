import torch
import utils
import models
import math
import copy
import numpy as np
import os
import re
import sys
import random
import subprocess

from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.data import DataLoader
import torch.nn as nn
from time import strftime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils import H5Dataset


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def make_windows_safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(" ", "_")
    return name


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DummyWriter:
    """
    Emergency writer for PC-room runs.
    No TensorBoard. No event-file crash. Txt logs only.
    """
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def github_upload_round(rnd):
    """
    Auto-upload output_logs, code_dump, Python files, and txt files.
    Does NOT upload TensorBoard logs. Faster and safer.
    """
    try:
        subprocess.run("git add output_logs code_dump *.py *.txt", shell=True, timeout=30)
        subprocess.run(f'git commit -m "PC01 auto upload round {rnd}"', shell=True, timeout=30)
        subprocess.run("git push", shell=True, timeout=60)
        print(f"GITHUB_UPLOAD_DONE_ROUND_{rnd}")
    except Exception as e:
        print(f"GITHUB_UPLOAD_FAILED_ROUND_{rnd}: {e}")


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
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0

    set_all_seeds(args.seed)

    run_name = (
        f"PC01_H1_noniid"
        f"_{args.data}"
        f"_r{args.rounds}"
        f"_cpa{args.class_per_agent}"
        f"_base{args.base_class}"
        f"_target{args.target_class}"
        f"_cor{args.num_corrupt}"
        f"_pf{args.poison_frac}"
        f"_rlr{args.robustLR_threshold}"
        f"_cl{getattr(args, 'clean_label', 0)}"
        f"_cltype{getattr(args, 'clean_label_type', 0)}"
        f"_seed{args.seed}"
        f"_{strftime('%Y%m%d_%H%M%S')}"
    )

    run_name = make_windows_safe_filename(run_name)

    os.makedirs("output_logs", exist_ok=True)
    os.makedirs("code_dump", exist_ok=True)

    txt_log_path = os.path.join("output_logs", f"{run_name}_console_output.txt")
    txt_log_file = open(txt_log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(original_stdout, txt_log_file)
    sys.stderr = TeeLogger(original_stderr, txt_log_file)

    print(f"Console output will be saved to: {txt_log_path}")
    print("TENSORBOARD DISABLED FOR FAST PC ROOM RUN")
    print("AUTO GITHUB UPLOAD ENABLED: output_logs/code_dump/*.py/*.txt every round")

    utils.print_exp_details(args)

    writer = DummyWriter()

    cum_poison_acc_mean = 0

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
        context="poisoned_validation_eval"
    )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist':
            agent = Agent(_id, args)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])

        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    n_model_params = len(parameters_to_vector(global_model.parameters()))

    aggregator = Aggregation(
        agent_data_sizes,
        n_model_params,
        poisoned_val_loader,
        args,
        writer
    )

    criterion = nn.CrossEntropyLoss().to(args.device)

    github_upload_round("START")

    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}

        for agent_id in np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False
        ):
            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update

            vector_to_parameters(
                copy.deepcopy(rnd_global_params),
                global_model.parameters()
            )

        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        txt_log_file.flush()

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    val_loader,
                    args
                )

                print(f'| Round: {rnd} |')
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} |')

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args
                )

                cum_poison_acc_mean += poison_acc

                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                print(f'| Cumulative Poison Acc Mean: {cum_poison_acc_mean / rnd:.3f} |')
                txt_log_file.flush()

        github_upload_round(rnd)

    writer.close()
    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')

    github_upload_round("FINAL")

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    txt_log_file.close()