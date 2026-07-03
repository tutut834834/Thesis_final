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
import shutil
import glob

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


PC_NAME = "PC10_H5_square_clean"
BRANCH_NAME = "pc10-h5-square-clean"


class DummyWriter:
    """
    Safe replacement for TensorBoard SummaryWriter.
    This avoids Windows/TensorBoard event-file crashes.
    The real thesis evidence is the txt console output in output_logs/.
    """
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def make_windows_safe_filename(name):
    """
    Windows does not allow these characters in file/folder names:
    < > : " / \\ | ? *
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(" ", "_")
    return name


def set_all_seeds(seed):
    """
    Makes dirty and clean-label Scenario 2 stealth runs comparable.
    Use the same --seed in both runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeeLogger:
    """
    Writes everything both to the terminal and to a text file.
    This produces thesis txt logs for dirty and clean runs.
    """
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


def run_cmd(cmd, timeout=60):
    """
    Runs a command and prints return code.
    Does not crash training if git fails.
    """
    try:
        print(f"RUN: {cmd}")
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        print(f"RETURN: {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"COMMAND_FAILED: {cmd}")
        print(f"ERROR: {e}")
        return 999


def sync_visible_pc_folder():
    """
    Keep a visible PC03 folder in GitHub:
      PC10_H5_square_clean/
      PC10_H5_square_clean/output_logs/
    """
    try:
        os.makedirs(PC_NAME, exist_ok=True)
        os.makedirs(os.path.join(PC_NAME, "output_logs"), exist_ok=True)

        for fname in [
            "federated.py",
            "agent.py",
            "aggregation.py",
            "models.py",
            "options.py",
            "utils.py",
            "PC10_H5_square_clean_description_expectation.txt"
        ]:
            if os.path.exists(fname):
                shutil.copy2(fname, os.path.join(PC_NAME, os.path.basename(fname)))

        for log_file in glob.glob(os.path.join("output_logs", "*hyp2_stealth*seed3*console_output.txt")):
            shutil.copy2(log_file, os.path.join(PC_NAME, "output_logs", os.path.basename(log_file)))

        proof_path = os.path.join(PC_NAME, "PC03_FOLDER_PROOF.txt")
        with open(proof_path, "w", encoding="utf-8") as f:
            f.write("PC03 H2 stealth dirty visible folder proof.\n")
            f.write("This folder stores code and txt logs for the PC03 H2 stealth dirty-label run.\n")

    except Exception as e:
        print(f"VISIBLE_FOLDER_SYNC_FAILED: {e}")


def github_upload_round(rnd):
    """
    Uploads current code and txt logs to GitHub branch pc10-h5-square-clean.
    Safe behavior:
    - If another git process has an index.lock, skip this upload instead of crashing.
    - Training continues even when GitHub upload fails.
    """
    try:
        if os.path.exists(os.path.join(".git", "index.lock")):
            print(f"GITHUB_UPLOAD_SKIPPED_ROUND_{rnd}: git index.lock exists")
            return

        sync_visible_pc_folder()

        run_cmd(f"git checkout {BRANCH_NAME}", timeout=30)
        run_cmd(f"git branch --set-upstream-to=origin/{BRANCH_NAME} {BRANCH_NAME}", timeout=30)

        run_cmd(f"git add output_logs code_dump {PC_NAME} *.py *.txt PC*.txt", timeout=30)

        commit_code = run_cmd(f'git commit -m "PC10 auto upload round {rnd}"', timeout=30)

        # commit_code 1 often just means "nothing to commit"; still push safely.
        push_code = run_cmd(f"git push origin HEAD:{BRANCH_NAME}", timeout=90)

        if push_code == 0:
            print(f"GITHUB_UPLOAD_DONE_ROUND_{rnd}")
        else:
            print(f"GITHUB_UPLOAD_PUSH_FAILED_ROUND_{rnd}")

    except Exception as e:
        print(f"GITHUB_UPLOAD_FAILED_ROUND_{rnd}: {e}")


if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0

    set_all_seeds(args.seed)

    run_name = (
        f"PC10_H5_square_clean"
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

    os.makedirs("logs", exist_ok=True)
    os.makedirs("output_logs", exist_ok=True)
    os.makedirs(PC_NAME, exist_ok=True)
    os.makedirs(os.path.join(PC_NAME, "output_logs"), exist_ok=True)

    txt_log_path = os.path.join("output_logs", f"{run_name}_console_output.txt")
    txt_log_file = open(txt_log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(original_stdout, txt_log_file)
    sys.stderr = TeeLogger(original_stderr, txt_log_file)

    print(f"Console output will be saved to: {txt_log_path}")

    utils.print_exp_details(args)

    print("TensorBoard disabled safely for Windows.")
    print("Txt logs are the thesis evidence.")
    writer = DummyWriter()

    github_upload_round("START")

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

    # Poisoned validation set:
    # Even for clean-label training, validation must test base_class + trigger -> target_class.
    # Therefore force_dirty_label=True.
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

            # make sure every agent gets same copy of the global model in a round
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())

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

                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)

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

                writer.add_scalar(
                    'Poison/Base_Class_Accuracy',
                    val_per_class_acc[args.base_class],
                    rnd
                )
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar(
                    'Poison/Cumulative_Poison_Accuracy_Mean',
                    cum_poison_acc_mean / rnd,
                    rnd
                )

                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')
                print(f'| Cumulative Poison Acc Mean: {cum_poison_acc_mean / rnd:.3f} |')
                print('STEALTH_METRICS_NOTE: compare labels_changed, Val_Acc, Val_Loss, and Poison Acc between dirty-label and clean-label txt logs.')
                txt_log_file.flush()

        github_upload_round(rnd)

    writer.close()
    print('Training has finished!')
    print(f'Full console output saved to: {txt_log_path}')
    github_upload_round("FINISH")

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    txt_log_file.close()