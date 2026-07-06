import torch 
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utils.print_exp_details(args)

    # =========================
    # LOGGING (NEW)
    # =========================
    import os
    os.makedirs("output_logs", exist_ok=True)

    log_path = f"output_logs/pc16_pc17_pc18_run_{args.pattern_type}.txt"
    log_file = open(log_path, "w", buffering=1)

    file_name = f"""time:{ctime()}-clip:{args.clip}-noise:{args.noise}-aggr:{args.aggr}-cor:{args.num_corrupt}-pattern:{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)

    cum_poison_acc_mean = 0

    # =========================
    # DATA
    # =========================
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)

    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # =========================
    # MODEL + AGENTS
    # =========================
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
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # =========================
    # HEADER LOG
    # =========================
    print("START TRAINING")
    log_file.write("START TRAINING\n")

    # =========================
    # TRAIN LOOP
    # =========================
    for rnd in tqdm(range(1, args.rounds + 1)):

        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}

        round_update_norm = 0.0

        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False):

            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update

            round_update_norm += torch.norm(update).item()

            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())

        # =========================
        # AGGREGATION
        # =========================
        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        # =========================
        # EVAL
        # =========================
        if rnd % args.snap == 0:

            with torch.no_grad():

                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)

                cum_poison_acc_mean += poison_acc

                # =========================
                # METRICS LINE (NEW)
                # =========================
                msg = (
                    f"Round {rnd:03d} | "
                    f"ValAcc={val_acc:.4f} | "
                    f"PoisonAcc={poison_acc:.4f} | "
                    f"ValLoss={val_loss:.4f} | "
                    f"PoisonLoss={poison_loss:.4f} | "
                    f"UpdateNorm={round_update_norm:.4f}"
                )

                print(msg)
                log_file.write(msg + "\n")

                # TensorBoard
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Poison/Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Loss', poison_loss, rnd)
                writer.add_scalar('Poison/CumMean', cum_poison_acc_mean / rnd, rnd)
                writer.add_scalar('Update/Norm', round_update_norm, rnd)

    print("TRAINING FINISHED")
    log_file.write("TRAINING FINISHED\n")
    log_file.close()