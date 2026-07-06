
import torch
from models import CNN
from options import args_parser
from utils import get_data
from agent import Agent
from cuckoo_v2_engine import CuckooV2
from hypothesis_engine import HypothesisEngine

args = args_parser()

train, test = get_data()

agents = []
H = HypothesisEngine(args)
engine = CuckooV2(args, H)

global_model = CNN()

for i in range(args.num_agents):
    agents.append(Agent(i, CNN(), train, args))

print("START V2 CUCKOO")

for r in range(args.rounds):

    updates = []

    for a in agents:
        clean = a.train()
        poison = clean * 1.1 if args.attack_mode != "clean" else clean

        if args.cuckoo_v2:
            u = engine.build(clean, poison, r)
        else:
            u = clean

        updates.append(u)

    new_global = sum(updates) / len(updates)
    torch.nn.utils.vector_to_parameters(new_global, global_model.parameters())

    if r % args.snap == 0:
        print("round", r)

print("DONE")
