import copy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

import utils
from cuckoo_classifier_nest import ClassifierNestEngine
from cuckoo_clean_label_egg import CleanLabelEggEngine
from cuckoo_framework import CuckooEngine
from cuckoo_multi_cuckoo import MultiCuckooCollusionEngine


class Agent:
    def __init__(self, id, args, train_dataset=None, data_idxs=None, surrogate_model=None):
        self.id = id
        self.args = args
        self.is_corrupt = self.id < args.num_corrupt
        self.cuckoo_enabled = self.is_corrupt and int(getattr(args, 'cuckoo', 0)) == 1
        self.cuckoo_engine = None

        if train_dataset is None:
            base_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            if self.cuckoo_enabled:
                self.clean_train_dataset = copy.deepcopy(base_dataset)
                self.train_dataset = copy.deepcopy(base_dataset)
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id, surrogate_model=surrogate_model)
            else:
                self.train_dataset = base_dataset
                self.clean_train_dataset = None
                if self.is_corrupt:
                    utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id, surrogate_model=surrogate_model)
        else:
            if self.cuckoo_enabled:
                self.clean_train_dataset = utils.DatasetSplit(copy.deepcopy(train_dataset), data_idxs)
                poisoned_full = copy.deepcopy(train_dataset)
                utils.poison_dataset(poisoned_full, args, data_idxs, agent_idx=self.id, surrogate_model=surrogate_model)
                self.train_dataset = utils.DatasetSplit(poisoned_full, data_idxs)
            elif self.is_corrupt:
                poisoned_full = copy.deepcopy(train_dataset)
                utils.poison_dataset(poisoned_full, args, data_idxs, agent_idx=self.id, surrogate_model=surrogate_model)
                self.train_dataset = utils.DatasetSplit(poisoned_full, data_idxs)
                self.clean_train_dataset = None
            else:
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                self.clean_train_dataset = None

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=False)
        self.clean_train_loader = None
        if self.cuckoo_enabled:
            self.clean_train_loader = DataLoader(self.clean_train_dataset, batch_size=self.args.bs, shuffle=True,
                                                 num_workers=args.num_workers, pin_memory=False)
        self.n_data = len(self.train_dataset)

    def _epochs(self):
        corrupt_ep = int(getattr(self.args, 'corrupt_local_ep', 0)) if self.is_corrupt else 0
        return corrupt_ep if corrupt_ep > 0 else int(self.args.local_ep)

    def _train_one_update(self, global_model, criterion, loader):
        initial = parameters_to_vector(global_model.parameters()).detach().clone()
        global_model.train()
        opt = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
        for _ in range(self._epochs()):
            for _, (inputs, labels) in enumerate(loader):
                opt.zero_grad()
                inputs = inputs.to(self.args.device, non_blocking=True)
                labels = labels.to(self.args.device, non_blocking=True)
                loss = criterion(global_model(inputs), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                opt.step()
                if self.args.clip > 0:
                    with torch.no_grad():
                        params = parameters_to_vector(global_model.parameters())
                        update = params - initial
                        denom = max(1, torch.norm(update, p=2) / self.args.clip)
                        update.div_(denom)
                        vector_to_parameters(initial + update, global_model.parameters())
        with torch.no_grad():
            return (parameters_to_vector(global_model.parameters()).double() - initial.double()).detach()

    def _engine(self, global_model):
        if self.cuckoo_engine is not None:
            return self.cuckoo_engine
        variant = str(getattr(self.args, 'cuckoo_variant', 'final_hybrid')).lower()
        if variant in ('classifier_nest', 'classifier', 'final_hybrid', 'v2'):
            self.cuckoo_engine = ClassifierNestEngine(self.args, global_model, self.id)
        elif variant in ('clean_label_egg', 'clean_egg'):
            self.cuckoo_engine = CleanLabelEggEngine(self.args, global_model, self.id)
        elif variant in ('multi_cuckoo', 'multi', 'collusion'):
            self.cuckoo_engine = MultiCuckooCollusionEngine(self.args, global_model, self.id)
        else:
            self.cuckoo_engine = CuckooEngine(self.args, global_model, self.id)
        return self.cuckoo_engine

    def local_train(self, global_model, criterion, cur_round=1):
        if not self.cuckoo_enabled:
            return self._train_one_update(global_model, criterion, self.train_loader)

        initial_params = parameters_to_vector(global_model.parameters()).detach().clone()
        clean_update = self._train_one_update(global_model, criterion, self.clean_train_loader)
        vector_to_parameters(initial_params.clone(), global_model.parameters())
        poison_update = self._train_one_update(global_model, criterion, self.train_loader)
        vector_to_parameters(initial_params.clone(), global_model.parameters())
        cuckoo_update, stats = self._engine(global_model).build(clean_update, poison_update, cur_round)
        diag_every = int(getattr(self.args, 'cuckoo_diag_every', 5))
        if cur_round == 1 or cur_round % int(self.args.snap) == 0 or (diag_every > 0 and cur_round % diag_every == 0):
            stats.print_block()
        return cuckoo_update.detach()
