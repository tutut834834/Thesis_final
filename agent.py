import copy
import torch
import utils
from cuckoo_v1_biology_nest import CuckooV1BiologyNestEngine
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn


class Agent:
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        self.is_corrupt = self.id < args.num_corrupt
        self.cuckoo_enabled = self.is_corrupt and int(getattr(args, 'cuckoo', 0)) == 1
        self.cuckoo_engine = None

        if train_dataset is None:
            base_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            if self.cuckoo_enabled:
                self.clean_train_dataset = copy.deepcopy(base_dataset)
                poison_dataset = copy.deepcopy(base_dataset)
                utils.poison_dataset(poison_dataset, args, data_idxs, agent_idx=self.id, context='train')
                self.train_dataset = poison_dataset
            else:
                self.clean_train_dataset = None
                self.train_dataset = copy.deepcopy(base_dataset) if self.is_corrupt else base_dataset
                if self.is_corrupt:
                    utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id, context='train')
        else:
            if self.cuckoo_enabled:
                self.clean_train_dataset = utils.DatasetSplit(copy.deepcopy(train_dataset), data_idxs)
                poison_full = copy.deepcopy(train_dataset)
                utils.poison_dataset(poison_full, args, data_idxs, agent_idx=self.id, context='train')
                self.train_dataset = utils.DatasetSplit(poison_full, data_idxs)
            else:
                self.clean_train_dataset = None
                if self.is_corrupt:
                    poison_full = copy.deepcopy(train_dataset)
                    utils.poison_dataset(poison_full, args, data_idxs, agent_idx=self.id, context='train')
                    self.train_dataset = utils.DatasetSplit(poison_full, data_idxs)
                else:
                    self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        self.clean_train_loader = DataLoader(self.clean_train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False) if self.cuckoo_enabled else None
        self.n_data = len(self.train_dataset)

    def _local_epochs_for_this_agent(self):
        cep = int(getattr(self.args, 'corrupt_local_ep', 0)) if self.is_corrupt else 0
        return cep if cep > 0 else int(self.args.local_ep)

    def _train_one_update(self, global_model, criterion, loader, local_epochs=None, allow_pgd=False):
        initial = parameters_to_vector(global_model.parameters()).detach().clone()
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr, momentum=self.args.client_moment)
        epochs = self._local_epochs_for_this_agent() if local_epochs is None else int(local_epochs)
        for _ in range(epochs):
            for _, (inputs, labels) in enumerate(loader):
                optimizer.zero_grad()
                inputs = inputs.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                if allow_pgd and self.is_corrupt and getattr(self.args, 'clean_label', 0) == 1 and getattr(self.args, 'clean_label_pgd', 0) == 1:
                    inputs = utils.apply_clean_label_pgd_batch(global_model, inputs, labels, self.args, criterion)
                outputs = global_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                if self.args.clip > 0:
                    with torch.no_grad():
                        params = parameters_to_vector(global_model.parameters())
                        update = params - initial
                        denom = max(1, torch.norm(update, p=2) / self.args.clip)
                        update.div_(denom)
                        vector_to_parameters(initial + update, global_model.parameters())
        with torch.no_grad():
            return (parameters_to_vector(global_model.parameters()).double() - initial.double()).detach()

    def local_train(self, global_model, criterion, cur_round=1):
        if not self.cuckoo_enabled:
            return self._train_one_update(global_model, criterion, self.train_loader, allow_pgd=True)
        if self.cuckoo_engine is None:
            variant = str(getattr(self.args, 'cuckoo_variant', 'v1_biology_nest')).lower()
            if variant in ('v1_biology_nest', 'biology_nest', 'bio_nest', 'nest_egg', 'framework1', 'f1'):
                self.cuckoo_engine = CuckooV1BiologyNestEngine(self.args, global_model, self.id)
            else:
                raise ValueError(f'Unknown Cuckoo variant: {variant}')
        initial = parameters_to_vector(global_model.parameters()).detach().clone()
        clean_update = self._train_one_update(global_model, criterion, self.clean_train_loader, allow_pgd=False)
        vector_to_parameters(initial.clone(), global_model.parameters())
        poison_update = self._train_one_update(global_model, criterion, self.train_loader, allow_pgd=True)
        vector_to_parameters(initial.clone(), global_model.parameters())
        cuckoo_update, stats = self.cuckoo_engine.build(clean_update, poison_update, cur_round)
        diag_every = int(getattr(self.args, 'cuckoo_diag_every', 10))
        if cur_round == 1 or (diag_every > 0 and cur_round % diag_every == 0) or cur_round % int(self.args.snap) == 0:
            stats.print_block()
        return cuckoo_update.detach()
