
import torch
from torch.utils.data import DataLoader
from utils import poison_batch

class Agent:
    def __init__(self, id, model, dataset, args):
        self.id = id
        self.model = model
        self.dataset = dataset
        self.args = args
        self.loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    def train(self):
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()

        for _ in range(self.args.local_ep):
            for x,y in self.loader:
                x = poison_batch(x, self.args.attack_mode)
                opt.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
