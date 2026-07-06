
import torch
from torchvision import datasets, transforms

def get_data():
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
    return train, test

def poison_batch(x, mode):
    if mode == "dirty":
        x[:, :, 26:28, 26:28] = 1.0
    elif mode == "pgd":
        x = x + 0.1 * torch.sign(x)
    return x
