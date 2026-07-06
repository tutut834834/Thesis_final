
import argparse
import torch

def args_parser():
    p = argparse.ArgumentParser()

    # core FL
    p.add_argument('--data', default='fmnist')
    p.add_argument('--num_agents', type=int, default=10)
    p.add_argument('--rounds', type=int, default=50)
    p.add_argument('--local_ep', type=int, default=1)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--snap', type=int, default=1)
    p.add_argument('--device', default='cpu')

    # attack control
    p.add_argument('--attack_mode', type=str, default='dirty') # dirty / clean / pgd
    p.add_argument('--poison_frac', type=float, default=0.1)
    p.add_argument('--base_class', type=int, default=5)
    p.add_argument('--target_class', type=int, default=7)

    # cuckoo v2
    p.add_argument('--cuckoo_v2', type=int, default=1)
    p.add_argument('--lambda_v2', type=float, default=1.2)
    p.add_argument('--norm_cap', type=float, default=5.0)

    # defenses
    p.add_argument('--robustLR_threshold', type=int, default=4)

    return p.parse_args()
