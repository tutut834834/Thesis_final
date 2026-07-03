#!/bin/bash

echo "Scenario 1 / Hypothesis 1: Non-IID dirty-label vs clean-label"
echo "This script runs BOTH experiments and creates txt logs in output_logs/"

# Important:
# With class_per_agent=2, corrupt client 0 gets classes [0,1].
# Therefore base_class=0 and target_class=1 are used so both dirty and clean attacks can poison.

python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=0 --verify_noniid=1 --verify_poisoning=1 --seed=1

python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=1 --clean_label_type=1 --verify_noniid=1 --verify_poisoning=1 --seed=1

echo "Scenario 1 experiments finished."
