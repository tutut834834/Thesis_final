@echo off
echo Running H1 Scenario 1 non-IID DIRTY-LABEL full thesis run, 200 rounds...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=0 --verify_noniid=1 --verify_poisoning=1 --seed=1
pause
