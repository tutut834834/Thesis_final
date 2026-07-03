@echo off
echo Running H1 Scenario 1 non-IID CLEAN-LABEL quick test, 10 rounds...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=1 --clean_label_type=1 --verify_noniid=1 --verify_poisoning=1 --seed=1
pause
