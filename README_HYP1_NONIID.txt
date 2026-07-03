Scenario 1 / Hypothesis 1
=========================

Thesis scenario:
Dirty-label vs clean-label attack under a non-IID federated data split.

Hypothesis:
Under a non-IID federated setting, dirty-label attacks are expected to produce stronger backdoor success
than clean-label attacks, while clean-label attacks remain more stealthy because labels are not changed.

Source basis:
- agent.py, aggregation.py, models.py: same logic as the uploaded clean-label project code.
- options.py, utils.py, federated.py: modified from the uploaded clean-label project code to add Scenario 1 non-IID controls and clearer thesis logs.

Changes made:
1. options.py
   - Added --class_per_agent
   - Added --verify_noniid
   - Added --seed
   - Kept clean-label options:
       --clean_label
       --clean_label_type
       --clean_label_auto_pattern
       --verify_poisoning

2. utils.py
   - distribute_data now reads args.class_per_agent.
   - With --class_per_agent=2, each client gets only 2 classes.
   - Added VERIFY_NONIID_START / VERIFY_NONIID_END block.
   - Added corrupt_client_check lines to prove whether corrupt clients have base_class and target_class.
   - Kept clean-label poison_dataset logic:
       dirty-label: base_class + trigger -> target_class, labels changed
       clean-label: target_class + trigger -> target_class, labels unchanged
       poisoned validation: force_dirty_label=True

3. federated.py
   - Added seed setting for fair dirty vs clean comparison.
   - Added non-IID information to run_name.
   - Keeps full txt logging in output_logs/.
   - Keeps TensorBoard logging in logs/.
   - Keeps poisoned validation forced to dirty-label evaluation.

4. agent.py
   - Same training logic.
   - Passes context="train" into poison_dataset, so verification logs clearly show training poisoning.

Recommended Scenario 1 commands:
-------------------------------

Quick dirty-label test:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=0 --verify_noniid=1 --verify_poisoning=1 --seed=1

Quick clean-label test:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=1 --clean_label_type=1 --verify_noniid=1 --verify_poisoning=1 --seed=1

Full dirty-label run:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=0 --verify_noniid=1 --verify_poisoning=1 --seed=1

Full clean-label run:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=1 --clean_label_type=1 --verify_noniid=1 --verify_poisoning=1 --seed=1

Important:
----------
In your earlier failed H1 attempt, class_per_agent=2 gave corrupt client 0 only classes [0,1],
but the attack used base_class=5 and target_class=7. Therefore training poisoning failed.

This Scenario 1 version uses:
    --base_class=0 --target_class=1

This makes the dirty and clean runs valid with num_corrupt=1 because corrupt client 0 contains both classes 0 and 1.

What to compare:
----------------
From both txt logs, compare rounds 30, 60, 100, 200:
- Val_Loss
- Val_Acc
- Val_Per_Class_Acc
- Poison Loss
- Poison Acc
- labels_changed
- changed_pixels_first_sample
- VERIFY_POISONING_PASSED
