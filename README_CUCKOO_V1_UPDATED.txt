Cuckoo V1 Biology Nest Egg - UPDATED with user code insights

Included insights:
1. Forced dirty-label poisoned validation for clean-label evaluation.
2. Clean-label PGD batch perturbation option from user's code.
3. malicious_boost retained as optional command flag.
4. corrupt_local_ep retained for corrupt-client local epoch testing.
5. Safe TeeLogger output_logs text logging.
6. Dirty-label and clean-label verification blocks.
7. Copy-deepcopy split: honest clients are not contaminated by corrupt-client poisoning.

Run:
for %f in (*.py) do python -m py_compile "%f"
run_v1_dirty_100_fmnist.bat
run_v1_clean_100_fmnist.bat
run_v1_clean_pgd_100_fmnist.bat
python extract_cuckoo_v1_metrics.py output_logs
