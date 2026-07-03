# Final Cuckoo 5NN package

This package is the final thesis Cuckoo version built from the earlier results.

## What it learns from previous hypotheses

- H2: dirty-label attacks were stronger than clean-label, so dirty Cuckoo is the first 5NN run.
- H3: simply increasing poison fraction did not guarantee stronger attack, so this Cuckoo uses hidden residual shaping rather than only more poison.
- H4: RobustLR threshold 1 produced the strongest poison signal in PC07, so the dirty Cuckoo uses RobustLR-1.
- H5: square trigger was weak, so dirty Cuckoo uses plus.
- H6: privacy is a later stress test, not the first 5NN model sensitivity run.
- PGD clean-label big-square run: this held a strong backdoor while labels_changed=0, so the second runner uses clean_label_type=4, PGD eps=24, PGD steps=20, big_square.
- Previous multi-cuckoo problem: if mimicry/norm constraints are too strict, the backdoor disappears. This final version relaxes top_frac, norm_cap, and target/base classifier-row rules.

## Biology mapping

Cuckoo = clean host/nest update + poisoned egg residual + hatch schedule + mimicry constraints.

The corrupt client trains twice:
1. clean host/nest update on clean local data;
2. poisoned egg update on poisoned local data;
3. final update = clean update + scheduled hidden residual.

## Files

- `agent.py`: corrupt clients compute clean host update and poisoned egg update.
- `cuckoo_framework.py`: final hybrid Cuckoo residual selection, hatch schedule, classifier-row relaxation.
- `models.py`: five FMNIST models: `cnn_mnist`, `cnn_no_dropout`, `cnn_deep`, `cnn_wide`, `mlp`.
- `run_final_cuckoo_5nn_dirty_100_sequential.slurm`: first DAIC runner.
- `run_final_cuckoo_5nn_clean_pgd_bigsquare_100_sequential.slurm`: second DAIC runner.

## DAIC commands

```bash
cd ~
rm -rf final_cuckoo_5nn_package
unzip final_cuckoo_5nn_package.zip -d final_cuckoo_5nn_package
cd final_cuckoo_5nn_package
python -m py_compile *.py
sbatch run_final_cuckoo_5nn_dirty_100_sequential.slurm
```

After dirty run succeeds:

```bash
sbatch run_final_cuckoo_5nn_clean_pgd_bigsquare_100_sequential.slurm
```

## Good output signs

- `CUCKOO_FINAL_FRAMEWORK_START`
- `BIOLOGY_MAPPING`
- `VERIFY_POISONING_PASSED`
- `DIRTY_LABEL_PROOF` or `CLEAN_LABEL_PROOF`
- `CUCKOO_UPDATE_START`
- `stage=HATCH_RAMP` then `stage=FULL_HATCH_BACKDOOR_RETENTION`
- `Poison Loss/Poison Acc`
