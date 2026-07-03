#!/bin/bash
#SBATCH --job-name=finalck_clean5nn
#SBATCH --output=finalck_clean5nn_%j.out
#SBATCH --error=finalck_clean5nn_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --partition=general
set -e
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc || true
conda activate base || true
python --version
mkdir -p output_logs
COMMON="--data=fmnist --device=cpu --local_ep=1 --bs=1024 --num_agents=10 --rounds=100 --snap=1 --num_corrupt=2 --poison_frac=1.0 --robustLR_threshold=1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=4 --clean_label_adv=1 --pgd_eps=24 --pgd_alpha=2 --pgd_steps=20 --surrogate_epochs=10 --verify_poisoning=1 --pattern_type=plus --cuckoo=1 --cuckoo_variant=clean_label_egg --cuckoo_lambda=2.0 --cuckoo_incubation_lambda=0.0 --cuckoo_hatch_round=8 --cuckoo_warmup_rounds=12 --cuckoo_schedule=cosine --cuckoo_norm_cap=3.2 --cuckoo_min_cosine=-1.0 --cuckoo_top_frac=0.75 --cuckoo_budget_mode=global --cuckoo_layer_policy=classifier --cuckoo_sign_align=1 --cuckoo_classifier_target_rows=1 --cuckoo_classifier_focus=target_base --cuckoo_classifier_row_relax=0.40 --cuckoo_classifier_row_boost=4.5 --cuckoo_classifier_bias_boost=2.2 --cuckoo_center_egg=0 --cuckoo_pgd_bias=1.35"
for spec in "cnn_mnist 71" "cnn_no_dropout 72" "cnn_deep 73" "cnn_wide 74" "mlp 75"; do
  set -- $spec
  echo "RUN FINAL CLEAN PGD BIGSQUARE CUCKOO model=$1 seed=$2"
  python -u federated.py $COMMON --model_name=$1 --seed=$2
done
