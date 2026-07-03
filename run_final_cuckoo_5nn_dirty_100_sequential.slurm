#!/bin/bash
#SBATCH --job-name=finalck_dirty5nn
#SBATCH --output=finalck_dirty5nn_%j.out
#SBATCH --error=finalck_dirty5nn_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
set -e
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc || true
conda activate base || true
python --version
mkdir -p output_logs
COMMON="--data=fmnist --device=cpu --local_ep=1 --bs=1024 --num_agents=10 --rounds=100 --snap=1 --num_corrupt=1 --poison_frac=0.10 --robustLR_threshold=1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poisoning=1 --pattern_type=plus --cuckoo=1 --cuckoo_variant=final_hybrid --cuckoo_lambda=1.8 --cuckoo_incubation_lambda=0.0 --cuckoo_hatch_round=12 --cuckoo_warmup_rounds=12 --cuckoo_schedule=cosine --cuckoo_norm_cap=3.0 --cuckoo_min_cosine=-1.0 --cuckoo_top_frac=0.70 --cuckoo_budget_mode=global --cuckoo_layer_policy=classifier --cuckoo_sign_align=1 --cuckoo_classifier_target_rows=1 --cuckoo_classifier_focus=target_base --cuckoo_classifier_row_relax=0.35 --cuckoo_classifier_row_boost=4.0 --cuckoo_classifier_bias_boost=2.0 --cuckoo_center_egg=0"
for spec in "cnn_mnist 61" "cnn_no_dropout 62" "cnn_deep 63" "cnn_wide 64" "mlp 65"; do
  set -- $spec
  echo "RUN FINAL DIRTY CUCKOO model=$1 seed=$2"
  python -u federated.py $COMMON --model_name=$1 --seed=$2
done
