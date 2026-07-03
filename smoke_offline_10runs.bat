@echo off
setlocal enabledelayedexpansion

echo START OFFLINE FINAL CUCKOO 5NN SMOKE TEST
echo This runs 10 local smoke tests: 5 dirty + 5 clean PGD, 2 rounds each.
echo.

set MODELS=cnn_residual_se cnn_vgg_bn cnn_inception cnn_depthwise_se mlp_mixer

set DIRTY_COMMON=--data=fmnist --device=cpu --local_ep=1 --bs=1024 --num_agents=10 --rounds=2 --snap=1 --num_corrupt=1 --poison_frac=0.10 --robustLR_threshold=1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poisoning=1 --pattern_type=plus --cuckoo=1 --cuckoo_variant=final_hybrid --cuckoo_lambda=1.8 --cuckoo_incubation_lambda=0.0 --cuckoo_hatch_round=1 --cuckoo_warmup_rounds=1 --cuckoo_schedule=step --cuckoo_norm_cap=3.0 --cuckoo_min_cosine=-1.0 --cuckoo_top_frac=0.70 --cuckoo_budget_mode=global --cuckoo_layer_policy=classifier --cuckoo_sign_align=1 --cuckoo_classifier_target_rows=1 --cuckoo_classifier_focus=target_base --cuckoo_classifier_row_relax=0.35 --cuckoo_classifier_row_boost=4.0 --cuckoo_classifier_bias_boost=2.0 --cuckoo_center_egg=0

set CLEAN_COMMON=--data=fmnist --device=cpu --local_ep=1 --bs=1024 --num_agents=10 --rounds=2 --snap=1 --num_corrupt=2 --poison_frac=1.0 --robustLR_threshold=1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=4 --clean_label_adv=1 --pgd_eps=24 --pgd_alpha=2 --pgd_steps=2 --surrogate_epochs=1 --verify_poisoning=1 --pattern_type=plus --cuckoo=1 --cuckoo_variant=clean_label_egg --cuckoo_lambda=2.0 --cuckoo_incubation_lambda=0.0 --cuckoo_hatch_round=1 --cuckoo_warmup_rounds=1 --cuckoo_schedule=step --cuckoo_norm_cap=3.2 --cuckoo_min_cosine=-1.0 --cuckoo_top_frac=0.75 --cuckoo_budget_mode=global --cuckoo_layer_policy=classifier --cuckoo_sign_align=1 --cuckoo_classifier_target_rows=1 --cuckoo_classifier_focus=target_base --cuckoo_classifier_row_relax=0.40 --cuckoo_classifier_row_boost=4.5 --cuckoo_classifier_bias_boost=2.2 --cuckoo_center_egg=0 --cuckoo_pgd_bias=1.35

set SEED=901

echo ============================================================
echo DIRTY CUCKOO OFFLINE SMOKE: 5 MODELS
echo ============================================================

for %%M in (%MODELS%) do (
    echo.
    echo RUN DIRTY OFFLINE SMOKE model=%%M seed=!SEED!
    python -u federated.py %DIRTY_COMMON% --model_name=%%M --seed=!SEED!
    if errorlevel 1 (
        echo FAILED DIRTY model=%%M
        exit /b 1
    )
    set /a SEED+=1
)

echo.
echo ============================================================
echo CLEAN PGD BIG-SQUARE CUCKOO OFFLINE SMOKE: 5 MODELS
echo ============================================================

for %%M in (%MODELS%) do (
    echo.
    echo RUN CLEAN PGD OFFLINE SMOKE model=%%M seed=!SEED!
    python -u federated.py %CLEAN_COMMON% --model_name=%%M --seed=!SEED!
    if errorlevel 1 (
        echo FAILED CLEAN model=%%M
        exit /b 1
    )
    set /a SEED+=1
)

echo.
echo DONE OFFLINE FINAL CUCKOO 5NN SMOKE TEST
pause