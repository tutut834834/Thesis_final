import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # ============================================================
    # Dataset / model
    # ============================================================
    parser.add_argument('--data', type=str, default='fmnist',
                        choices=['fmnist', 'fedemnist', 'cifar10'])

    parser.add_argument('--model_name', type=str, default='cnn_residual_se',
                        choices=[
                            'cnn_residual_se',
                            'cnn_vgg_bn',
                            'cnn_inception',
                            'cnn_depthwise_se',
                            'mlp_mixer'
                        ],
                        help='Large 5NN architecture test models.')

    parser.add_argument('--seed', type=int, default=1)

    # ============================================================
    # Federated learning setup
    # ============================================================
    parser.add_argument('--num_agents', type=int, default=10)
    parser.add_argument('--agent_frac', type=float, default=1.0)
    parser.add_argument('--class_per_agent', type=int, default=10)
    parser.add_argument('--num_corrupt', type=int, default=0)

    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--aggr', type=str, default='avg',
                        choices=['avg', 'comed', 'sign'])

    parser.add_argument('--local_ep', type=int, default=1)
    parser.add_argument('--corrupt_local_ep', type=int, default=0)

    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--client_lr', type=float, default=0.1)
    parser.add_argument('--client_moment', type=float, default=0.9)
    parser.add_argument('--server_lr', type=float, default=1.0)

    # ============================================================
    # Backdoor setup
    # ============================================================
    parser.add_argument('--base_class', type=int, default=5)
    parser.add_argument('--target_class', type=int, default=7)

    parser.add_argument('--poison_frac', type=float, default=0.0)

    parser.add_argument('--pattern_type', type=str, default='plus',
                        choices=['plus', 'square', 'big_square', 'apple', 'copyright'])

    # ============================================================
    # Clean-label / PGD options
    # ============================================================
    parser.add_argument('--clean_label', type=int, default=0,
                        help='0 = dirty-label attack, 1 = clean-label attack')

    parser.add_argument('--clean_label_type', type=int, default=0,
                        help='0 dirty/default, 1 plus, 2 square, 3 apple, 4 big_square')

    parser.add_argument('--clean_label_auto_pattern', type=int, default=1,
                        help='1 = clean_label_type controls pattern automatically')

    parser.add_argument('--verify_poisoning', type=int, default=1)

    parser.add_argument('--clean_label_adv', type=int, default=0,
                        help='1 = PGD clean-label perturbation before trigger stamping')

    parser.add_argument('--pgd_eps', type=float, default=16.0)
    parser.add_argument('--pgd_alpha', type=float, default=2.0)
    parser.add_argument('--pgd_steps', type=int, default=10)
    parser.add_argument('--surrogate_epochs', type=int, default=5)

    # ============================================================
    # Defense / RobustLR / privacy style options
    # ============================================================
    parser.add_argument('--robustLR_threshold', type=int, default=0)
    parser.add_argument('--clip', type=float, default=0.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--top_frac', type=int, default=100)

    # ============================================================
    # Logging / device
    # ============================================================
    parser.add_argument('--snap', type=int, default=1)
    parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--num_workers', type=int, default=0)

    # ============================================================
    # Final Cuckoo biology-inspired options
    # ============================================================
    parser.add_argument('--cuckoo', type=int, default=0,
                        help='1 = enable final Cuckoo clean-host + poisoned-egg update')

    parser.add_argument('--cuckoo_variant', type=str, default='final_hybrid',
                        choices=['final_hybrid', 'classifier_nest', 'clean_label_egg', 'multi_cuckoo', 'v2'])

    parser.add_argument('--cuckoo_lambda', type=float, default=1.8)
    parser.add_argument('--cuckoo_incubation_lambda', type=float, default=0.0)

    parser.add_argument('--cuckoo_hatch_round', type=int, default=12)
    parser.add_argument('--cuckoo_warmup_rounds', type=int, default=12)

    parser.add_argument('--cuckoo_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step'])

    parser.add_argument('--cuckoo_norm_cap', type=float, default=3.0,
                        help='Relaxed because previous multi-cuckoo over-constrained and lost the backdoor.')

    parser.add_argument('--cuckoo_min_cosine', type=float, default=-1.0)
    parser.add_argument('--cuckoo_sign_align', type=int, default=1)
    parser.add_argument('--cuckoo_allow_zero_host_sign', type=int, default=1)

    parser.add_argument('--cuckoo_top_frac', type=float, default=0.70,
                        help='Keep more poisoned egg residual coordinates than earlier failed variants.')

    parser.add_argument('--cuckoo_budget_mode', type=str, default='global',
                        choices=['global', 'layerwise'])

    parser.add_argument('--cuckoo_layer_policy', type=str, default='classifier',
                        choices=['all', 'classifier', 'last', 'no_conv', 'conv_only'])

    parser.add_argument('--cuckoo_mimic_mode', type=str, default='sign',
                        choices=['sign', 'none', 'raw'])

    parser.add_argument('--cuckoo_center_egg', type=int, default=0,
                        help='0 = do not remove too much poisoned residual.')

    parser.add_argument('--cuckoo_blend_clean_direction', type=float, default=0.03)
    parser.add_argument('--cuckoo_diag_every', type=int, default=5)

    # ============================================================
    # Classifier-row Cuckoo focus
    # ============================================================
    parser.add_argument('--cuckoo_classifier_target_rows', type=int, default=1)

    parser.add_argument('--cuckoo_classifier_focus', type=str, default='target_base',
                        choices=['target_base', 'target', 'base', 'none'])

    parser.add_argument('--cuckoo_classifier_row_relax', type=float, default=0.35,
                        help='Lets target/base classifier rows pass even when sign mimicry would suppress them.')

    parser.add_argument('--cuckoo_classifier_boost', type=float, default=2.5)
    parser.add_argument('--cuckoo_classifier_row_boost', type=float, default=4.0)
    parser.add_argument('--cuckoo_classifier_bias_boost', type=float, default=2.0)

    parser.add_argument('--cuckoo_pgd_bias', type=float, default=1.25,
                        help='Extra residual strength for clean-label PGD big-square egg runs.')

    args = parser.parse_args()
    return args