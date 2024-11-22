import os
import argparse
from constants import LOG_PATH, PRETRAINED_PATH
from configs import get_dataset_opts, get_moon_training_params


def parse_arguments():
    parser = argparse.ArgumentParser(description="FLISM Experiments")
    # ============================================================ #
    # ========================= General ========================== #
    # ============================================================ #
    parser.add_argument("--dataset", type=str, default="pamap2", choices=["wesad", "realworldhar", "pamap2", "sleepedf20"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedprox", "moon", "harmony",
                                                                         "fedmultimodal", "autofed", "flism"])
    parser.add_argument("--drop_p", type=float, default=0.0, help='drop percentage')
    parser.add_argument("--log_path", type=str, default=LOG_PATH) # Replace with your own path (ex: "./logs")
    parser.add_argument("--pretrained_path", type=str, default=PRETRAINED_PATH)  # Replace with your own path (ex: "./logs")

    # ============================================================ #
    # ===================== Method-Specific ====================== #
    # ============================================================ #

    # FL training settings
    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=32)
    parser.add_argument("--local_lr", type=float, default=0.001)
    parser.add_argument("--local_wd", type=float, default=0.001)
    parser.add_argument("--global_rnd", type=int, default=100)
    parser.add_argument("--client_selection_rate", type=float, default=0.3)

    # Baseline: MOON
    parser.add_argument("--moon_temp", type=float, default=0.5, help="Temperature for MOON")
    parser.add_argument("--moon_mu", type=float, default=5.0, help="Mu for MOON")

    # Baseline: Harmony
    parser.add_argument("--hrm_local_epochs", type=int, default=5,
                        help="Number of local epochs before server aggregation in Harmony")
    parser.add_argument("--hrm_stg1_fl_rounds", type=int, default=50,
                        help="Number of FL rounds in Stage 1 of Harmony")
    parser.add_argument("--hrm_stg2_fl_rounds", type=int, default=50,
                        help="Number of FL rounds in Stage 2 of Harmony")

    # Baseline:  AutoFed
    parser.add_argument("--autofed_gen_local_epochs", type=int, default=3,
                        help="Number of local epochs for autoencoder pre-training")
    parser.add_argument("--autofed_gen_fl_rounds", type=int, default=50,
                        help="Number of FL rounds for autoencoder pre-training")
    parser.add_argument('--is_generative', action='store_true', help='Whether to use imputation or not(FedAvg)')

    # Baseline: FedProx
    parser.add_argument("--prox_mu", type=float, default=0.001, help='the mu parameter for fedprox')


    # To simulate the clients and modalities with more modalities
    parser.add_argument("--simulated", action="store_true", help="Whether to simulate the clients & modalities")

    # =====================Our Method=========================== #
    # FLISM
    # --ver could be tested for ablation study at the same time
    parser.add_argument('--ver', type=str, default='none', choices=['none', 'supcon', 'supcon_wavg', 'supcon_wavg_kd'],
                        help='Version of FLISM')
    parser.add_argument('--kd_weight_mode', type=str, default='global_model', choices=['global_model'])
    parser.add_argument('--kd_coef', type=float, default=1.0,
                        help='The coefficient for the knowledge distillation loss')
    parser.add_argument("--autofed", action="store_true", help="Whether it is for comparison with AutoFed")

    # ============================================================ #
    # ======================= Other Args ========================= #
    # ============================================================ #
    parser.add_argument("--hps", action="store_true",
                        help="Whether to use the hyperparameters from the passed arguments")
    parser.add_argument("--note", type=str, default='')
    parser.add_argument("--verbose", action="store_true", help="Set to print more information")
    parser.add_argument("--print_every", type=int, default=1, help="Print the training stats every X FL rounds")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate the model every X FL rounds")

    args = parser.parse_args()
    dataset_opts = get_dataset_opts(args.dataset)
    # ======================================================= #
    # ========= Set dataset-specific FL hyperparameters ===== #
    if not args.hps: # default hyperparameters
        if args.method == 'moon':
            moon_training_opts = get_moon_training_params(args.dataset)
            args.optim = moon_training_opts['optim']
            args.local_ep = moon_training_opts['local_ep']
            args.local_bs = moon_training_opts['local_bs']
            args.local_lr = moon_training_opts['local_lr']
            args.local_wd = moon_training_opts['local_wd']
            args.global_rnd = moon_training_opts['global_rnd']
            args.client_selection_rate = moon_training_opts['client_selection_rate']
        else:
            args.local_ep = dataset_opts['local_ep']
            args.global_rnd = dataset_opts['global_rnd']
            args.local_bs = dataset_opts['local_bs']
            args.local_lr = dataset_opts['local_lr']
            args.local_wd = dataset_opts['local_wd']
            args.client_selection_rate = dataset_opts['client_selection_rate']
            args.optim = dataset_opts['optim']
            args.hrm_local_epochs = dataset_opts['local_ep'] # same as the FL local epochs
            args.hrm_stg1_fl_rounds = int(2/3 * dataset_opts['global_rnd']) # same as the total global rounds, considering the unimodal clients
            args.hrm_stg2_fl_rounds = int(dataset_opts['global_rnd'] - args.hrm_stg1_fl_rounds) # same as the total global rounds//2, considering the unimodal clients
        if args.method == 'harmony':
            print(f'// [{args.dataset}] HRM Stage 1 FL rounds: {args.hrm_stg1_fl_rounds}, HRM Stage 2 FL rounds: {args.hrm_stg2_fl_rounds} //')

    else: # use the hyperparameters passed as arguments
        args.hrm_local_epochs = args.local_ep
        args.hrm_stg1_fl_rounds = int(2/3 * args.global_rnd)
        args.hrm_stg2_fl_rounds = int(args.global_rnd - args.hrm_stg1_fl_rounds)

    # ======================================================= #
    # ======================================================= #
    args.dataset_opts = dataset_opts

    # Argument grouping
    arg_meta_dict = {
        'general': [
            ('seed', 's'),
            ('drop_p', 'dp'),
            ('optim', 'opt'),
            ('client_selection_rate', 'm')
        ],
        'moon': [
            ('global_rnd', 'grnd'),
            ('local_ep', 'ep'),
            ('local_bs', 'bs'),
            ('local_lr', 'lr'),
            ('local_wd', 'wd'),
            ('moon_temp', 't'),
            ('moon_mu', 'mu'),
        ],
        'fedmultimodal': [
            ('global_rnd', 'grnd'),
            ('local_ep', 'ep'),
            ('local_bs', 'bs'),
            ('local_lr', 'lr'),
            ('local_wd', 'wd'),
        ],
        'fedavg': [
            ('global_rnd', 'grnd'),
            ('local_ep', 'ep'),
            ('local_bs', 'bs'),
            ('local_lr', 'lr'),
            ('local_wd', 'wd'),
        ],
        'fedprox': [
            ('global_rnd', 'grnd'),
            ('local_ep', 'ep'),
            ('local_bs', 'bs'),
            ('local_lr', 'lr'),
            ('local_wd', 'wd'),
            ('prox_mu', 'pmu'),
        ],
        'harmony': [
            ('hrm_local_epochs', 'hle'),
            ('hrm_stg1_fl_rounds', 'h1fge'),
            ('hrm_stg2_fl_rounds', 'h2fge'),
        ],
        'autofed': [
            ('autofed_gen_local_epochs', 'gle'),
            ('autofed_gen_fl_rounds', 'gge'),
            ('is_generative', 'isgen'),
        ],
        'flism': [
            ('global_rnd', 'grnd'),
            ('local_ep', 'ep'),
            ('local_bs', 'bs'),
            ('local_lr', 'lr'),
            ('local_wd', 'wd'),
            ('ver', 'ver'),
            ('autofed', 'autofed')
        ]
    }

    if args.ver == 'supcon_wavg_kd':
        arg_meta_dict['flism'].append(('kd_weight_mode', 'kdw'))
        arg_meta_dict['flism'].append(('kd_coef', 'kdc'))

    folder = 'hps' if args.hps else ''

    file_name = {key: "" for key in arg_meta_dict.keys()}
    for group_name, list_of_args in arg_meta_dict.items():
        for item_name, item_short_name in list_of_args:
            if hasattr(args, item_name):
                arg_value = getattr(args, item_name)
                file_name[group_name] += f"{item_short_name}:{arg_value}_"
            else:
                print(f"No arg with {item_name}")

    final_general_filename = file_name["general"][:-1]

    if args.method not in file_name.keys():
        final_filename = file_name["fedavg"][:-1]
    else:
        final_filename = file_name[args.method][:-1]
    if args.verbose:
        print(f"===== General filename: {final_general_filename}")
        print(f"===== Filename for {args.method}: {final_filename}")

    # ======================= Log Paths ====================== #
    args.json_log_path = f"{args.log_path}/json_logs/{args.method}/{folder}/{args.dataset}"
    args.tensorboard_log_dir = f"{args.log_path}/tensorboard_logs/{args.method}/{folder}/{args.dataset}"
    args.model_save_dir = f"{args.log_path}/model_cpts/{args.method}{folder}/{args.dataset}"
    args.tsne_save_dir = f"{args.log_path}/tsne/{args.method}/{folder}/{args.dataset}"

    for path in [args.json_log_path, args.tensorboard_log_dir, args.model_save_dir, args.tsne_save_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            if args.verbose:
                print(f"Created directory: {path}")

    args.final_filename = final_filename
    args.final_general_filename = final_general_filename

    return args
