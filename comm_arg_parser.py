import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="System Efficiency Experiments")

    # ========================= General ========================== #
    # ============================================================ #
    parser.add_argument("--dataset",  type=str, default="wesad", choices=["wesad", "realworldhar", "pamap2", "sleepedf20"])
    # parser.add_argument("--domain", type=str, default="sensor")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--method", type=str, default='fedavg', choices=["fedavg", "fedprox", "moon",
                                                                         "harmony", "fedmultimodal", "autofed", "flism"
                                                                         ])
    parser.add_argument("--drop_p", type=float, default=0.0, help='drop percentage')
    parser.add_argument("--percentage_selected_clients_per_round", type=float, default=0.3)
    parser.add_argument("--acc_gyro_as_1_modal", action="store_true", help="Whether to treat acc+gyro as 1 modality")

    parser.add_argument("--simulated", action="store_true",
                        help="Whether to simulate the clients & modalities")
    parser.add_argument("--simulated_clients_number", type=int, default=100)
    parser.add_argument("--simulated_number_of_modalities", type=int, default=5)

    parser.add_argument("--fl_rounds", type=int, default=100)
    parser.add_argument("--hrm_stg1_fl_rounds", type=int, default=50,
                        help="Number of FL rounds in Stage 1 of Harmony")
    parser.add_argument("--hrm_stg2_fl_rounds", type=int, default=50,
                        help="Number of FL rounds in Stage 2 of Harmony")
    parser.add_argument("--hrm_stg1_use_all_clients", action="store_true",  # assumption of Harmony, not realistic
                        help="Assume that all clients are available all the time to conduct Stage 1")


    # ============================================================ #
    # ======================= Other Args ========================= #
    # ============================================================ #
    parser.add_argument("--note", type=str, default='')
    parser.add_argument("--debug", action="store_true",
                        help="Whether to print the debugging messages while training/inference")
    parser.add_argument("--print_every", type=int, default=1, help="Print the training stats every X FL rounds")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate the model every X FL rounds")

    parser.add_argument("--verbose", action="store_true", help="Set to print more information")

    args = parser.parse_args()

    print_args(args)

    args.hrm_stg1_fl_rounds = int(
        2 / 3 * args.fl_rounds)  # same as the total global rounds, considering the unimodal clients
    args.hrm_stg2_fl_rounds = int(args.fl_rounds - args.hrm_stg1_fl_rounds)  # same as the total global rounds//2, considering the unimodal clients

    # Argument grouping
    arg_meta_dict = {
        'general': [
            ('seed', 's'),
            ('drop_p', 'dp'),
            ('percentage_selected_clients_per_round', 'm'),
        ],
        'moon': [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
        'fedmm': [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
        'fedavg' : [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
        'fedprox': [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
        'harmony': [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
        'flism': [
            ('simulated_clients_number', 'scn'),
            ('simulated_number_of_modalities', 'snom'),
        ],
    }

    if args.simulated:
        args.dataset = 'simulated'


    # METHOD_LIST = ['moon', 'fedavg', 'supcon', 'harmony']
    file_name = {key : "" for key in arg_meta_dict.keys()}
    for group_name, list_of_args in arg_meta_dict.items():
        for item_name, item_short_name in list_of_args:
            if hasattr(args, item_name):
            # if vars(args).get(item_name):
                arg_value = getattr(args, item_name)
                file_name[group_name] +=f"{item_short_name}:{arg_value}_"
            else:
                print(f"No arg with {item_name}")
    # print(f"file_name for all group: {file_name}")

    final_general_filename = file_name["general"][:-1]

    if args.method not in file_name.keys():
        final_filename = file_name["fedavg"][:-1]
    else:
        final_filename = file_name[args.method][:-1]
    print(f"===== Final general filename: {final_general_filename}")
    print(f"===== Final filename for {args.method}: {final_filename}")

    args.log_path = f"/mnt/sting/jaehyun/flism/logs/print_logs/communication/{args.method}/{args.dataset}"
    args.json_log_path = f"/mnt/sting/jaehyun/flism/logs/json_logs/communication/{args.method}/{args.dataset}"


    args.final_filename = final_filename
    args.final_general_filename = final_general_filename

    return args


def print_args(args):
    print("==========================================")
    print("=========== Passed Arguments =============")
    print("==========================================")
    for idx, (arg, value) in enumerate((args.__dict__.items())):
        print(f"[{idx}]: {arg.upper()}:{value}")
    print("\n")

if __name__ == "__main__":
    parse_arguments()