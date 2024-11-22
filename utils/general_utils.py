import os
import json
import torch
import random
import numpy as np


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



def log_test_results_to_json(individual_test_results, avg_f1_macro, args):
    """Log the test results to a JSON file."""
    log_file_path = f'{args.json_log_path}/{args.final_filename}_{args.note}/{args.final_general_filename}'
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    note_str = f"{args.note}_" if args.note != "" else ""
    full_path_to_json_file = f"{log_file_path}/{note_str}test_results.json"
    all_test_results = {
        'avg_f1_macro': avg_f1_macro,
        'individual_test_results': individual_test_results}

    with open(full_path_to_json_file, 'w') as f:
        json.dump(all_test_results, f, indent=4)
    print(f"====Saved the results to {full_path_to_json_file}====")


def print_header(message, decorator_char="="):
    len_message = len(message)
    top_bottom_decoration = decorator_char * len_message + decorator_char * 12
    msg_decoration = f"{decorator_char * 5} {message} {decorator_char * 5}"
    print(top_bottom_decoration)
    print(msg_decoration)
    print(top_bottom_decoration)
