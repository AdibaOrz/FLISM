import os
import argparse
from glob import glob
import fnmatch
import re
import json
import statistics
# from arg_parser import parse_arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--metric', type=str, default='comm')
    parser.add_argument('--note', type=str, default='')
    args = parser.parse_args()
    return args


SEED = [1000, 1001, 1002, 1003, 1004]
NUM_MODALITIES = [5, 10, 15, 20, 25, 30]
def run(args):
    if args.metric == 'comm':
        metric = 'total_communication_cost'
    else:
        metric = 'total_number_of_trained_parameters'

    temp = {}
    result = {}

    # folders = os.listdir(args.path)
    folders = glob(os.path.join(args.path, f'*{args.note}'))

    method = args.path.split('/')[-3]

    for cur_num_modal_folder in folders:
        cur_num_modal_folder_name  = os.path.join(args.path, cur_num_modal_folder)
        seeds_folders = os.listdir(cur_num_modal_folder_name)
        for final_folder in seeds_folders:
            file = glob(os.path.join(cur_num_modal_folder_name, final_folder, '*.json'))[0]
            seed_info = file.split('/')[-2]
            parameters = seed_info.split('_')

            for param in parameters:
                if param.startswith('s:'):
                    s_value = int(param.split(':')[1])

            num_mod_info = file.split('/')[-3]
            parameters = num_mod_info.split('_')
            for param in parameters:
                if param.startswith('snom:'):
                    num_mod_value = int(param.split(':')[1])

            if num_mod_value not in temp:
                temp[num_mod_value] = []
                result[num_mod_value] = None

            with open(file, 'r') as f:
                content = json.load(f)
                score = float(content[metric])
                temp[num_mod_value].append(score)

    for num_mod_value in NUM_MODALITIES:
        try:
            result[num_mod_value] = sum(temp[num_mod_value]) / len(temp[num_mod_value])
        except:
            result[num_mod_value] = -1

    print(f"Result of Method {method}")
    for num_mod, average in result.items():
        print(f"{average:.2f}", end=" ")
    print("")

if __name__ == '__main__':
    args = parse_args()
    # main_args = parse_arguments()
    run(args)