
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
    args = parser.parse_args()
    return args

CLIENT_PERCENTAGE = {
    "wesad" : 0.3,
    "realworldhar" : 0.3,
    "pamap2" : 0.5,
    "sleepedf20" : 0.3,
}

DROP_PERCENTAGE = [0.4, 0.6, 0.8]
SEED = [1000, 1001, 1002, 1003, 1004]
def run(args):

    res = {x : {y : -1 for y in SEED} for x in DROP_PERCENTAGE}
    dataset = args.path.split('/')[-3]
    client_percentage = CLIENT_PERCENTAGE[dataset]

    folders = os.listdir(args.path)
    folders = [folder for folder in folders if fnmatch.fnmatch(folder, f'*m:{client_percentage}*')]


    for folder in folders:
        file = glob(os.path.join(args.path, folder, '*.json'))[0]
        general_info = file.split('/')[-2]
        parameters = general_info.split('_')

        for param in parameters:
            if param.startswith('s:'):
                s_value = int(param.split(':')[1])
            elif param.startswith('dp:'):
                dp_value = float(param.split(':')[1])

        # print(s_value, dp_value)
        # print(domain)
        with open(file, 'r') as f:
            content = json.load(f)

            try:
                score = float(content['total_communication_cost'])
            except:
                score = -1

            try:
                res[dp_value][s_value] = score
            except:
                pass

    print(f"Result of Communication cost : 40% 60% 80%\n")
    final_avg, final_std = [], []
    for drop_p in res.keys():
        # print(f"Result of Drop percentage : {drop_p * 100}%")
        sum = 0
        for seed, value in res[drop_p].items():
            sum += value
            print(f"{value:.2f}", end=" ")
        final_avg.append(sum / len(res[drop_p]))
        final_std.append(statistics.stdev(res[drop_p].values()))
        print(f"Average : {sum / len(res[drop_p]):.2f} Std : {statistics.stdev(res[drop_p].values())}\n")

    for value in final_avg:
        print(f"{value:.2f}", end=" ")
    for value in final_std:
        print(f"{value:.2f}", end=" ")
    print("")

if __name__ == '__main__':
    args = parse_args()
    # main_args = parse_arguments()
    run(args)