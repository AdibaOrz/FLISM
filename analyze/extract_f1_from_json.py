import os
import json
import fnmatch
import argparse
from glob import glob

CLIENT_PERCENTAGE = {
    "pamap2": 0.5,
    "realworldhar": 0.3,
    "sleepedf20" : 0.3,
    "wesad": 0.3,
}
DROP_PERCENTAGE = [0, 0.4, 0.6, 0.8]
SEED = [1000, 1001, 1002, 1003, 1004]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    return parser.parse_args()



def run(args):
    res = {x : {y : -1 for y in SEED} for x in DROP_PERCENTAGE}
    dataset = args.path.split('/')[-2]
    client_percentage = CLIENT_PERCENTAGE[dataset]

    folders = os.listdir(args.path)
    folders = [folder for folder in folders if fnmatch.fnmatch(folder, f'*m:{client_percentage}*')]

    for folder in folders:
        file = glob(os.path.join(args.path, folder, '*.json'))[0]
        general_info = file.split('/')[-2]
        parameters = general_info.split('_')

        for param in parameters:
            if param.startswith('s:'):
                s_value = int(param.split(':')[1]) # seed
            elif param.startswith('dp:'):
                dp_value = float(param.split(':')[1]) # drop percentage

        with open(file, 'r') as f:
            content = json.load(f)
            try:
                score = float(content['avg_f1_macro'])
            except:
                score = -1
            try:
                res[dp_value][s_value] = score
            except:
                pass

    print(f"Result of Drop percentage : 0% 40% 60% 80% 100%\n")
    for drop_p in res.keys():
        sum = 0
        for seed, value in res[drop_p].items():
            sum += value
            print(f"{value:.2f}", end=" ")
        print('')


if __name__ == '__main__':
    args = parse_args()
    run(args)