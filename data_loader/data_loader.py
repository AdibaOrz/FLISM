import os
import sys
import json
import torch
import pandas as pd
import torch.utils.data

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader.WESAD_DataLoader import WESAD_Dataloader
from data_loader.PAMAP2_DataLoader import PAMAP2_Dataloader
from data_loader.SleepEDF20_DataLoader import SleepEDF20_DataLoader
from data_loader.RealWorldHAR_DataLoader import RealWorldHAR_Dataloader


def load_train_test_datasets(dataset_name):
    json_dataset_paths = './data_loader/dataset_paths.json'
    with open(json_dataset_paths, 'r') as f:
        dataset_paths = json.load(f)
    dataset_paths = dataset_paths[dataset_name]

    if dataset_name == "pamap2":
        train_df, test_df = split_to_train_and_test(dataset_name, dataset_paths['all'], train_ratio=0.7)
        train_dataset = PAMAP2_Dataloader(is_train=True, df=train_df, mean=None, std=None)
        test_dataset = PAMAP2_Dataloader(is_train=False, df=test_df, mean=train_dataset.mean, std=train_dataset.std)

    elif dataset_name == "realworldhar":
        train_df, test_df = split_to_train_and_test(dataset_name, dataset_paths['all'], train_ratio=0.7)
        train_dataset = RealWorldHAR_Dataloader(is_train=True, df=train_df, mean=None, std=None)
        test_dataset = RealWorldHAR_Dataloader(is_train=False, df=test_df, mean=train_dataset.mean, std=train_dataset.std)

    elif dataset_name == 'sleepedf20':
        train_dataset = SleepEDF20_DataLoader(is_train=True)
        test_dataset = SleepEDF20_DataLoader(is_train=False)

    elif dataset_name == "wesad":
        train_path, test_path = dataset_paths['train'], dataset_paths['test']
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_dataset = WESAD_Dataloader(is_train=True, df=train_df)
        test_dataset = WESAD_Dataloader(is_train=False, df=test_df)

    train_user_groups = train_dataset.get_user_groups()
    test_user_groups = test_dataset.get_user_groups()
    return train_dataset, test_dataset, train_user_groups, test_user_groups


def split_to_train_and_test(dataset_name, file_path_to_dataset, train_ratio):
    assert dataset_name in ['realworldhar', 'pamap2']
    all_df = pd.read_csv(file_path_to_dataset)
    unique_users = all_df['user_idx'].unique().tolist()
    train_df_all, test_df_all = None, None

    for user_label in unique_users:
        cur_user_df = all_df[all_df['user_idx'] == user_label]
        unique_labels = cur_user_df['label'].unique().tolist()

        for cur_label in unique_labels:
            cur_user_cur_label_df = cur_user_df[cur_user_df['label'] == cur_label]
            total_num_rows_cur_label = len(cur_user_cur_label_df)
            num_train_rows = int(total_num_rows_cur_label * train_ratio)

            cur_train_df = cur_user_cur_label_df.iloc[:num_train_rows]
            cur_test_df = cur_user_cur_label_df.iloc[num_train_rows:]

            train_df_all = pd.concat([train_df_all, cur_train_df]) if train_df_all is not None else cur_train_df
            test_df_all = pd.concat([test_df_all, cur_test_df]) if test_df_all is not None else cur_test_df

    return train_df_all, test_df_all



class DatasetSplit(torch.utils.data.Dataset):
    """To extract data of a specific user from the dataset."""
    def __init__(self, all_dataset, curr_user_indices, dataset_name):
        self.all_dataset = all_dataset  # data of all clients
        self.curr_user_indices = [int(i) for i in curr_user_indices]  # indices corresponding to this user only
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.curr_user_indices)

    def __getitem__(self, item):
        if self.dataset_name == 'sleepedf20':
            _, input_data, class_label, user_label = self.all_dataset[self.curr_user_indices[item]]
        else:
            input_data, class_label, user_label = self.all_dataset[self.curr_user_indices[item]]
        return input_data.clone().detach(), class_label.clone().detach()
