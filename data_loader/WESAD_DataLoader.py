import os
import sys
import torch
import numpy as np
import torch.utils.data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import WESAD_Opt

WIN_LEN = WESAD_Opt['window_len']
OVERLAP_ROWS = int(WESAD_Opt['sampling_rate']//2)
OVERLAPPING_WIN_LEN = WIN_LEN - OVERLAP_ROWS
FEAT_IDX_START, FEAT_IDX_END = 0, len(WESAD_Opt['feature_columns']) - 1 # 0, 9


class WESAD_Dataloader(torch.utils.data.Dataset):
    def __init__(self, is_train, df):
        self.is_train = is_train
        self.df = df
        self.process_dataframe()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.dataset[idx]


    def get_user_groups(self):
        return self.user_groups


    def process_dataframe(self):
        features = []
        class_labels = []
        user_labels = []
        datasets = []


        unique_users = self.df['user_id'].unique().tolist()
        unique_activities = self.df['label'].unique().tolist()
        stats_dict = {}
        for user_id in unique_users:
            for activity in unique_activities:
                # print(f'Handling user: {user_id} and activity: {activity}')
                curr_df = self.df[(self.df['user_id'] == user_id) & (self.df['label'] == activity)]
                windowed_features = self.get_windowed_features(curr_df)
                stats_dict[f'{user_id}_{activity}'] = len(windowed_features)
                features.extend(windowed_features)

                len_data = len(windowed_features)
                class_labels.extend([int(activity)] * len_data)
                user_id_int = int(user_id[1:])
                user_labels.extend([user_id_int] * len_data)

        # print(f'type of features: {type(features)}, len features: {len(features)}')
        features = np.array(features, dtype=np.float)
        class_labels = np.array(class_labels)
        user_labels = np.array(user_labels)

        # Create dataset and user groups
        self.user_groups = {}
        for user_id in unique_users:
            user_id_int = int(user_id[1:])
            indices = np.where(user_labels == user_id_int)[0]
            self.user_groups[user_id_int] = indices
            datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(features[indices]).float(),
                                                           torch.from_numpy(class_labels[indices]).long(),
                                                           torch.from_numpy(user_labels[indices]).long()
            ))
        self.dataset = torch.utils.data.ConcatDataset(datasets)
        # print(f"Stats: {stats_dict}")


    def get_windowed_features(self, curr_df):
        windowed_features = []
        for idx in range(max(len(curr_df) // OVERLAPPING_WIN_LEN, 0)):
            window_start_idx = idx * OVERLAPPING_WIN_LEN
            window_end_idx = (idx * OVERLAPPING_WIN_LEN) + WIN_LEN
            if window_end_idx >= len(curr_df):
                break
            # print(f'[{idx}/N], window_start_idx: {window_start_idx}, window_end_idx: {window_end_idx}')
            feature_window = curr_df.iloc[window_start_idx:window_end_idx, FEAT_IDX_START:FEAT_IDX_END + 1].values
            feature_window = feature_window.T
            windowed_features.append(feature_window)
        return windowed_features

