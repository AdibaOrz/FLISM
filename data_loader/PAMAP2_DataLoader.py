import sys
import numpy as np
import torch.utils.data
from configs import PAMAP2_Opt


sys.path.append('../')
WIN_LEN = PAMAP2_Opt['window_len']


class PAMAP2_Dataloader(torch.utils.data.Dataset):
    def __init__(self, is_train, df, mean = None, std = None):
        self.is_train = is_train
        self.df = df
        self.mean, self.std = mean, std
        if self.is_train:
            self.mean, self.std = self.extract_stats_from_df()
        self.process_dataframe(self.mean, self.std)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.dataset[idx]  # (feature, class_label, user_label)


    def get_user_groups(self):
        return self.user_groups


    def extract_stats_from_df(self):
        column_name_index_dict = self.get_column_name_index_dict()
        FEAT_IDX_START = column_name_index_dict['feat_start']
        FEAT_IDX_END = column_name_index_dict['feat_end']
        feature_dataframe = self.df.iloc[:, FEAT_IDX_START: (FEAT_IDX_END + 1)]
        mean_vec = feature_dataframe.mean()
        std_vec = feature_dataframe.std()
        return mean_vec, std_vec


    def process_dataframe(self, mean_from_train=None, std_from_train=None):
        features = []
        class_labels = []
        users = []
        datasets = []

        column_name_index_dict = self.get_column_name_index_dict()

        USER_IDX = column_name_index_dict['user_idx']
        LABEL_IDX = column_name_index_dict['label']
        FEAT_IDX_START = column_name_index_dict['feat_start']
        FEAT_IDX_END = column_name_index_dict['feat_end']
        OVERLAPPING_WIN_LEN = WIN_LEN

        feature_dataframe = self.df.iloc[:, FEAT_IDX_START: (FEAT_IDX_END + 1)]
        # Standardize the train dataset
        self.df.iloc[:, FEAT_IDX_START: (FEAT_IDX_END + 1)] = self.standardize_features(feature_dataframe, mean_from_train, std_from_train)
        # print(f'mean of columns : {self.df.mean()}, std of columns : {self.df.std()}\n')

        unique_users = self.df.user_idx.unique().tolist()
        for idx in range(max(len(self.df) // OVERLAPPING_WIN_LEN, 0)):
            user = self.df.iloc[idx * OVERLAPPING_WIN_LEN, USER_IDX]
            label = self.df.iloc[idx * OVERLAPPING_WIN_LEN, LABEL_IDX]
            feature = self.df.iloc[idx*OVERLAPPING_WIN_LEN: (idx + 1) * OVERLAPPING_WIN_LEN,  FEAT_IDX_START:(FEAT_IDX_END + 1)].values
            feature = feature.T

            features.append(feature)
            class_labels.append(label)
            users.append(user)

        features = np.array(features, dtype=np.float)
        class_labels = np.array(class_labels)
        users = np.array(users)

        class_labels[np.isnan(class_labels)] = 0
        self.user_groups = {}
        for unq_user in unique_users:
            indices = np.where(users == unq_user)[0]
            self.user_groups[unq_user] = indices
            datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(features[indices]).float(),
                                                                torch.from_numpy(class_labels[indices]),
                                                                torch.from_numpy(users[indices])))
        self.dataset = torch.utils.data.ConcatDataset(datasets)


    def standardize_features(self, features, mean_from_train, std_from_train):
        features[np.isnan(features)] = 0
        for column in mean_from_train.index:
            features.loc[:, [column]] = ((features.loc[:, [column]] - mean_from_train[column]) / std_from_train[column])

        return features


    def get_column_name_index_dict(self):
        dict = {}
        column_list = self.df.columns.tolist()
        for col_idx, col_name in enumerate(column_list):
            if col_name in ['user_idx', 'label']:
                dict[col_name] = col_idx

        dict['feat_start'] = dict['label'] + 1
        dict['feat_end'] = dict['user_idx'] - 1
        return dict
