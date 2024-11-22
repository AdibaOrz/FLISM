import sys
import json
import torch.utils.data

sys.path.append('../')
class SleepEDF20_DataLoader(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.dataset, self.user_groups = self.load_dataset_and_user_groups()
        self.user_groups = self.get_user_groups()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]


    def get_user_groups(self):
        return self.user_groups


    def load_dataset_and_user_groups(self):
        import os
        print(os.getcwd())
        json_dataset_paths = "./data_loader/dataset_paths.json"
        with open(json_dataset_paths, 'r') as f:
            dataset_paths = json.load(f)
        dataset_paths = dataset_paths['sleepedf20']

        if self.is_train:
            dataset_path = dataset_paths['train']
            user_groups_path = dataset_paths['train_user_groups']
        else:
            dataset_path = dataset_paths['test']
            user_groups_path = dataset_paths['test_user_groups']
        try:
            dataset = torch.load(dataset_path)
            with open(user_groups_path, 'r') as f:
                user_groups = json.load(f)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            dataset, user_groups = None, None


        return dataset, user_groups