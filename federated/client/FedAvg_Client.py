"""
Baseline Method
Client-side code
=> Uses zero-imputation for missing modalities.
=> Does not apply any other methods to handle missing modalities.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils.missing_utils import drop_modalities
from data_loader.data_loader import DatasetSplit


class FedAvg_Client(object):
    def __init__(self, client_id, train_dataset, test_dataset, client_train_data_indices, client_test_data_indices,
                 modality_indices_to_drop, device, args):
        self.client_id = client_id
        self.device = device
        self.args = args

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_train_data_indices = client_train_data_indices
        self.client_test_data_indices = client_test_data_indices
        self.modality_indices_to_drop = modality_indices_to_drop

        self.train_loader = DataLoader(DatasetSplit(self.train_dataset, self.client_train_data_indices, args.dataset),
                                      batch_size=args.local_bs, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(DatasetSplit(self.test_dataset, self.client_test_data_indices, args.dataset),
                                      batch_size=self.args.local_bs, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


    def local_train(self, model):
        model = model.to(self.device)
        model.train()

        epoch_loss = []
        local_epochs = self.args.local_ep
        self.optimizer = self.get_optimizer(model)

        for local_epoch in range(local_epochs):
            batch_loss = []
            for batch_idx, (input_data, labels) in enumerate(self.train_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()

                self.optimizer.zero_grad()
                input_data = drop_modalities(self.args.dataset, input_data, self.modality_indices_to_drop)
                model_outputs = model(input_data, extract_embeds_only=False)
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            # Finished one mini-batch training
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        avg_train_loss = sum(epoch_loss) / len(epoch_loss)
        return model, avg_train_loss


    def local_test(self, model):
        model = model.to(self.device)
        model.eval()

        correct_labels_all, predicted_labels_all = [], []
        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(self.test_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                outputs = model(input_data, extract_embeds_only=False)
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                predicted_labels_all = np.concatenate((predicted_labels_all, pred_labels.cpu()), axis=0)
                correct_labels_all = np.concatenate((correct_labels_all, labels.cpu()), axis=0)
        # Calculate test f1-score
        f1_macro = f1_score(correct_labels_all, predicted_labels_all, average='macro')
        return f1_macro


    def get_optimizer(self, model):
        # Set optimizer
        optimizer_name = self.args.optim
        lr = self.args.local_lr
        weight_decay = self.args.local_wd
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer')
        return optimizer