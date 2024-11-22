import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils.supcon_loss_util import SupConLoss
from utils.missing_utils import drop_modalities
from data_loader.data_loader import DatasetSplit


class FLISM_Client:
    def __init__(self, client_id, train_dataset, test_dataset, client_train_data_indices, client_test_data_indices,
                 modality_indices_to_drop, device, args):
        # General settings
        self.client_id = client_id
        self.device = device
        self.args = args

        # Dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_train_data_indices = client_train_data_indices
        self.client_test_data_indices = client_test_data_indices
        self.modality_indices_to_drop = modality_indices_to_drop
        self.has_ONE_modality = len(self.modality_indices_to_drop) == (self.args.dataset_opts['num_modalities'] - 1)

        # Train and test data Loaders
        self.train_loader = DataLoader(DatasetSplit(self.train_dataset, self.client_train_data_indices, args.dataset),
                                       batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(DatasetSplit(self.test_dataset, self.client_test_data_indices, args.dataset),
                                        batch_size=self.args.local_bs, shuffle=False)

        # Define loss function
        self.xent_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.supcon_loss_fn = SupConLoss(device=self.device, temperature=0.1, contrast_mode='all')
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

        # FL training
        self.optimizer = None
        self.local_epochs = self.args.local_ep


    def local_train(self, main_model, kd_model):
        metadata_dict = {}
        self.optimizer = self.get_optimizer(main_model)

        if self.args.ver in ['none', 'supcon', 'supcon_wavg']: # ver=='none' is equivalent to FedAvg
            locally_updated_model, loss_dict = self.local_train_without_kd(main_model)
        elif self.args.ver == 'supcon_wavg_kd':
            locally_updated_model, loss_dict = self.local_train_with_kd(main_model, kd_model)
        else:
            raise ValueError(f"Unknown version: {self.args.ver}")
        metadata_dict['train_loss'] = loss_dict

        # Server expects entropy on train data to perform weighted average
        if self.args.ver in ['supcon_wavg', 'supcon_wavg_kd']:
            entropy_on_train_data = self.calculate_entropy_after_model_update(main_model)
            metadata_dict['entropy_on_train_data'] = entropy_on_train_data
        return locally_updated_model, metadata_dict


    def local_train_without_kd(self, main_model):
        main_model = main_model.to(self.device)
        main_model.train()

        drop_indices_for_supcon = None
        if not self.has_ONE_modality and 'supcon' in self.args.ver: # Round-wise drop for SupCon training
            drop_indices_for_supcon = self.get_indices_to_drop_for_supcon(self.modality_indices_to_drop)

        epoch_cls_loss, epoch_supcon_loss, epoch_total_loss = [], [], []
        for epoch in range(self.local_epochs):
            batch_cls_loss, batch_supcon_loss, batch_total_loss = [], [], []
            for batch_idx, (input_data, labels) in enumerate(self.train_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                # Originally missing modalities
                input_data = drop_modalities(self.args.dataset, copy.deepcopy(input_data), self.modality_indices_to_drop)

                # (1) Classification
                classifier_outputs = main_model(input_data, extract_embeds_only=False)
                loss_cls = self.xent_loss_fn(classifier_outputs, labels)
                batch_cls_loss.append(loss_cls.item())

                # (2) SupCon
                if 'supcon' in self.args.ver and not self.has_ONE_modality:
                    input_data_for_supcon = copy.deepcopy(input_data)
                    input_data_for_supcon = drop_modalities(self.args.dataset, copy.deepcopy(input_data_for_supcon),
                                                            drop_indices_for_supcon)
                    input_data_for_supcon = self.add_gaussian_noise(input_data_for_supcon)
                    stacked_inputs = torch.cat((input_data, input_data_for_supcon), dim=0)
                    features = main_model(stacked_inputs, extract_embeds_only=True)
                    f1, f2 = torch.split(features, [self.args.local_bs, self.args.local_bs], dim=0)
                    f1, f2 = F.normalize(f1, dim=1), F.normalize(f2, dim=1)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss_supcon = self.supcon_loss_fn(features, labels)
                    batch_supcon_loss.append(loss_supcon.item())
                    total_loss = loss_cls + loss_supcon
                else:
                    total_loss = loss_cls # no supcon loss in this case

                # Backward and optimize
                total_loss.backward()
                self.optimizer.step()
                batch_total_loss.append(total_loss.item())

            epoch_cls_loss.append(np.mean(batch_cls_loss))
            epoch_supcon_loss.append(np.mean(batch_supcon_loss))
            epoch_total_loss.append(np.mean(batch_total_loss))

        # -> end of local update <- #
        loss_dict = {
            'cls_loss': np.average(epoch_cls_loss),
            'supcon_loss': np.average(epoch_supcon_loss) if len(epoch_supcon_loss) > 0 else np.NAN,
            'total_loss': np.average(epoch_total_loss)
        }
        return main_model, loss_dict


    def local_train_with_kd(self, main_model, kd_model):
        main_model = main_model.to(self.device)
        kd_model = kd_model.to(self.device)
        main_model.train()
        kd_model.eval()

        drop_indices_for_supcon = None
        if not self.has_ONE_modality and 'supcon' in self.args.ver:  # Round-wise drop for SupCon training
            drop_indices_for_supcon = self.get_indices_to_drop_for_supcon(self.modality_indices_to_drop)

        epoch_kd_loss, epoch_cls_loss, epoch_supcon_loss, epoch_total_loss = [], [], [], []
        for epoch in range(self.local_epochs):
            batch_kd_loss, batch_cls_loss, batch_supcon_loss, batch_total_loss = [], [], [], []
            for batch_idx, (input_data, labels) in enumerate(self.train_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                # Originally missing modalities
                input_data = drop_modalities(self.args.dataset, copy.deepcopy(input_data), self.modality_indices_to_drop)

                # (1) Classification
                classifier_outputs = main_model(input_data, extract_embeds_only=False)
                loss_cls = self.xent_loss_fn(classifier_outputs, labels)
                batch_cls_loss.append(loss_cls.item())

                # (2) KD
                with torch.no_grad():
                    kd_classifier_outputs = kd_model(input_data, extract_embeds_only=False)
                detached_kd_classifier_outputs = kd_classifier_outputs.detach()

                T = 1.0
                loss_kd = self.kd_loss_fn(F.log_softmax(classifier_outputs / T, dim=1),
                                          F.softmax(detached_kd_classifier_outputs / T, dim=1)) * (T * T)
                batch_kd_loss.append(loss_kd.item())

                # (2) SupCon
                if 'supcon' in self.args.ver and not self.has_ONE_modality:
                    input_data_for_supcon = copy.deepcopy(input_data)
                    input_data_for_supcon = drop_modalities(self.args.dataset, copy.deepcopy(input_data_for_supcon),
                                                            drop_indices_for_supcon)
                    input_data_for_supcon = self.add_gaussian_noise(input_data_for_supcon)
                    stacked_inputs = torch.cat((input_data, input_data_for_supcon), dim=0)  # (2B, seq_len, channels)

                    features = main_model(stacked_inputs, extract_embeds_only=True)
                    f1, f2 = torch.split(features, [self.args.local_bs, self.args.local_bs], dim=0)
                    f1, f2 = F.normalize(f1, dim=1), F.normalize(f2, dim=1)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss_supcon = self.supcon_loss_fn(features, labels)
                    batch_supcon_loss.append(loss_supcon.item())
                    # Calculate total loss
                    total_loss = loss_cls + loss_supcon + (self.args.kd_coef *  loss_kd)
                else:
                    total_loss = loss_cls + (self.args.kd_coef  * loss_kd)

                # Backward and optimize
                total_loss.backward()
                self.optimizer.step()
                batch_total_loss.append(total_loss.item())

            epoch_kd_loss.append(np.mean(batch_kd_loss))
            epoch_cls_loss.append(np.mean(batch_cls_loss))
            epoch_supcon_loss.append(np.mean(batch_supcon_loss))
            epoch_total_loss.append(np.mean(batch_total_loss))

        # -> end of local update <- #
        loss_dict = {
            'kd_loss': np.average(epoch_kd_loss),
            'cls_loss': np.average(epoch_cls_loss),
            'supcon_loss': np.average(epoch_supcon_loss) if len(epoch_supcon_loss) > 0 else np.NAN,
            'total_loss': np.average(epoch_total_loss)
        }
        return main_model, loss_dict


    def get_indices_to_drop_for_supcon(self, except_these_modalities):
        all_modalities = list(range(0, self.args.dataset_opts['num_modalities']))
        remaining_modalities = list(set(all_modalities).difference(set(except_these_modalities)))
        num_modalities_to_drop = random.randint(1, len(remaining_modalities) - 1)
        modality_indices_to_drop_for_supcon = random.sample(remaining_modalities, num_modalities_to_drop)
        return modality_indices_to_drop_for_supcon


    def add_gaussian_noise(self, data, mean_value=0.0, std_value=0.7):
        noise = torch.randn_like(data) * std_value + mean_value
        return data + noise


    def local_test(self, main_model):
        main_model = main_model.to(self.device)
        main_model.eval()

        correct_labels_all, predicted_labels_all = [], []
        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(self.test_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                classifier_outputs = main_model(input_data, extract_embeds_only=False)
                _, pred_labels = torch.max(classifier_outputs, dim=1)
                pred_labels = pred_labels.view(-1)
                predicted_labels_all = np.concatenate((predicted_labels_all, pred_labels.cpu()), axis=0)
                correct_labels_all = np.concatenate((correct_labels_all, labels.cpu()), axis=0)

        # Calculate test f1-score
        f1_macro = f1_score(correct_labels_all, predicted_labels_all, average='macro')
        return f1_macro


    def calculate_entropy_after_model_update(self, main_model):
        main_model = main_model.to(self.device)
        main_model.eval()

        local_entropy_list = []
        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(self.train_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                input_data = drop_modalities(self.args.dataset, copy.deepcopy(input_data), self.modality_indices_to_drop)

                classifier_outputs = main_model(input_data, extract_embeds_only=False)
                probabilities = F.softmax(classifier_outputs, dim=1)
                entropy = self.calculate_entropy_from_probabilities(probabilities)
                local_entropy_list.append(entropy.mean().item())
            average_entropy = np.mean(local_entropy_list)

        return average_entropy


    def calculate_entropy_from_probabilities(self, probabilities):
        return -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)


    def get_optimizer(self, model):
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


