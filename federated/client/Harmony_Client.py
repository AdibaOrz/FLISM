import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils.missing_utils import drop_modalities
from data_loader.data_loader import DatasetSplit


class Harmony_Client(object):
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
        self.total_modalities = args.dataset_opts['num_modalities']
        self.available_modalities = [i for i in range(self.total_modalities) if i not in self.modality_indices_to_drop]
        self.train_dataloader = DataLoader(DatasetSplit(self.train_dataset, self.client_train_data_indices, args.dataset),
                                           batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(DatasetSplit(self.test_dataset, self.client_test_data_indices, args.dataset),
                                            batch_size=self.args.local_bs, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


    def set_initial_pretrained_models(self, pretrained_model):
        """Initialize initial model as the pre-trained model from the server"""
        self.initial_model = copy.deepcopy(pretrained_model)
        self.current_model = copy.deepcopy(pretrained_model)


    def get_optimizer_for_stage_1(self, model):
        optimizer_name = self.args.optim
        lr = self.args.local_lr
        wd = self.args.local_wd
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")
        return optimizer


    def train_stage_1(self, model, modality_to_train):
        """=== Stage 1 (Uni-modal) Training  ==="""
        assert modality_to_train in self.available_modalities, f"Modality {modality_to_train} is not available for training"
        model = model.to(self.device)
        model.train()
        self.optimizer = self.get_optimizer_for_stage_1(model)

        epoch_loss = []
        local_epochs = self.args.hrm_local_epochs
        for local_epoch in range(local_epochs):
            batch_loss = []
            for batch_idx, (input_data, labels) in enumerate(self.train_dataloader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                input_data = drop_modalities(self.args.dataset, copy.deepcopy(input_data), self.modality_indices_to_drop)
                self.optimizer.zero_grad()
                model_outputs = model(input_data)
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            # Finished one epoch of training
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        avg_train_loss = sum(epoch_loss)/len(epoch_loss)
        return model, avg_train_loss


    def test_unimodal_client(self, model, modality_index, modality_name):
        print(f"Testing unimodal client {self.client_id} on modality idx ({modality_index}) with name {modality_name}...")
        model = model.to(self.device)
        model.eval()

        correct_labels_all, predicted_labels_all = [], []
        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(self.test_dataloader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                outputs = model(input_data)
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                predicted_labels_all = np.concatenate((predicted_labels_all, pred_labels.cpu()), axis=0)
                correct_labels_all = np.concatenate((correct_labels_all, labels.cpu()), axis=0)

        torch.cuda.empty_cache()
        # Calculate test f1-score
        f1_macro = f1_score(correct_labels_all, predicted_labels_all, average='macro')
        return f1_macro


    def compare_curr_and_initial_model_weights(self, curr_model, initial_model, modality_idx, msg):
        print(f"===== MEAN and STD of initial and current models {msg} =====")
        with torch.no_grad():
            curr_mod_encoder_params = torch.cat([p.flatten() for p in curr_model.encoder.modality_layers[modality_idx].parameters()])
            init_mod_encoder_params = torch.cat([p.flatten() for p in initial_model.encoder.modality_layers[modality_idx].parameters()])
            print(f"Mean of curr_mod_encoder_params: {torch.mean(curr_mod_encoder_params)}, std: {torch.std(curr_mod_encoder_params)}")
            print(f"Mean of init_mod_encoder_params: {torch.mean(init_mod_encoder_params)}, std: {torch.std(init_mod_encoder_params)}")


    def train_stage_2(self, curr_classifier_weights):
        self.current_model.classifier.load_state_dict(copy.deepcopy(curr_classifier_weights))
        self.current_model.train()

        optimizer_params = [{'params': self.current_model.classifier.parameters(),
                             'lr': self.args.local_lr}]
        # Based on Harmony's code, in Stage 2, encoder is trained with lr=1e-4 for all datasets
        for modality_idx in self.available_modalities:
            optimizer_params.append({'params': self.current_model.encoder.modality_layers[modality_idx].parameters(),
                                     'lr': 1e-4})

        if self.args.optim == 'sgd':
            self.optimizer = torch.optim.SGD(optimizer_params, weight_decay=self.args.local_wd)
        elif self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_params, weight_decay=self.args.local_wd)

        epoch_loss = []
        local_epochs = self.args.hrm_local_epochs
        for _ in tqdm(range(local_epochs)):
            batch_loss = []
            for batch_idx, (input_data, labels) in enumerate(self.train_dataloader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                input_data = drop_modalities(self.args.dataset, copy.deepcopy(input_data), self.modality_indices_to_drop)
                model_outputs = self.current_model(input_data)
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        avg_train_loss = sum(epoch_loss) / len(epoch_loss)

        # Part 2: Encoder dist computation
        encoder_dist_list = self.calculate_cosine_distance(self.current_model, self.initial_model, debug=self.args.verbose)
        return self.current_model.classifier, encoder_dist_list, avg_train_loss


    def local_test_after_stage_2(self, latest_classifier_weights):
        self.current_model.classifier.load_state_dict(copy.deepcopy(latest_classifier_weights))
        self.current_model.eval()

        correct_labels_all, predicted_labels_all = [], []
        with torch.no_grad():
            for batch_idx, (input_data, labels) in enumerate(self.test_dataloader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()
                outputs = self.current_model(input_data)

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                predicted_labels_all = np.concatenate((predicted_labels_all, pred_labels.cpu()), axis=0)
                correct_labels_all = np.concatenate((correct_labels_all, labels.cpu()), axis=0)

            torch.cuda.empty_cache()
        # Calculate test f1-score
        f1_macro = f1_score(correct_labels_all, predicted_labels_all, average='macro')
        return f1_macro


    def calculate_cosine_distance(self, curr_model, initial_model, debug=False):
        encoder_dist_list = []
        for modality_idx in self.available_modalities:
            curr_mod_encoder = curr_model.encoder.modality_layers[modality_idx]
            init_mod_encoder = initial_model.encoder.modality_layers[modality_idx]
            curr_mod_encoder_params = torch.cat([p.flatten() for p in curr_mod_encoder.parameters()])
            init_mod_encoder_params = torch.cat([p.flatten() for p in init_mod_encoder.parameters()])

            with torch.no_grad():
                cos_similarity = F.cosine_similarity(curr_mod_encoder_params, init_mod_encoder_params, dim=0)
            cos_distance = 1 - cos_similarity.item() # 0(perfect similarity) to 2 (perfect dissimilarity)
            encoder_dist_list.append(cos_distance)

        if debug:
            print("Encoder dist list: ", encoder_dist_list)
        return encoder_dist_list
