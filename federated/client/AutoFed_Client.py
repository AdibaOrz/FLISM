import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from utils.autofed_utils import extract_indices, get_min_loss_for_dropped_indices, get_key_from_value
from utils.missing_utils import drop_modalities
from data_loader.data_loader import DatasetSplit


class AutoFed_Client(object):
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


    def pre_train(self, generative_model):
        generative_model.to(self.device)
        generative_model.train()

        dim_per_modalities = generative_model.dim_per_modalities
        num_autoencoders = len(generative_model.generation_model_mapping_dict.keys())

        dim_list = []
        current_index = 0

        for i in range(len(dim_per_modalities)):
            dims = dim_per_modalities[i]
            dim_list.append(list(range(current_index, current_index + dims)))
            current_index += dims

        local_epochs = self.args.autofed_gen_local_epochs
        autoencoder_loss = []
        for autoencoder_idx in range(num_autoencoders):

            cur_generative_model = generative_model.autoencoder_layers[autoencoder_idx]
            cur_optimizer = torch.optim.Adam(cur_generative_model.parameters(), lr=0.001, weight_decay=0.0001)
            cur_criterion = nn.MSELoss()

            cur_combination_key = generative_model.generation_model_mapping_dict[
                autoencoder_idx]  # 'from_{in_idx}_to_{out_idx}'
            cur_in_idx, cur_out_idx = extract_indices(cur_combination_key)
            cur_in_dims, cur_out_dims = dim_list[cur_in_idx], dim_list[cur_out_idx]

            epoch_loss = []
            for local_epoch in range(local_epochs):
                batch_loss = []
                for batch_idx, (input_data, _) in enumerate(self.train_loader):
                    input_data = input_data.to(self.device)

                    cur_optimizer.zero_grad()

                    cur_input = input_data[:, cur_in_dims, :]
                    cur_label = input_data[:, cur_out_dims, :]

                    cur_output = cur_generative_model(cur_input)

                    loss = cur_criterion(cur_output, cur_label)

                    loss.backward()
                    cur_optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            avg_train_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"[For Autoencoder {autoencoder_idx}, {cur_generative_model.name}] Average loss : {avg_train_loss}")
            autoencoder_loss.append(avg_train_loss)

        total_avg_loss = sum(autoencoder_loss) / len(autoencoder_loss)

        return generative_model, total_avg_loss

    def local_train(self, model, generative_model, autoencoder_loss_dict):
        model.to(self.device)
        model.train()
        generative_model.to(self.device)
        generative_model.eval()

        modality_indices_to_drop = self.modality_indices_to_drop
        modality_indices_not_to_drop = list(
            set([i for i in range(self.args.dataset_opts['num_modalities'])]) - set(modality_indices_to_drop))

        autoencoder_source_idx_dict_per_dropped_mod = get_min_loss_for_dropped_indices(autoencoder_loss_dict,
                                                                                       modality_indices_to_drop,
                                                                                       modality_indices_not_to_drop)

        dim_per_modalities = generative_model.dim_per_modalities

        dim_list = []
        current_index = 0

        for i in range(len(dim_per_modalities)):
            dims = dim_per_modalities[i]
            dim_list.append(list(range(current_index, current_index + dims)))
            current_index += dims

        optimizer = self.get_optimizer(model)

        local_epochs = self.args.local_ep
        epoch_loss = []
        for local_epoch in range(local_epochs):
            batch_loss = []

            for batch_idx, (input_data, labels) in enumerate(self.train_loader):
                input_data, labels = input_data.to(self.device), labels.to(self.device).long()

                optimizer.zero_grad()  # Zero gradients from previous iteration
                # Drop modalities
                input_data = drop_modalities(self.args.dataset, input_data, self.modality_indices_to_drop)

                # inputation
                with torch.no_grad():
                    for drop_index in modality_indices_to_drop:
                        source_index_string = autoencoder_source_idx_dict_per_dropped_mod[drop_index]  # 'from_x_to_y
                        source_idx = int(source_index_string[5])

                        # Sanity Check
                        dropped_index = int(source_index_string[-1])
                        assert (drop_index == dropped_index)

                        autoencoder_idx = get_key_from_value(generative_model.generation_model_mapping_dict,
                                                             source_index_string)
                        cur_generative_model = generative_model.autoencoder_layers[autoencoder_idx]
                        cur_combination_key = source_index_string
                        cur_in_idx, cur_out_idx = extract_indices(cur_combination_key)
                        cur_in_dims, cur_out_dims = dim_list[cur_in_idx], dim_list[cur_out_idx]

                        cur_input = input_data[:, cur_in_dims, :]

                        generated_data = cur_generative_model(cur_input)
                        input_data[:, cur_out_dims, :] = generated_data

                model_outputs = model(input_data, extract_embeds_only=False)  # forward pass input with dropped modalities
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            # Finished one mini-batch training
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Finished local training
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