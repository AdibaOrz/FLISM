import copy
import random
import time
from tqdm import tqdm
import torch
import os
import json
import numpy as np

from torch.utils.data import Dataset, DataLoader
from itertools import chain
from data_loader.data_loader import DatasetSplit
from utils.autofed_utils import extract_indices
from utils.general_utils import log_test_results_to_json
from utils.missing_utils import get_drop_list_for_all_users
from federated.client.AutoFed_Client import AutoFed_Client


class AutoFed_Server:
    def __init__(self, model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict,
                 device, args):
        self.args = args
        self.device = device

        # 1. Model
        self.main_model = model['model']
        self.generative_model = model['generative_model']
        self.global_weights = None
        self.global_generative_weights = None

        # 2. Datasets and indices
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_user_groups = train_user_groups
        self.test_user_groups = test_user_groups

        # 3. Drop settings
        self.num_modalities = args.dataset_opts['num_modalities']
        self.divide_user_pre_train_and_main_train()
        self.train_user_drop_dict = get_drop_list_for_all_users(args, list(self.main_train_user_groups.keys()))


    def divide_user_pre_train_and_main_train(self):
        total_users = list(self.train_user_groups.keys())
        pre_train_ratio = 0.3
        num_pre_train_user = int(len(total_users) * pre_train_ratio)

        random.shuffle(total_users)

        pre_train_users = total_users[:num_pre_train_user]
        main_train_users = total_users[num_pre_train_user:]
        assert(len(main_train_users)==len(total_users)-num_pre_train_user)

        self.pre_train_user_groups = {key: self.train_user_groups[key] for key in pre_train_users}
        self.pre_test_user_groups = {key: self.test_user_groups[key] for key in pre_train_users}
        self.main_train_user_groups = {key: self.train_user_groups[key] for key in main_train_users}
        self.main_test_user_groups = {key: self.test_user_groups[key] for key in main_train_users}

        # # Save the user info
        # save_path = f'{self.args.json_log_path}/user_info/{self.args.seed}'
        # os.makedirs(save_path, exist_ok=True)
        # user_dict = {'pre_train': list(self.pre_train_user_groups.keys()),
        #              'main_train': list(self.main_train_user_groups.keys())}
        # with open(f'{save_path}/user_info.json', 'w') as json_file:
        #     json.dump(user_dict, json_file)
        #
        # exit()


    def run(self):
        print("=== Running AutoFed Server ===")
        self.federate()
        self.evaluate()


    def federate(self):
        start_time = time.time()
        model_save_path = f'{self.args.pretrained_path}/{self.args.method}/note:{self.args.note}_dataset:{self.args.dataset}_seed:{self.args.seed}_autoencoder'
        if not os.path.exists(model_save_path + '.pth'):
            autoencoder_loss_dict = self.run_pre_train()
            self.save_model_and_dict(model_save_path, autoencoder_loss_dict)
        else:
            autoencoder_loss_dict = self.load_model_and_dict(model_save_path)
        elapsed_time = time.time() - start_time
        print(f"Total time spent on pre-training Autoencoder: {elapsed_time:.2f} seconds")

        self.run_main_train(autoencoder_loss_dict)
        elapsed_time = time.time() - start_time
        print(f"Total time spent on training AutoFed (pre-train + main-train): {elapsed_time:.2f} seconds")


    def run_pre_train(self):
        self.generative_model.to(self.device)
        self.generative_model.train()

        pretrain_clients = list(self.pre_train_user_groups.keys())
        print(f"======>>>> Clients for pre-training : {pretrain_clients} : {len(pretrain_clients)} clients")
        generative_pretrain_rounds = self.args.autofed_gen_fl_rounds

        for rnd in tqdm(range(generative_pretrain_rounds), desc="Pre-training Global Rounds"):
            updated_local_generative_model_list, local_loss_list = [], []

            for client_idx in pretrain_clients:
                local_generative_client = AutoFed_Client(client_idx,
                                                         self.train_dataset,
                                                         self.test_dataset,
                                                         self.pre_train_user_groups[client_idx],
                                                         self.pre_test_user_groups[client_idx],
                                                         None,
                                                         self.device,
                                                         self.args)

                updated_local_generative_model, local_loss = local_generative_client.pre_train(
                    copy.deepcopy(self.generative_model))

                # Add trained weights and losses
                updated_local_generative_model_list.append(updated_local_generative_model)
                local_loss_list.append(local_loss)

            self.generative_model = self.aggregate_models(updated_local_generative_model_list)
            loss_avg = sum(local_loss_list) / len(local_loss_list)
            print(f'\n Avg. Pre-Train loss after {rnd + 1}-th global round: {loss_avg:.3f}')

        autoencoder_loss_dict = self.rank_autoencoders()
        # todo : Should we save the model and this dict?

        return autoencoder_loss_dict

    def rank_autoencoders(self):
        self.generative_model.to(self.device)
        self.generative_model.eval()

        pretrain_eval_dataloader = DataLoader(
            DatasetSplit(self.train_dataset, list(chain(*self.pre_test_user_groups.values())), self.args.dataset),
            batch_size=self.args.local_bs,
            shuffle=False)

        dim_per_modalities = self.generative_model.dim_per_modalities
        num_autoencoders = len(self.generative_model.generation_model_mapping_dict.keys())

        dim_list = []
        current_index = 0

        for i in range(len(dim_per_modalities)):
            dims = dim_per_modalities[i]
            dim_list.append(list(range(current_index, current_index + dims)))
            current_index += dims

        autoencoder_loss_dict = {}
        with torch.no_grad():
            for autoencoder_idx in range(num_autoencoders):

                cur_generative_model = self.generative_model.autoencoder_layers[autoencoder_idx]
                cur_criterion = torch.nn.MSELoss()
                cur_combination_key = self.generative_model.generation_model_mapping_dict[
                    autoencoder_idx]  # 'from_{in_idx}_to_{out_idx}'
                cur_in_idx, cur_out_idx = extract_indices(cur_combination_key)
                cur_in_dims, cur_out_dims = dim_list[cur_in_idx], dim_list[cur_out_idx]
                batch_loss = []

                for batch_idx, (input_data, _) in enumerate(pretrain_eval_dataloader):
                    input_data = input_data.to(self.device)

                    cur_input = input_data[:, cur_in_dims, :]
                    cur_label = input_data[:, cur_out_dims, :]
                    cur_output = cur_generative_model(cur_input)

                    loss = cur_criterion(cur_output, cur_label)
                    batch_loss.append(loss.item())

                autoencoder_loss_dict[cur_combination_key] = sum(batch_loss) / len(batch_loss)

        return autoencoder_loss_dict

    def run_main_train(self, autoencoder_loss_dict):
        total_main_train_num_users = len(self.main_train_user_groups)
        client_selection_rate = self.args.client_selection_rate
        num_clients_selected_per_round = int(total_main_train_num_users * client_selection_rate)

        self.main_model = self.main_model.to(self.device)
        self.generative_model.eval()
        self.global_weights = self.main_model.state_dict()

        global_rounds = self.args.global_rnd
        for epoch in tqdm(range(global_rounds)):
            print(f"------Global FL training round: {epoch + 1}/{global_rounds}-------")
            updated_local_main_model_list, local_loss_list = [], []
            self.main_model.train()
            selected_clients_in_current_round = np.random.choice(list(self.main_train_user_groups.keys()),
                                                                 size=num_clients_selected_per_round,
                                                                 replace=False)

            selected_clients_in_current_round.sort()
            print(
                f"======>>>> Selected clients [{len(selected_clients_in_current_round)}]: {selected_clients_in_current_round}")
            for client_idx in selected_clients_in_current_round:
                local_main_client = AutoFed_Client(client_idx,
                                                   self.train_dataset,
                                                   self.test_dataset,
                                                   self.main_train_user_groups[client_idx],
                                                   self.main_test_user_groups[client_idx],
                                                   self.train_user_drop_dict[client_idx],
                                                   self.device,
                                                   self.args)

                updated_local_main_model, local_loss = local_main_client.local_train(copy.deepcopy(self.main_model),
                                                                              self.generative_model,
                                                                              autoencoder_loss_dict)
                updated_local_main_model_list.append(updated_local_main_model)
                local_loss_list.append(local_loss)

            self.main_model = self.aggregate_models(updated_local_main_model_list)

            loss_avg = sum(local_loss_list) / len(local_loss_list)
            print(f'\n Avg. Main-Train loss after {epoch + 1}-th global round: {loss_avg:.3f}')
    def evaluate(self):
        self.main_model.eval()
        """Evaluate the global model for each client and log the results"""
        local_test_result_dict = {}
        f1_macro_list = []
        for enum_id, client_id in enumerate(self.main_test_user_groups.keys()):
            client = AutoFed_Client(client_id, self.train_dataset, self.test_dataset,
                                   self.main_train_user_groups[client_id], self.main_test_user_groups[client_id],
                                   self.train_user_drop_dict[client_id], self.device, self.args)
            test_f1_macro = client.local_test(copy.deepcopy(self.main_model))
            f1_macro_list.append(test_f1_macro)
            local_test_result_dict[client_id] = {'test_f1': test_f1_macro,
                                                  'missing_mods': self.train_user_drop_dict[client_id]}

            missing_modality_num = 0 if client_id not in self.train_user_drop_dict.keys() else len(self.train_user_drop_dict[client_id])
            print(f'[{enum_id}] Client {client_id} w/ {missing_modality_num} missing modals:')
            print(f'\tTEST F1-MACRO:{test_f1_macro:.3f}')
        # Average of all clients
        avg_f1_macro = sum(f1_macro_list) / len(f1_macro_list)
        log_test_results_to_json(local_test_result_dict, avg_f1_macro, self.args)
        print(f'Average F1-Macro: {avg_f1_macro:.3f}')

    def save_model_and_dict(self, model_save_path, autoencoder_loss_dict):
        torch.save(self.generative_model.state_dict(),
                   f'{model_save_path}.pth')
        with open(f'{model_save_path}.json',
                  'w') as json_file:
            json.dump(autoencoder_loss_dict, json_file, indent=4)

    def load_model_and_dict(self, model_save_path):
        generative_model_weight = torch.load(f'{model_save_path}.pth')
        self.generative_model.load_state_dict(generative_model_weight)
        print(f"Model weight loaded!")
        with open(f'{model_save_path}.json', 'r') as f:
            autoencoder_loss_dict = json.load(f)
        return autoencoder_loss_dict

    def aggregate_models(self, updated_local_model_list):
        new_model = copy.deepcopy(updated_local_model_list[0])
        weight_coefficient_list = [1/ len(updated_local_model_list) for _ in updated_local_model_list]
        model_weights = [local_model.state_dict() for local_model in updated_local_model_list]
        w_avg = {key: torch.zeros_like(value) for key, value in model_weights[0].items()}
        for key in w_avg.keys():
            for i in range(len(model_weights)): # i-th client, key-th layer
                w_avg[key] += model_weights[i][key] * weight_coefficient_list[i]
        new_model.load_state_dict(w_avg)
        return new_model
