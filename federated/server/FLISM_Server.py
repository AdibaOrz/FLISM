import copy
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.general_utils import log_test_results_to_json
from utils.missing_utils import get_drop_list_for_all_users
from federated.client.FLISM_Client import FLISM_Client


class FLISM_Server:
    def __init__(self, model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict,
                 device, args):
        # 0. General setup
        self.args = args
        self.device = device

        # 1. Main model: aggregates all clients, to train main task (classification)
        #    KD model: used for knowledge distillation
        self.global_main_model = copy.deepcopy(model)
        self.global_kd_model = copy.deepcopy(model)

        # 2. Data
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_user_groups = train_user_groups
        self.test_user_groups = test_user_groups

        # 3. Missing modality simulation in train set
        self.num_modalities = self.args.dataset_opts['num_modalities']
        self.train_user_drop_dict = user_mod_drop_dict

        # 4. Client information
        self.total_clients = len(self.train_user_groups)
        client_selection_rate = self.args.client_selection_rate
        self.num_clients_selected_per_round = int(self.total_clients * client_selection_rate)

        # 5. Common setup
        self.global_rounds = self.args.global_rnd

        if self.args.autofed:
            print(f"Autofed setting version FLISM is running")
            # 6. for AutoFed evaluation
            self.autofed_user_info()
            ###### BECAUSE OF RANDOM ... #####
            for_shuffle = copy.deepcopy(list(self.train_user_groups.keys()))
            random.shuffle(for_shuffle)
            ##################################
            autofed_train_user_drop_dict = get_drop_list_for_all_users(args, self.maintrain_users)
            self.train_user_drop_dict = {**autofed_train_user_drop_dict, **{i: [] for i in self.pretrain_users}}


    def run(self):
        print(f"=== Running FLISM Server [version {self.args.ver}] ===")
        self.federate()
        self.evaluate()


    def federate(self):
        for rnd in tqdm(range(self.global_rounds), desc="Global Rounds"):
            main_model_list, training_loss_list, metadata_list = [], [], []
            selected_clients = np.random.choice(list(self.train_user_groups.keys()), self.num_clients_selected_per_round,
                                                replace=False)
            selected_clients.sort()

            # Selected clients perform local training
            for client_id in selected_clients:
                client = FLISM_Client(client_id, self.train_dataset, self.test_dataset,
                                      self.train_user_groups[client_id], self.test_user_groups[client_id],
                                      self.train_user_drop_dict[client_id], self.device, self.args)
                if self.args.ver in ['none', 'supcon', 'supcon_wavg']:
                    updated_local_model, meta = client.local_train(copy.deepcopy(self.global_main_model), None)
                elif self.args.ver == 'supcon_wavg_kd':
                    updated_local_model, meta = client.local_train(copy.deepcopy(self.global_main_model),
                                                                    copy.deepcopy(self.global_kd_model))

                main_model_list.append(copy.deepcopy(updated_local_model))
                training_loss_list.append(meta['train_loss']['total_loss'])
                metadata_list.append(meta)
            # Aggregate weights
            self.aggregate_weights(main_model_list, selected_clients, metadata_list)
            print(f'Round {rnd+1}/{self.global_rounds} training loss: {np.mean(training_loss_list):.3f}')



    def aggregate_weights(self, main_model_list, selected_clients, metadata_list):
        if self.args.ver in ['none', 'supcon']: # FedAvg w/ equal weights, no KD involved
            self.weighted_average_main(main_model_list, selected_clients, metadata_list, 'equal')
        elif self.args.ver in ['supcon_wavg']: # Entropy-based weighted average, no KD involved
            self.weighted_average_main(main_model_list, selected_clients, metadata_list, 'entropy')
        elif self.args.ver in ['supcon_wavg_kd']: # Entropy-based weighted average, KD involved
            self.weighted_average_main(main_model_list, selected_clients, metadata_list, 'entropy')
            self.weighted_average_kd(main_model_list)
        else:
            raise ValueError(f"No such version {self.args.ver} exists")


    def weighted_average_main(self, main_model_list, selected_clients, metadata_list, weight_mode):
        if weight_mode == 'equal':
            weight_coefficient_list = [1/len(main_model_list)] * len(main_model_list)
        elif weight_mode == 'entropy':
            weight_coefficient_list = self.compute_weights_for_aggregation(selected_clients, metadata_list, weight_mode)
        else:
            raise ValueError(f"No such weight mode {weight_mode} exists")

        model_weights = [local_model.state_dict() for local_model in main_model_list]
        w_avg = {key: torch.zeros_like(value) for key, value in model_weights[0].items()}
        for key in w_avg.keys():
            for i in range(len(model_weights)):
                w_avg[key] += (model_weights[i][key] * weight_coefficient_list[i])
        # Update main global model
        self.global_main_model.load_state_dict(w_avg)


    def weighted_average_kd(self, main_model_list):
        # Weighting based on global model
        weight_coefficient_list = [1 / len(main_model_list)] * len(main_model_list)

        model_weights = [local_model.state_dict() for local_model in main_model_list]
        w_avg = {key: torch.zeros_like(value) for key, value in model_weights[0].items()}
        for key in w_avg.keys():
            for i in range(len(model_weights)):
                w_avg[key] += (model_weights[i][key] * weight_coefficient_list[i])
        # Update KD global model
        self.global_kd_model.load_state_dict(w_avg)


    def compute_weights_for_aggregation(self, selected_clients, metadata_list, weight_mode):
        if weight_mode == 'entropy':
            selected_client_entropy_list = []
            for client_id, metadata in zip(selected_clients, metadata_list):
                selected_client_entropy_list.append(metadata['entropy_on_train_data'])
            # the lower the entropy, the more weight it should have
            reciprocal_entropy_list = [1/entropy for entropy in selected_client_entropy_list]
            normalized_weight_coefficients = [reciprocal_entropy / sum(reciprocal_entropy_list) for reciprocal_entropy in reciprocal_entropy_list]
        else:
            raise ValueError(f"No such weight mode {weight_mode} exists")
        return normalized_weight_coefficients


    def evaluate(self):
        print(f"Testing the global model. FLISM version: {self.args.ver}")
        self.global_main_model = self.global_main_model.to(self.device)
        self.global_main_model.eval()

        local_test_results_dict = {}
        f1_macro_list = []
        for enum_id, client_id in enumerate(self.train_user_groups.keys()):
            client = FLISM_Client(client_id, self.train_dataset, self.test_dataset,
                                  self.train_user_groups[client_id], self.test_user_groups[client_id],
                                  self.train_user_drop_dict[client_id], self.device, self.args)
            test_f1_macro = client.local_test(copy.deepcopy(self.global_main_model))
            f1_macro_list.append(test_f1_macro)
            local_test_results_dict[client_id] = {'test_f1': test_f1_macro,
                                                  'missing_mods': self.train_user_drop_dict[client_id]}
            # Print results
            missing_modality_num = 0 if client_id not in self.train_user_drop_dict.keys() else len(self.train_user_drop_dict[client_id])
            print(f'[{enum_id}] Client {client_id} w/ {missing_modality_num} missing modals.')
            print(f'F1-Macro: {test_f1_macro:.3f}')

        # Average results
        avg_f1_macro = np.mean(f1_macro_list)
        log_test_results_to_json(local_test_results_dict, avg_f1_macro, self.args)
        print(f'Average F1-Macro: {avg_f1_macro:.3f}')

    def autofed_user_info(self):
        saved_path = self.args.json_log_path.replace('flism', 'autofed')
        user_info_path = f'{saved_path}/user_info/{self.args.seed}/user_info.json'
        with open(user_info_path, 'r') as f:
            user_info_data = json.load(f)

        self.pretrain_users = user_info_data['pre_train']
        self.maintrain_users = user_info_data['main_train']

        print(f'pretrain : {self.pretrain_users}, maintrain : {self.maintrain_users}')
