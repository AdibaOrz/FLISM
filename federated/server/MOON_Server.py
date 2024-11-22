import copy
import torch
import numpy as np
from tqdm import tqdm
from utils.general_utils import log_test_results_to_json
from federated.client.MOON_Client import MOON_Client

class MOON_Server:
    def __init__(self, model, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict,
                 device, args):
        self.args = args
        self.device = device

        # 1. Model
        self.model = model
        self.global_weights = None

        # 2. Datasets and indices
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_user_groups = train_user_groups
        self.test_user_groups = test_user_groups

        # 3. Drop settings
        self.num_modalities = args.dataset_opts['num_modalities']
        self.train_user_drop_dict = user_mod_drop_dict

        # 4. Other settings
        self.total_num_users = len(self.train_user_groups)
        client_selection_rate = self.args.client_selection_rate
        self.num_clients_selected_per_round = int(self.total_num_users * client_selection_rate)

        # 5. MOON specific settings
        self.previous_models = {i: model for i in self.train_user_groups.keys()}

    def run(self):
        print("=== Running MOON Server ===")
        self.federate()
        self.evaluate()

    def federate(self):
        """Federated learning process"""
        global_rounds = self.args.global_rnd
        train_loss_list = []

        for rnd in tqdm(range(global_rounds), desc="Global Rounds"):
            updated_local_model_list, local_loss_list = [], []
            selected_clients = np.random.choice(list(self.train_user_groups.keys()),
                                                size=self.num_clients_selected_per_round, replace=False)
            selected_clients.sort()

            for client_id in selected_clients:
                client = MOON_Client(client_id, self.train_dataset, self.test_dataset,
                                       self.train_user_groups[client_id], self.test_user_groups[client_id],
                                       self.train_user_drop_dict[client_id], self.device, self.args)

                updated_local_model, local_loss = client.local_train(copy.deepcopy(self.model),
                                                                     copy.deepcopy(self.previous_models[client_id]))
                updated_local_model_list.append(updated_local_model)
                local_loss_list.append(local_loss)

                # Update previous local models
                self.previous_models[client_id] = updated_local_model

            # Average the weights
            self.model = self.aggregate_models(updated_local_model_list)
            loss_avg = sum(local_loss_list) / len(local_loss_list)
            train_loss_list.append(loss_avg)
            if rnd % 10 == 0:
                print(f'Round: {rnd} | Average Loss: {loss_avg:.3f}')

    def evaluate(self):
        """Evaluate the global model for each client and log the results"""
        local_test_result_dict = {}
        f1_macro_list = []
        for enum_id, client_id in enumerate(self.train_user_groups.keys()):
            client = MOON_Client(client_id, self.train_dataset, self.test_dataset,
                                   self.train_user_groups[client_id], self.test_user_groups[client_id],
                                   self.train_user_drop_dict[client_id], self.device, self.args)
            test_f1_macro = client.local_test(copy.deepcopy(self.model))
            f1_macro_list.append(test_f1_macro)
            local_test_result_dict[client_id] = {'test_f1': test_f1_macro,
                                                  'missing_mods': self.train_user_drop_dict[client_id]}

            missing_modality_num = 0 if client_id not in self.train_user_drop_dict.keys() else len(self.train_user_drop_dict[client_id])
            print(f'[{enum_id}] Client {client_id} w/ {missing_modality_num} missing modals:')
            print(f'\tTEST F1-MACRO:{test_f1_macro:.3f}')
        # Average of all clients
        avg_f1_macro = sum(f1_macro_list) / len(f1_macro_list)
        log_test_results_to_json(local_test_result_dict, avg_f1_macro, self.args)
        return local_test_result_dict

    def aggregate_models(self, updated_local_model_list):
        new_model = copy.deepcopy(self.model)
        weight_coefficient_list = [1/ len(updated_local_model_list) for _ in updated_local_model_list]
        model_weights = [local_model.state_dict() for local_model in updated_local_model_list]
        w_avg = {key: torch.zeros_like(value) for key, value in model_weights[0].items()}
        for key in w_avg.keys():
            for i in range(len(model_weights)): # i-th client, key-th layer
                w_avg[key] += model_weights[i][key] * weight_coefficient_list[i]
        new_model.load_state_dict(w_avg)
        return new_model
