import copy

import numpy as np
from tqdm import tqdm
from comm_federated.client.Comm_FedMultiModal_Client import FedMultiModal_Client
import random
from comm_federated.comm_utils import log_communication_result_to_json


class FedMultiModal_Server():
    def __init__(self, model, unique_user_list, user_mod_drop_dict, speed_distri, args):
        self.args = args

        # 1. Model
        self.global_model = model

        # 2. Datasets and indices
        self.unique_user_list = unique_user_list
        self.user_mod_drop_dict = user_mod_drop_dict


        # 3. Other settings
        self.total_num_users = len(self.unique_user_list)
        percentage_selected_clients_per_round = self.args.percentage_selected_clients_per_round
        self.num_clients_selected_per_round = int(self.total_num_users * percentage_selected_clients_per_round)

        # 4. Speed Configuration
        self.speed_distri = speed_distri
        self.allocate_device_per_client()

        self.init()

    def init(self):
        self.existing_mod = {}
        num_modalities = self.args.dataset_opts[
            'num_modalities'] if not self.args.simulated else self.args.simulated_number_of_modalities
        all_modalities = list(range(0, num_modalities))
        for client_idx, drop_mod_list in self.user_mod_drop_dict.items():
            remaining_modalities = list(set(all_modalities).difference(set(drop_mod_list)))
            self.existing_mod[client_idx] = remaining_modalities

    def allocate_device_per_client(self):
        if self.args.simulated:
            random_devices = [random.choice(list(self.speed_distri.keys())) for client_id in
                              range(self.total_num_users)]
        else:
            if len(self.speed_distri.keys()) < self.total_num_users:
                keys = list(self.speed_distri.keys())
                while len(keys) < self.total_num_users:
                    keys.extend(keys)
                random.shuffle(keys)
                random_devices = keys[:self.total_num_users]
            else:
                random_devices = random.sample(self.speed_distri.keys(), self.total_num_users)
        self.device_dict = {i: {} for i in self.unique_user_list}
        for i in range(len(random_devices)):
            cur_client = self.unique_user_list[i]
            self.device_dict[cur_client] = self.speed_distri[random_devices[i]]

    def run(self):
        global_rounds = self.args.fl_rounds
        total_communication_cost = 0.0
        total_number_of_trained_parameters = 0
        for epoch in tqdm(range(global_rounds)):
            # federated training round
            epoch_communication_cost = 0.0
            epoch_number_of_trained_parameters = 0
            print(f"------Global FL training round: {epoch + 1}/{global_rounds}-------")
            selected_clients_in_current_round = np.random.choice(self.unique_user_list,
                                                                 size=self.num_clients_selected_per_round,
                                                                 replace=False)

            selected_clients_in_current_round.sort()
            print(
                f"======>>>> Selected clients [{len(selected_clients_in_current_round)}]: {selected_clients_in_current_round}")

            for client_idx in selected_clients_in_current_round:
                cur_client = FedMultiModal_Client(copy.deepcopy(self.global_model), client_idx, self.existing_mod[client_idx], self.device_dict[client_idx],
                                         self.args)
                local_communication_cost, local_number_of_trained_parameters = cur_client.get_communication_cost()

                print(f"Client [{client_idx}]: communication cost ==> {local_communication_cost}")
                epoch_communication_cost += local_communication_cost
                epoch_number_of_trained_parameters += local_number_of_trained_parameters

            print(f"Epoch [{epoch}]: communication cost ==> {epoch_communication_cost}")
            total_communication_cost += epoch_communication_cost
            total_number_of_trained_parameters += epoch_number_of_trained_parameters

        print(f"Total Communication Cost ==> {total_communication_cost}")
        print(f"Total Computation Cost ==> {total_number_of_trained_parameters}")
        # todo: save the result in json file

        log_communication_result_to_json(total_communication_cost, total_number_of_trained_parameters, self.args)