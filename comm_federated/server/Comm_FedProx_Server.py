import copy

import numpy as np
from tqdm import tqdm
from comm_federated.client.Comm_FedProx_Client import FedProx_Client
import random
from comm_federated.comm_utils import log_communication_result_to_json


class FedProx_Server():
    def __init__(self, model, unique_user_list, speed_distri, args):
        self.args = args

        # 1. Model
        self.model = copy.deepcopy(model)

        # 2. Datasets and indices
        self.unique_user_list = unique_user_list

        # 3. Other settings
        self.total_num_users = len(self.unique_user_list)
        percentage_selected_clients_per_round = self.args.percentage_selected_clients_per_round
        self.num_clients_selected_per_round = int(self.total_num_users * percentage_selected_clients_per_round)

        # 4. Speed Configuration
        self.speed_distri = speed_distri
        self.allocate_device_per_client()

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
                cur_client = FedProx_Client(copy.deepcopy(self.model), client_idx, self.device_dict[client_idx],
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