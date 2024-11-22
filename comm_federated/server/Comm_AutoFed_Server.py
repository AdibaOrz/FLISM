import copy

import numpy as np
from tqdm import tqdm
from comm_federated.client.Comm_AutoFed_Client import AutoFed_Client
import random
from utils.missing_utils import get_drop_list_for_all_users
import time
from comm_federated.comm_utils import log_communication_result_to_json

class AutoFed_Server():
    def __init__(self, model, generative_model, unique_user_list, user_groups_train, user_groups_test, speed_distri, args):
        self.args = args

        # 1. Model
        self.model = copy.deepcopy(model)
        self.generative_model = generative_model

        # 2. Datasets and indices
        self.unique_user_list = unique_user_list
        self.user_groups_train = user_groups_train
        self.user_groups_test = user_groups_test

        # 3. Other settings
        self.total_num_users = len(self.unique_user_list)
        percentage_selected_clients_per_round = self.args.percentage_selected_clients_per_round
        self.num_clients_selected_per_round = int(self.total_num_users * percentage_selected_clients_per_round)

        # 4. Divide user for pre-train and main-train
        self.divide_user_pre_train_and_main_train()
        self.train_user_drop_dict = get_drop_list_for_all_users(args, list(self.user_groups_main_train.keys()))

        # 4. Speed Configuration
        self.speed_distri = speed_distri
        self.allocate_device_per_client()
        self.total_communication_cost = 0.0
        self.total_number_of_trained_parameters = 0

    def divide_user_pre_train_and_main_train(self):
        if not self.args.simulated:
            total_users = list(self.user_groups_train.keys())
            pretrain_ratio = 0.3
            num_pretrain_user = int(len(total_users) * pretrain_ratio)
            num_maintrain_user = len(total_users) - num_pretrain_user

            random.shuffle(total_users)

            pretrain_users = total_users[:num_pretrain_user]
            main_train_users = total_users[num_pretrain_user:]

            assert(len(main_train_users)==num_maintrain_user)

            self.user_groups_pre_train = {key: self.user_groups_train[key] for key in pretrain_users}
            # self.user_groups_pre_test : for validate (ranking) pretrain-generative model
            self.user_groups_pre_test = {key: self.user_groups_test[key] for key in pretrain_users}
            self.user_groups_main_train = {key: self.user_groups_train[key] for key in main_train_users}
            self.user_groups_main_test = {key: self.user_groups_test[key] for key in main_train_users}
        else:
            total_users = self.unique_user_list
            pretrain_ratio = 0.3
            num_pretrain_user = int(len(total_users) * pretrain_ratio)
            num_maintrain_user = len(total_users) - num_pretrain_user

            random.shuffle(total_users)

            pretrain_users = total_users[:num_pretrain_user]
            main_train_users = total_users[num_pretrain_user:]

            assert (len(main_train_users) == num_maintrain_user)

            self.user_groups_pre_train = {key: [] for key in pretrain_users}
            # self.user_groups_pre_test : for validate (ranking) pretrain-generative model
            self.user_groups_pre_test = {key: [] for key in pretrain_users}
            self.user_groups_main_train = {key: [] for key in main_train_users}
            self.user_groups_main_test = {key: [] for key in main_train_users}

    def allocate_device_per_client(self):
        if self.args.simulated:
            random_devices = [random.choice(list(self.speed_distri.keys())) for client_id in range(self.total_num_users)]
        else:
            if len(self.speed_distri.keys()) < self.total_num_users:
                keys = list(self.speed_distri.keys())
                while len(keys) < self.total_num_users:
                    keys.extend(keys)
                random.shuffle(keys)
                random_devices = keys[:self.total_num_users]
            else:
                random_devices = random.sample(self.speed_distri.keys(), self.total_num_users)
        self.device_dict = {i : {} for i in self.unique_user_list}
        for i in range(len(random_devices)):
            cur_client = self.unique_user_list[i]
            self.device_dict[cur_client] = self.speed_distri[random_devices[i]]

    def run(self):
        start_time = time.time()
        self.pre_train()
        self.main_train()
        elapsed_time = time.time() - start_time
        print(f"Total Communication Cost ==> {self.total_communication_cost}")
        print(f"Total Computation Cost ==> {self.total_number_of_trained_parameters}")
        log_communication_result_to_json(self.total_communication_cost, self.total_number_of_trained_parameters,
                                               self.args)
        print(f"Total time spent on training Harmony (stage 1 + stage 2): {elapsed_time:.2f} seconds")

    def pre_train(self):
        global_rounds = self.args.fl_rounds // 2

        pretrain_clients = list(self.user_groups_pre_train.keys())

        for epoch in tqdm(range(global_rounds)):
            # federated training round
            epoch_communication_cost = 0.0
            epoch_number_of_trained_parameters = 0
            print(f"------Global FL training round: {epoch + 1}/{global_rounds}-------")

            for client_idx in pretrain_clients:
                cur_client = AutoFed_Client(copy.deepcopy(self.model), copy.deepcopy(self.generative_model), client_idx, self.device_dict[client_idx], [], self.args)
                local_communication_cost, local_number_of_trained_parameters = cur_client.get_communication_cost_pretrain()

                print(f"Client [{client_idx}]: communication cost ==> {local_communication_cost}")
                epoch_communication_cost += local_communication_cost
                epoch_number_of_trained_parameters += local_number_of_trained_parameters

            print(f"Epoch [{epoch}]: communication cost ==> {epoch_communication_cost}")
            self.total_communication_cost += epoch_communication_cost
            self.total_number_of_trained_parameters += epoch_number_of_trained_parameters

        print(f"Total Communication Cost ==> {self.total_communication_cost}")

    def main_train(self):
        global_rounds = self.args.fl_rounds

        total_main_train_num_users = len(self.user_groups_main_train)
        percentage_selected_clients_per_round = self.args.percentage_selected_clients_per_round
        num_clients_selected_per_round = int(total_main_train_num_users * percentage_selected_clients_per_round)

        for epoch in tqdm(range(global_rounds)):
            # federated training round
            epoch_communication_cost = 0.0
            epoch_number_of_trained_parameters = 0
            print(f"------Global FL training round: {epoch + 1}/{global_rounds}-------")
            selected_clients_in_current_round = np.random.choice(list(self.user_groups_main_train.keys()),
                                                                 size=num_clients_selected_per_round,
                                                                 replace=False)

            selected_clients_in_current_round.sort()
            print(f"======>>>> Selected clients [{len(selected_clients_in_current_round)}]: {selected_clients_in_current_round}")

            for client_idx in selected_clients_in_current_round:
                modality_to_drop = self.train_user_drop_dict[client_idx]
                cur_client = AutoFed_Client(copy.deepcopy(self.model), copy.deepcopy(self.generative_model), client_idx, self.device_dict[client_idx], modality_to_drop, self.args)
                local_communication_cost, local_number_of_trained_parameters = cur_client.get_communication_cost_maintrain()

                print(f"Client [{client_idx}]: communication cost ==> {local_communication_cost}")
                epoch_communication_cost += local_communication_cost
                epoch_number_of_trained_parameters += local_number_of_trained_parameters

            print(f"Epoch [{epoch}]: communication cost ==> {epoch_communication_cost}")
            self.total_communication_cost += epoch_communication_cost
            self.total_number_of_trained_parameters += epoch_number_of_trained_parameters

        # print(f"Total Communication Cost ==> {self.total_communication_cost}")








