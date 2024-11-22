import copy
import os
import time
import torch

from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

from utils.missing_utils import modality_idx_to_name

from comm_federated.client.Comm_Harmony_Client import Harmony_Client
from models.harmony_models.harmony_model_utils import get_dynamic_model
import random
import statistics
import json
from comm_federated.comm_utils import log_communication_result_to_json

class Harmony_Server():
    def __init__(self, unique_user_list, user_mod_drop_dict, speed_distri, args):
        self.args = args

        self.num_modalities = args.dataset_opts["num_modalities"] if not args.simulated else args.simulated_number_of_modalities
        self.train_user_drop_dict = user_mod_drop_dict
        self.train_user_available_mod_dict = self.get_train_user_available_mod_dict()

        self.unique_user_list = unique_user_list

        self.save_parent_dir = f"results/{args.method}_{args.dataset}"
        self.create_save_dirs()
        # 4. Other settings
        self.total_num_users = len(self.unique_user_list)

        # Speed Configuration
        self.speed_distri = speed_distri
        self.allocate_device_per_client()
        self.init()
        self.total_communication_cost = 0.0
        self.total_number_of_trained_parameters = 0.0
        self.total_number_of_modalities_across_stage_1 = 0
        self.total_number_of_modalities_across_stage_2 = 0

        self.download_speed_list = []
        self.upload_speed_list = []
        self.comm_cost_list = []

    def init(self):
        self.existing_mod = self.get_train_user_available_mod_dict()
        # all_modalities = list(range(0, self.args.dataset_opts['num_modalities'])) if not self.args.simulated else list(range(0, self.args.simulated_number_of_modalities))
        # for client_idx, drop_mod_list in self.train_user_drop_dict.items():
        #     remaining_modalities = list(set(all_modalities).difference(set(drop_mod_list)))
        #     self.existing_mod[client_idx] = remaining_modalities

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
        start_time = time.time()
        self.run_stage1()  # 1. Stage 1: Modality-wise FL training (for all clients)
        self.run_stage2()  # 2. Stage 2: Federated fusion (for only multi-modal clients)
        elapsed_time = time.time() - start_time
        print(f"Total Communication Cost ==> {self.total_communication_cost}")
        print(f"Total Computation Cost ==> {self.total_number_of_trained_parameters}")
        print(f"Total Number of Modalities across all Training Rounds ==> {self.total_number_of_modalities_across_stage_1} + {self.total_number_of_modalities_across_stage_2}")
        log_communication_result_to_json(self.total_communication_cost, self.total_number_of_trained_parameters, self.args)
        print(f"Total time spent on training Harmony (stage 1 + stage 2): {elapsed_time:.2f} seconds")


    def run_stage1(self):
        self.print_header(message="Harmony: Stage 1", decorator_char="#")

        self.stage1_saved_model = {}

        for modality_idx in range(self.num_modalities):
            if self.args.simulated:
                modality_name = None
            else:
                modality_name = modality_idx_to_name(self.args.dataset, modality_idx)
            self.print_header(message=f"Running Stage 1 for Modality Idx:{modality_idx}, Name: {modality_name} ", decorator_char="-")

            # 1. Get list of clients with such available modality
            client_list_with_curr_modality = []
            for user_id in self.train_user_available_mod_dict.keys():
                if modality_idx in self.train_user_available_mod_dict[user_id]:
                    client_list_with_curr_modality.append(int(user_id))

            if len(client_list_with_curr_modality) == 0:
                print(f"No clients found with modality idx: {modality_idx}, name: {modality_name}. Skipping training for this modality")
                continue

            # print(modality_idx, len(client_list_with_curr_modality))

            client_list_with_curr_modality.sort()

            # 2. Set the modality indices
            boolean_modality_indices = [True if i == modality_idx else False for i in range(self.num_modalities)]
            # model = Dynamic_Model(boolean_modality_indices, self.device) # remove debug later
            model = get_dynamic_model(boolean_modality_indices, 'cpu', self.args)
            # 3. Train for each modality
            trained_model = self.train_unimodal_networks(client_list_with_curr_modality, model, modality_idx)

            # 4. Save the model
            self.save_unimodal_model(trained_model, modality_idx, modality_name)


            # 5. Save local test results for unimodal networks
            # self.save_local_test_results(local_test_result_dict, modality_idx, modality_name)

            print(f"------Stage 1 for Modality Idx: {modality_idx}, Name: {modality_name} completed-------")



    def train_unimodal_networks(self, client_list, model, modality_idx):
        train_loss_list = []
        available_client_size = len(client_list)
        num_clients_selected_per_round = int(available_client_size * self.args.percentage_selected_clients_per_round)

        global_rounds = self.args.hrm_stg1_fl_rounds
        print_every = self.args.print_every
        eval_every = self.args.eval_every

        total_communication = 0.0
        total_number_of_trained_parameters = 0
        total_number_of_modalities_across_all_training_rounds = 0
        for gl_round in tqdm(range(global_rounds)):
            model.train()

            print(f"Global round {gl_round} training for modality index: {modality_idx}")
            if self.args.hrm_stg1_use_all_clients:
                selected_clients = client_list # assume that all clients available do run stage 1, unrealistic
            else:
                selected_clients = np.random.choice(client_list, # typical setting, random choice
                                                    size=num_clients_selected_per_round,
                                                    replace=False)
                if len(selected_clients) == 0:
                    print("No clients selected. Skipping this round")
                    continue
            selected_clients.sort()
            print(f"Selected clients: {selected_clients}")
            for client_id in selected_clients:
                client_id = str(client_id)
                client = Harmony_Client(client_id, modality_info=self.existing_mod[client_id], device_info=self.device_dict[client_id], args=self.args)
                local_model, communication_cost, number_of_trained_parameters = client.train_stage_1(model, modality_idx)
                total_communication += communication_cost
                total_number_of_trained_parameters += number_of_trained_parameters
                total_number_of_modalities_across_all_training_rounds += len(self.existing_mod[client_id])

                self.comm_cost_list.append(communication_cost)

        # print(f'Global round: {gl_round + 1}, Total cost until now: {self.total_communication_cost} & {self.total_number_of_trained_parameters}')
        self.total_communication_cost += total_communication
        self.total_number_of_trained_parameters += total_number_of_trained_parameters
        self.total_number_of_modalities_across_stage_1 += total_number_of_modalities_across_all_training_rounds

        return model



    def run_stage2(self):
        self.print_header(message="Harmony: Stage 2", decorator_char="#")

        # 1. Create modality groups
        self.modality_groups = self.create_modality_groups()

        total_num = 0
        # 2. Perform modality fusion stage for each modality group
        for mod_group, value_dict in self.modality_groups.items():
            num_users_in_mod_group = len(value_dict['user_id_list'])
            self.print_header(message=f"Running MM-Fusion for Modality Group: {mod_group} "
                                      f"with {num_users_in_mod_group} users", decorator_char="-")
            self.train_multimodal_networks(value_dict)


    def train_multimodal_networks(self, modality_group_info):
        mod_group = modality_group_info['mod_group']
        available_mods = modality_group_info['available_mods']
        client_id_list = sorted(modality_group_info['user_id_list'])
        num_clusters = modality_group_info['num_clusters']

        # Debugging
        print(f"Within train_multimodal_networks: mod_group: {mod_group}, available_mods: {available_mods}, "
                f"client_id_list: {client_id_list}, num_clusters: {num_clusters}")

        train_loss_list = []
        available_client_size = len(client_id_list)
        # Here, we will use all available clients, not randomly sampled. ==> This might
        # give advantage to the Harmony baseline

        print_every, eval_every = self.args.print_every, self.args.eval_every
        global_rounds = self.args.hrm_stg2_fl_rounds

        boolean_modality_indices = [True if i in available_mods else False for i in range(self.num_modalities)]
        # model = Dynamic_Model(boolean_modality_indices, self.device)
        model = get_dynamic_model(boolean_modality_indices, 'cpu', self.args)
        model = self.load_pretrained_encoder_weights(model, available_mods)

        # Initially, all clusters have the same model
        # global_models = [copy.deepcopy(model) for _ in range(num_clusters)]
        global_classifier_layers = {
            cluster_id: copy.deepcopy(model.classifier).state_dict() for cluster_id, _ in enumerate(range(num_clusters))
        }
        # global_classifier_layers = [copy.deepcopy(model.classifier) for _ in range(num_clusters)]

        # This will be used to compute modality bias
        initial_model = copy.deepcopy(model)
        initial_model.encoder.eval()
        initial_model.classifier.eval()
        initial_model.eval()

        self.cluster_indices = np.zeros(available_client_size).astype(int)

        client_objects = []
        for client_id in client_id_list:
            client_id = str(client_id)
            client_obj = Harmony_Client(client_id, modality_info=self.existing_mod[client_id], device_info=self.device_dict[client_id], args=self.args)
            client_objects.append(client_obj)
            client_obj.set_initial_pretrained_models(initial_model)

        stage2_communication_cost = 0.0
        stage2_number_of_trained_parameters = 0
        stage2_number_of_modalities_across_all_training_rounds = 0
        for gl_round in tqdm(range(global_rounds)):
            local_classifier_weight_list = []
            all_encoder_dist_list = np.zeros((available_client_size, len(available_mods)))

            print(f"Stage 2, FL-Round: {gl_round} for Modality Group: {mod_group} "
                  f"with {available_client_size} and K={num_clusters}")
            for temp_idx, client_id in enumerate(client_id_list):
                # client = client_objects[temp_idx]
                client = Harmony_Client(client_id, modality_info=self.existing_mod[client_id],
                                        device_info=self.device_dict[client_id], args=self.args)
                client.set_initial_pretrained_models(initial_model)

                cluster_idx = self.get_cluster_idx(client_id, temp_idx, gl_round, verbose=True)
                # model_in_curr_cluster = global_models[cluster_idx]
                classifier_in_curr_cluster = global_classifier_layers[cluster_idx]

                local_classifier, local_encoder_list, local_communication_cost, local_number_of_trained_parameters = client.train_stage_2(classifier_in_curr_cluster,
                                                                                    gl_round)

                stage2_communication_cost += local_communication_cost
                stage2_number_of_trained_parameters += local_number_of_trained_parameters
                stage2_number_of_modalities_across_all_training_rounds += len(self.existing_mod[client_id])

                self.comm_cost_list.append(local_communication_cost)

                local_classifier_weight_list.append(copy.deepcopy(local_classifier))
                all_encoder_dist_list[temp_idx] = np.array(local_encoder_list)


            print(f"All valid clients completed training for FL-Round: {gl_round}")
            # Now need to aggregate and cluster
            all_encoder_dist_list = np.array(all_encoder_dist_list) # (num_users, num_modalities)
            print(f"Shape of all_encoder_dist_list: {all_encoder_dist_list.shape}")
            self.update_cluster_indices(all_encoder_dist_list, available_mods, num_clusters, gl_round)
            # Model aggregation using cluster indices
            global_classifier_layers = self.aggregate_clustered_classifiers(local_classifier_weight_list,
                                                                            client_id_list)

        self.total_communication_cost += stage2_communication_cost
        self.total_number_of_trained_parameters += stage2_number_of_trained_parameters
        self.total_number_of_modalities_across_stage_2 += stage2_number_of_modalities_across_all_training_rounds
        # utils.log_communication_result_to_json(self.total_communication_cost, self.args)

    def update_cluster_indices(self, encoder_dist_matrix, available_mods, num_clusters, fl_round):

        # 1. Compute max distance for each modality
        for temp_mod_id, modality_idx in enumerate(available_mods):
            max_dist = np.max(encoder_dist_matrix[:, temp_mod_id])
            if max_dist != 0:
                encoder_dist_matrix[:, temp_mod_id] = encoder_dist_matrix[:, temp_mod_id] / max_dist
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.args.seed).fit(encoder_dist_matrix)

        self.cluster_k_means_labels = kmeans.labels_

        return kmeans.labels_, encoder_dist_matrix


    def get_cluster_idx(self, client_id, temp_idx, gl_round, verbose=False):
        cluster_id = 0
        if gl_round == 0:
            cluster_id = 0
        else:
            cluster_id = self.cluster_k_means_labels[temp_idx]
        if verbose:
            print(f"Client: {client_id} with temp_idx: {temp_idx} assigned to cluster: {cluster_id} at round: {gl_round}")
        return cluster_id



    def load_pretrained_encoder_weights(self, model, available_mods):
        # Load uni-modal encoder weights trained in the first stage
        # for modality_idx in available_mods:
        #     modality_name = utils.modality_idx_to_name(self.args.dataset, modality_idx, self.args.dataset_opts['num_modalities'])
        #     path_to_model_cpt = f"{self.save_stage_1_dir}/{modality_idx}_{modality_name}_model.pt"
        #     state_dict = torch.load(path_to_model_cpt)
        #     encoder_weights = state_dict['encoder']
        #     model.encoder.modality_layers[modality_idx].load_state_dict(encoder_weights)
        #     print("Loaded encoder weights for modality: ", modality_name)

        for modality_idx in available_mods:
            # modality_name = utils.modality_idx_to_name(self.args.dataset, modality_idx,
            #                                            self.args.dataset_opts['num_modalities'])
            key = modality_idx
            pretrained_model = self.stage1_saved_model[key]
            encoder_weights = pretrained_model.encoder.modality_layers[modality_idx].state_dict()
            model.encoder.modality_layers[modality_idx].load_state_dict(encoder_weights)
            print("Loaded encoder weights for modality: ", key)

        return model


    def create_modality_groups(self, debug=False):
        # 1. Find unique modality sets
        modality_groups = {}
        for user_id, mod_list in self.train_user_available_mod_dict.items():
            # Sort mod_list and convert to string
            if len(mod_list) == 1: # Skip unimodal clients
                print(f"Will skip uni-modal client: {user_id} from Stage 2")
                continue

            unique_modality_value = "".join(map(str, sorted(mod_list)))
            if unique_modality_value not in modality_groups.keys():
                modality_groups[unique_modality_value] = {
                    'user_id_list': [user_id], # Append here as we go
                    'available_mods': mod_list,
                }
            else:
                modality_groups[unique_modality_value]['user_id_list'].append(user_id)

        # 2. Compute K (number of clusters) for each modality group. Will be used to cluster the clients.
        for mod_group, value_dict in modality_groups.items():
            if debug:
                print(f"Modality group: {mod_group}, user_id_list: {value_dict['user_id_list']}, available_mods: {value_dict['available_mods']}")
            updated_value_dict = {
                'mod_group': mod_group,
                'available_mods': value_dict['available_mods'],
                'user_id_list': value_dict['user_id_list'],
                'num_clusters': (int(len(value_dict['user_id_list']) - 1) // 5) + 1,
            }
            modality_groups[mod_group] = updated_value_dict
            if debug:
                print(f'Modality group: {mod_group}, updated_value_dict: {updated_value_dict}')
        return modality_groups


    def aggregate_clustered_classifiers(self, local_model_weight_list, client_id_list):
        clustered_classifier_weight_list = {} # classifier only
        global_classifiers = {}
        for temp_idx, client_id in enumerate(client_id_list):
            cluster_id = self.cluster_k_means_labels[temp_idx]
            if cluster_id not in clustered_classifier_weight_list.keys():
                clustered_classifier_weight_list[cluster_id] = [local_model_weight_list[temp_idx].state_dict()]
            else:
                clustered_classifier_weight_list[cluster_id].append(local_model_weight_list[temp_idx].state_dict())

        for cluster_id in clustered_classifier_weight_list.keys():
            local_weight_list_in_this_cluster = clustered_classifier_weight_list[cluster_id]
            agg_cls = self.aggregate_weights(local_weight_list_in_this_cluster)
            global_classifiers[cluster_id] = agg_cls

        return global_classifiers

    def aggregate_weights(self, local_weight_list):
        # fed average algorithm here
        w_avg = copy.deepcopy(local_weight_list[0])
        for key in w_avg.keys():
            for i in range(1, len(local_weight_list)):
                w_avg[key] += local_weight_list[i][key]
            w_avg[key] = torch.div(w_avg[key], len(local_weight_list))
        return w_avg





    def get_train_user_available_mod_dict(self):
        train_user_available_mod_dict = {}
        for user_id in self.train_user_drop_dict.keys():
            train_user_available_mod_dict[user_id] = [mod_id for mod_id in range(self.num_modalities) if mod_id not in self.train_user_drop_dict[user_id]]
        return train_user_available_mod_dict


    def print_header(self, message, decorator_char="="):
        len_message = len(message)
        top_bottom_decoration = decorator_char * len_message + decorator_char * 12
        msg_decoration = f"{decorator_char * 5} {message} {decorator_char * 5}"
        print(top_bottom_decoration)
        print(msg_decoration)
        print(top_bottom_decoration)



    def create_save_dirs(self):
        self.save_stage_1_dir = f"{self.save_parent_dir}/stage_1"
        self.save_stage_2_dir = f"{self.save_parent_dir}/stage_2"

        if not os.path.exists(self.save_stage_1_dir):
            os.makedirs(self.save_stage_1_dir)
        if not os.path.exists(self.save_stage_2_dir):
            os.makedirs(self.save_stage_2_dir)


    def save_unimodal_model(self, model, modality_idx, modality_name):
        # # Save encoder and classifier separately
        # state_dict = {
        #     'encoder': model.encoder.modality_layers[modality_idx].state_dict(),
        #     'classifier': model.classifier.state_dict(),
        # }
        # save_path = f"{self.save_stage_1_dir}/{modality_idx}_{modality_name}_model.pt"
        # print(f'Saving trained model to path: {save_path}')
        # torch.save(state_dict, save_path)


        key = modality_idx
        self.stage1_saved_model[key] = model
        print(f'Saving trained model to dictionary')