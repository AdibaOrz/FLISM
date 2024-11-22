import os
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.missing_utils import modality_idx_to_name
from federated.client.Harmony_Client import Harmony_Client
from utils.general_utils import log_test_results_to_json, print_header
from models.harmony_models.harmony_model_utils import get_dynamic_model


class Harmony_Server():
    def __init__(self, train_set, test_set, train_user_groups, test_user_groups, user_mod_drop_dict,
                 device, args):
        self.args = args
        self.device = device

        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_user_groups = train_user_groups
        self.test_user_groups = test_user_groups

        self.num_modalities = args.dataset_opts['num_modalities']
        self.train_user_drop_dict = user_mod_drop_dict
        self.train_user_available_mod_dict = self.get_train_user_available_mod_dict()
        self.total_num_users = len(self.train_user_groups)

        self.modality_groups = {}
        self.mm_client_local_test_dict = {}

        self.save_parent_dir = f"results/{args.method}_{args.dataset}"
        self.create_save_dirs()


    def run(self):
        print("=== Running Harmony Server ===")
        self.federate()
        self.evaluate()


    def federate(self):
        start_time = time.time()
        print_header(message="Harmony: Stage 1", decorator_char="#") # Modality-wise FL training (for all clients)
        self.run_stage1()

        print_header(message="Harmony: Stage 2", decorator_char="#") # Federated fusion (for only multi-modal clients)
        self.run_stage2()
        elapsed_time = time.time() - start_time
        print(f"Total time spent training Harmony (stage 1 + stage 2): {elapsed_time:.2f} sec.")


    def evaluate(self):
        print_header(message="Test Harmony", decorator_char="=")
        unimodal_clients = []
        for client_id in self.train_user_available_mod_dict.keys():
            if len(self.train_user_available_mod_dict[client_id]) == 1:
                modality_idx = self.train_user_available_mod_dict[client_id][0]
                modality_name = modality_idx_to_name(self.args.dataset, modality_idx)
                unimodal_clients.append((client_id, modality_idx, modality_name))

        unimodal_test_results = self.test_unimodal_clients(unimodal_clients)
        multimodal_test_results = self.test_multimodal_clients()
        local_test_result_dict = {**unimodal_test_results, **multimodal_test_results}

        f1_macro_list = []
        for client_id, result_dict in local_test_result_dict.items():
            f1_macro_list.append(local_test_result_dict[client_id]['f1_macro'])
        avg_f1_macro = sum(f1_macro_list) / len(f1_macro_list)
        # Average of all clients
        log_test_results_to_json(local_test_result_dict, avg_f1_macro, self.args)
        print(f"Average F1 (macro) across all clients: {avg_f1_macro:.4f}")


    def test_multimodal_clients(self):
        mm_test_result = {}
        for client_id, result_dict in self.mm_client_local_test_dict.items():
            print(f"[Multi-modal] Client Idx: {client_id}, Result Dict: {result_dict}")
            mm_test_result[client_id] ={
                "missing_mods" : result_dict["missing_mods"],
                "f1_macro" : float(result_dict["f1_macro"])
            }
        return mm_test_result


    def test_unimodal_clients(self, unimodal_clients):
        print_header(message="Test for Unimodal Clients", decorator_char="=")
        if len(unimodal_clients) == 0:
            print("No unimodal clients found. Skipping test for unimodal clients")
            return {}

        # ====== Test for unimodal clients ====== #
        test_result_dict = {}
        for (client_id, modality_idx, modality_name) in unimodal_clients:
            print(f"Testing for Client Idx: {client_id}, Modality Idx: {modality_idx}, Modality Name: {modality_name}")

            # Define client
            client = Harmony_Client(client_id, self.train_dataset, self.test_dataset,
                                    self.train_user_groups[client_id], self.test_user_groups[client_id],
                                    self.train_user_drop_dict[client_id], self.device, self.args)

            model = self.load_unimodal_full_model(modality_idx, modality_name)
            local_test_f1_macro = client.test_unimodal_client(model, modality_idx, modality_name)
            test_result_dict[client_id] = {
                "missing_mods": client.modality_indices_to_drop,
                "f1_macro": local_test_f1_macro
            }
            print(f"[Unimodal] Client Idx: {client_id}, Result_dict: {test_result_dict[client_id]}")
        return test_result_dict


    def run_stage1(self):
        for modality_idx in range(self.num_modalities):
            modality_name = modality_idx_to_name(self.args.dataset, modality_idx)
            print_header(f"Run Stage#1 for Modality Idx:{modality_idx}, Name: {modality_name}", "-")

            # 1. Get list of clients with such available modality
            client_list_with_curr_modality = []
            for user_id in self.train_user_available_mod_dict.keys():
                if modality_idx in self.train_user_available_mod_dict[user_id]:
                    client_list_with_curr_modality.append(user_id)

            if len(client_list_with_curr_modality) == 0:
                print(f"No clients found w/ modality_idx: {modality_idx}, name: {modality_name}. Skipping training for this modality")
                continue
            client_list_with_curr_modality.sort()

            # 2. Set the modality indices
            boolean_modality_indices = [True if i == modality_idx else False for i in range(self.num_modalities)]
            model = get_dynamic_model(boolean_modality_indices, self.device, self.args)

            # 3. Train for each modality
            trained_model = self.train_unimodal_networks(client_list_with_curr_modality, model, modality_idx)

            # 4. Save the model
            self.save_unimodal_model(trained_model, modality_idx, modality_name)
            # print(f"------Stage 1 for Modality Idx: {modality_idx}, Name: {modality_name} completed-------")


    def train_unimodal_networks(self, client_list, model, modality_idx):
        train_loss_list = []
        available_client_size = len(client_list)
        num_clients_selected_per_round = int(available_client_size * self.args.client_selection_rate)
        global_rounds = self.args.hrm_stg1_fl_rounds

        for rnd in tqdm(range(global_rounds), desc=f"Unimodal [modality={modality_idx}] Training Round"):
            model.train()
            local_encoder_weight_list, local_classifier_weight_list = [], []
            local_loss_list = []
            selected_clients = np.random.choice(client_list, num_clients_selected_per_round, replace=False)
            if len(selected_clients) == 0:
                print("No clients selected (No client satisfies selection condition). Skipping this round")
                continue
            selected_clients.sort()
            for client_id in selected_clients:
                client = Harmony_Client(client_id, self.train_dataset, self.test_dataset,
                                        self.train_user_groups[client_id], self.test_user_groups[client_id],
                                        self.train_user_drop_dict[client_id], self.device, self.args)
                updated_local_model, local_loss = client.train_stage_1(copy.deepcopy(model), modality_idx)
                local_encoder_weight_list.append(copy.deepcopy(updated_local_model.encoder.state_dict()))
                local_classifier_weight_list.append(copy.deepcopy(updated_local_model.classifier.state_dict()))
                local_loss_list.append(copy.deepcopy(local_loss))

            # Aggregate weights from clients
            global_encoder_weight = self.aggregate_weights(local_encoder_weight_list)
            global_classifier_weight = self.aggregate_weights(local_classifier_weight_list)
            model.encoder.load_state_dict(global_encoder_weight)
            model.classifier.load_state_dict(global_classifier_weight)

            loss_avg = sum(local_loss_list) / len(local_loss_list)
            train_loss_list.append(loss_avg)
            if rnd % 10 == 0:
                print(f"Round: {rnd} | Average Loss: {loss_avg:.3f}")
        # All global training rounds for this modality completed, return the trained model
        return model


    def run_stage2(self):
        self.modality_groups = self.create_modality_groups()
        local_test_results_all_clients = {}
        local_test_result_by_modality_groups = {}

        # 2. Perform modality fusion stage for each modality group
        for mod_group, value_dict in self.modality_groups.items():
            num_users_in_mod_group = len(value_dict['user_id_list'])
            print_header(f"Running MM-Fusion for Modality Group: {mod_group} with {num_users_in_mod_group} users", "-")

            local_test_dict = self.train_multimodal_networks(value_dict)
            local_test_result_by_modality_groups[mod_group] = local_test_dict
            for client_id, test_result in local_test_dict.items():
                test_result['mod_group'] = mod_group
                local_test_results_all_clients[client_id] = test_result

        self.mm_client_local_test_dict = local_test_results_all_clients


    def train_multimodal_networks(self, modality_group_info):
        mod_group = modality_group_info['mod_group']
        available_mods = modality_group_info['available_mods']
        client_id_list = sorted(modality_group_info['user_id_list'])
        num_clusters = modality_group_info['num_clusters']

        # Debugging message
        print(f"Within train_multimodal_networks: mod_group: {mod_group}, available_mods: {available_mods}, "
                f"client_id_list: {client_id_list}, num_clusters: {num_clusters}")

        available_client_size = len(client_id_list) # Here, we use all available clients
        global_rounds = self.args.hrm_stg2_fl_rounds
        boolean_modality_indices = [True if i in available_mods else False for i in range(self.num_modalities)]
        model = get_dynamic_model(boolean_modality_indices, self.device, self.args)
        model = self.load_pretrained_encoder_weights(model, available_mods)

        # Initially, all clusters have the same model
        # global_models = [copy.deepcopy(model) for _ in range(num_clusters)]
        global_classifier_layers = {
            cluster_id: copy.deepcopy(model.classifier).state_dict() for cluster_id, _ in enumerate(range(num_clusters))
        }

        # This will be used to compute modality bias
        initial_model = copy.deepcopy(model)
        initial_model.encoder.eval()
        initial_model.classifier.eval()
        initial_model.eval()

        self.cluster_indices = np.zeros(available_client_size).astype(int)
        client_objects = []
        for client_id in client_id_list:
            client_obj = Harmony_Client(client_id, self.train_dataset, self.test_dataset,
                                        self.train_user_groups[client_id], self.test_user_groups[client_id],
                                        self.train_user_drop_dict[client_id], self.device, self.args)
            client_objects.append(client_obj)
            client_obj.set_initial_pretrained_models(initial_model)

        for gl_round in tqdm(range(global_rounds)):
            local_classifier_weight_list = []
            all_encoder_dist_list = np.zeros((available_client_size, len(available_mods)))
            local_loss_list = []

            if self.args.verbose:
                print(f"Stage 2, FL-Round: {gl_round} for Modality Group: {mod_group} "
                      f"with {available_client_size} and K={num_clusters}")
            for temp_idx, client_id in enumerate(client_id_list):
                client = client_objects[temp_idx]

                cluster_idx = self.get_cluster_idx(client_id, temp_idx, gl_round, verbose=self.args.verbose)
                # model_in_curr_cluster = global_models[cluster_idx]
                classifier_in_curr_cluster = global_classifier_layers[cluster_idx]

                local_classifier, local_encoder_dist_list, local_loss = client.train_stage_2(classifier_in_curr_cluster)
                local_classifier_weight_list.append(copy.deepcopy(local_classifier))
                all_encoder_dist_list[temp_idx] = np.array(local_encoder_dist_list)
                local_loss_list.append(local_loss)

            if self.args.verbose:
                print(f"All valid clients completed training for FL-Round: {gl_round}")
            # Now need to aggregate and cluster
            all_encoder_dist_list = np.array(all_encoder_dist_list) # (num_users, num_modalities)
            self.update_cluster_indices(all_encoder_dist_list, available_mods, num_clusters, gl_round)
            # Model aggregation using cluster indices
            global_classifier_layers = self.aggregate_clustered_classifiers(local_classifier_weight_list,client_id_list)


        # ======================================= #
        # ===== Local Test after Stage 2 FL ===== #
        # ======================================= #
        print_header(message="Local Test after Stage 2 FL", decorator_char="=")
        local_test_result_dict = {}
        for enum_id, client_idx in enumerate(client_id_list):
            client = client_objects[enum_id]
            local_test_f1_macro = client.local_test_after_stage_2(latest_classifier_weights =
                                                              global_classifier_layers[self.cluster_indices[enum_id]])
            local_test_result_dict[client_idx] = {
                "missing_mods": client.modality_indices_to_drop,
                "f1_macro": local_test_f1_macro}

            missing_modality_num = len(client.modality_indices_to_drop)
            print(f"[{enum_id}] Client {client_idx} w/ {missing_modality_num} mis. modals:")
            print(f"F1-MACRO: {local_test_f1_macro:.3f}")
        return local_test_result_dict



    def update_cluster_indices(self, encoder_dist_matrix, available_mods, num_clusters, fl_round):
        print(f"==== Updating cluster indices at global fl round: {fl_round} ====")

        # 1. Compute max distance for each modality
        for temp_mod_id, modality_idx in enumerate(available_mods):
            max_dist = np.max(encoder_dist_matrix[:, temp_mod_id])
            if max_dist != 0:
                encoder_dist_matrix[:, temp_mod_id] = encoder_dist_matrix[:, temp_mod_id] / max_dist
        # print(f"Normalized encoder_dist_matrix: {encoder_dist_matrix}")
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.args.seed).fit(encoder_dist_matrix)
        # print(f"kmeans.labels_: {kmeans.labels_}")

        self.cluster_k_means_labels = kmeans.labels_
        return kmeans.labels_, encoder_dist_matrix


    def get_cluster_idx(self, client_id, temp_idx, gl_round, verbose=False):
        if gl_round == 0:
            cluster_id = 0
        else:
            cluster_id = self.cluster_k_means_labels[temp_idx]
        if verbose:
            print(f"Client: {client_id} with temp_idx: {temp_idx} assigned to cluster: {cluster_id} at round: {gl_round}")
        return cluster_id


    def load_pretrained_encoder_weights(self, model, available_mods):
        # Load uni-modal encoder weights trained in the first stage
        for modality_idx in available_mods:
            modality_name = modality_idx_to_name(self.args.dataset, modality_idx)
            path_to_model_cpt = f"{self.save_stage_1_dir}/{modality_idx}_{modality_name}_model.pt"
            state_dict = torch.load(path_to_model_cpt)
            encoder_weights = state_dict['encoder']
            model.encoder.modality_layers[modality_idx].load_state_dict(encoder_weights)
            if self.args.verbose:
                print("Loaded encoder weights for modality: ", modality_name)
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
                    'user_id_list': [user_id],
                    'available_mods': mod_list}
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


    def aggregate_weights(self, updated_local_model_list):
        # Equal weights
        weight_coefficient_list = [1 / len(updated_local_model_list) for _ in updated_local_model_list]
        w_avg = {key: torch.zeros_like(value) for key, value in updated_local_model_list[0].items()}
        for key in w_avg.keys():
            for i in range(len(updated_local_model_list)):  # i-th client, key-th layer
                w_avg[key] += updated_local_model_list[i][key] * weight_coefficient_list[i]
        return w_avg


    def create_save_dirs(self):
        self.save_stage_1_dir = f"{self.save_parent_dir}/stage_1"
        self.save_stage_2_dir = f"{self.save_parent_dir}/stage_2"

        if not os.path.exists(self.save_stage_1_dir):
            os.makedirs(self.save_stage_1_dir)
        if not os.path.exists(self.save_stage_2_dir):
            os.makedirs(self.save_stage_2_dir)


    def get_train_user_available_mod_dict(self):
        train_user_available_mod_dict = {}
        for user_id in self.train_user_drop_dict.keys():
            train_user_available_mod_dict[user_id] = [mod_id for mod_id in range(self.num_modalities) if mod_id not in self.train_user_drop_dict[user_id]]
        return train_user_available_mod_dict



    def load_unimodal_full_model(self, modality_idx, modality_name):
        boolean_modality_indices = [True if i == modality_idx else False for i in range(self.num_modalities)]
        model = get_dynamic_model(boolean_modality_indices, self.device, self.args)

        # Load pre-trained encoder and classifier weights
        path_to_model_cpt = f"{self.save_stage_1_dir}/{modality_idx}_{modality_name}_model.pt"
        state_dict = torch.load(path_to_model_cpt)
        model.encoder.modality_layers[modality_idx].load_state_dict(state_dict['encoder'])
        model.classifier.load_state_dict(state_dict['classifier'])
        return model


    def save_unimodal_model(self, model, modality_idx, modality_name):
        # Save encoder and classifier separately
        state_dict = {
            'encoder': model.encoder.modality_layers[modality_idx].state_dict(),
            'classifier': model.classifier.state_dict(),
        }
        save_path = f"{self.save_stage_1_dir}/{modality_idx}_{modality_name}_model.pt"
        print(f'Saving trained model to path: {save_path}')
        torch.save(state_dict, save_path)