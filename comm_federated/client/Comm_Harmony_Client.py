# Built-in imports
import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from comm_federated.client.client_utils import calculate_model_size, calculate_number_of_parameter

class Harmony_Client(object):
    def __init__(self, user_idx, modality_info, device_info, args):
        self.user_idx = user_idx
        self.total_modalities = args.dataset_opts['num_modalities'] if not args.simulated else args.simulated_number_of_modalities
        self.device_info = device_info
        self.available_modalities = modality_info

        self.args = args
        self.init_upload_and_download_speed()

    def init_upload_and_download_speed(self):
        # print(f'client {self.user_idx} has device {self.device_info}')

        self.download_speed_u = self.device_info['down_u']
        self.download_speed_sigma = self.device_info['down_sigma']
        self.upload_speed_u = self.device_info['up_u']
        self.upload_speed_sigma = self.device_info['up_sigma']

        self.download_speed = random.gauss(self.download_speed_u, self.download_speed_sigma)
        while self.download_speed < 0:
            self.download_speed = random.gauss(self.download_speed_u, self.download_speed_sigma)

        self.upload_speed = random.gauss(self.upload_speed_u, self.upload_speed_sigma)
        while self.upload_speed < 0:
            self.upload_speed = random.gauss(self.upload_speed_u, self.upload_speed_sigma)

    def get_download_time(self, models):
        download_time = 0.0
        for model in models:
            model_size = calculate_model_size(model)
            if model_size == 0.0:
                return 0.0
            download_time += model_size / self.download_speed
        return float(download_time)

    def get_upload_time(self, models):
        upload_time = 0.0
        for model in models:
            model_size = calculate_model_size(model)
            if model_size == 0.0:
                return 0.0
            upload_time += model_size / self.upload_speed
        return float(upload_time)

    def set_initial_pretrained_models(self, pretrained_model):
        """
        Initialize initial model as the pre-trained model from the server
        # Was pre-trained model uni-modal then?
        """
        self.initial_model = copy.deepcopy(pretrained_model)
        self.current_model = copy.deepcopy(pretrained_model)


    def train_stage_1(self, model, modality_to_train):
        """
        === Stage 1 (Uni-modal) Training  ===
        """
        download_time_model = self.get_download_time([model])
        upload_time_model = self.get_upload_time([model])

        local_communication_cost = download_time_model + upload_time_model

        local_number_of_trained_parameters = calculate_number_of_parameter(model)

        # Return trained model and average loss
        return model, local_communication_cost, local_number_of_trained_parameters



    def compare_curr_and_initial_model_weights(self, curr_model, initial_model, modality_idx, msg):
        print(f"===== MEAN and STD of initial and current models {msg} =====")
        with torch.no_grad():
            curr_mod_encoder_params = torch.cat([p.flatten() for p in curr_model.encoder.modality_layers[modality_idx].parameters()])
            init_mod_encoder_params = torch.cat([p.flatten() for p in initial_model.encoder.modality_layers[modality_idx].parameters()])
            print(f"Mean of curr_mod_encoder_params: {torch.mean(curr_mod_encoder_params)}, std: {torch.std(curr_mod_encoder_params)}")
            print(f"Mean of init_mod_encoder_params: {torch.mean(init_mod_encoder_params)}, std: {torch.std(init_mod_encoder_params)}")



    def train_stage_2(self, curr_classifier_weights, gl_round):
        # For debugging
        download_time_model = self.get_download_time([self.current_model])
        upload_time_model = self.get_upload_time([self.current_model])

        local_communication_cost = download_time_model + upload_time_model
        local_number_of_trained_parameters = calculate_number_of_parameter(self.current_model)
        # Part 2: Encoder dist computation
        encoder_dist_list = self.calculate_cosine_distance(self.current_model, self.initial_model, debug=True)
        return self.current_model.classifier, encoder_dist_list, local_communication_cost, local_number_of_trained_parameters



    def calculate_cosine_distance(self, curr_model, initial_model, debug=False):
        encoder_dist_list = []
        print(f"self.available_modalities : {self.available_modalities}")
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
            # print("Encoder dist dict: ", encoder_dist_dict)
            print("Encoder dist list: ", encoder_dist_list)

        return encoder_dist_list




class DatasetSplit(Dataset):
    def __init__(self, all_dataset, curr_user_indices):
        self.all_dataset = all_dataset  # data of all clients
        self.curr_user_indices = [int(i) for i in curr_user_indices]  # indices corresponding to this user only

    def __len__(self):
        return len(self.curr_user_indices)

    def __getitem__(self, item):
        input_data, label, class_label = self.all_dataset[self.curr_user_indices[item]]  # only items related to the current user (not all clients)
        return torch.tensor(input_data), torch.tensor(label)