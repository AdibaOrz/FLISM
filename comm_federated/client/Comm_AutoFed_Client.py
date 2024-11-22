"""
Baseline Method
Client-side code
=> Uses zero-imputation for missing modalities.
=> Does not apply any other methods to handle missing modalities.
"""
from comm_federated.client.client_utils import calculate_model_size, calculate_number_of_parameter
import random

class AutoFed_Client(object):
    def __init__(self, model, generative_model, user_idx, device_info, modality_to_drop, args):
        self.args = args
        self.user_idx = user_idx
        self.device_info = device_info
        self.model = model
        self.generative_model = generative_model
        self.is_complete = True if len(modality_to_drop)==0 else False

        self.init_upload_and_download_speed()

    def init_upload_and_download_speed(self):
        # print(f'client {self.user_idx} has device {self.device_info}')

        self.download_speed_u = self.device_info['down_u']
        self.download_speed_sigma = self.device_info['down_sigma']
        self.upload_speed_u = self.device_info['up_u']
        self.upload_speed_sigma = self.device_info['up_sigma']

    def get_download_time(self, models):
        download_speed = random.gauss(self.download_speed_u, self.download_speed_sigma)
        while download_speed < 0:
            download_speed = random.gauss(self.download_speed_u, self.download_speed_sigma)

        download_time = 0.0
        for model in models:
            model_size = calculate_model_size(model)
            if model_size == 0.0:
                return 0.0
            download_time += model_size / download_speed
        return float(download_time)

    def get_upload_time(self, models):
        upload_speed = random.gauss(self.upload_speed_u, self.upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = random.gauss(self.upload_speed_u, self.upload_speed_sigma)

        upload_time = 0.0
        for model in models:
            model_size = calculate_model_size(model)
            if model_size == 0.0:
                return 0.0
            upload_time += model_size / upload_speed
        return float(upload_time)
    def get_communication_cost_pretrain(self): # Local training
        download_time = self.get_download_time([self.generative_model])
        upload_time = self.get_upload_time([self.generative_model])

        local_communication_cost = download_time + upload_time

        local_number_of_trained_parameters = calculate_number_of_parameter(self.generative_model)

        return local_communication_cost, local_number_of_trained_parameters

    def get_communication_cost_maintrain(self):  # Local training

        if self.is_complete:
            download_time = self.get_download_time([self.model])
            upload_time = self.get_upload_time([self.model])
        else:
            download_time = self.get_download_time([self.model, self.generative_model])
            upload_time = self.get_upload_time([self.model, self.generative_model])

        local_communication_cost = download_time + upload_time

        local_number_of_trained_parameters = calculate_number_of_parameter(self.model)

        return local_communication_cost, local_number_of_trained_parameters



