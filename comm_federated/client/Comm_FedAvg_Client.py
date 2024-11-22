"""
Baseline Method
Client-side code
=> Uses zero-imputation for missing modalities.
=> Does not apply any other methods to handle missing modalities.
"""
from comm_federated.client.client_utils import calculate_model_size, calculate_number_of_parameter
import random

class FedAvg_Client(object):
    def __init__(self, model, user_idx, device_info, args):
        self.args = args
        self.user_idx = user_idx
        self.device_info = device_info
        self.model = model

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
    def get_communication_cost(self): # Local training
        download_model_lists = [self.model.encoder, self.model.classifier]
        upload_model_lists = [self.model.encoder, self.model.classifier]
        download_time = self.get_download_time(download_model_lists)
        upload_time = self.get_upload_time(upload_model_lists)

        local_communication_cost = download_time + upload_time

        local_number_of_trained_parameters = calculate_number_of_parameter(self.model.encoder)
        local_number_of_trained_parameters += calculate_number_of_parameter(self.model.classifier)

        return local_communication_cost, local_number_of_trained_parameters