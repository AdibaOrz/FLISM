"""
Harmony Models for RealWorldHAR dataset
"""
import torch
from torch import nn
from configs import get_dataset_opts

# Dataset-specific meta
RealWorldHAR_Opt = get_dataset_opts('realworldhar')
FEATURE_FLATTEN_DIM = 96


class single_modal_layers(nn.Module):
    def __init__(self, modality_type, debug=False):
        super().__init__()
        assert modality_type in ["other", "acc_or_gyro_3axes", "acc_gyro_6axes"]
        if modality_type == "acc_or_gyro_3axes":
            self.input_channel_dim = 3
        elif modality_type == "acc_gyro_6axes":
            self.input_channel_dim = 6
        else:
            raise ValueError(f"Invalid modality type: {modality_type}")

        self.debug = debug
        self.features = nn.Sequential(
            nn.Conv1d(self.input_channel_dim, 32, kernel_size=24, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=16, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(64, 96, kernel_size=8, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        features = self.features(x)
        features_flattened = features.view(x.size(0), -1)
        if self.debug:
            print(f'Class: {self.__class__.__name__}')
            print(f'Shape of input: {x.shape}')
            print(f'Shape of features after encoder: {features.shape}')
            print(f'Shape of features after flattening: {features_flattened.shape}')
        return features_flattened


class Dynamic_Encoder(nn.Module):
    def __init__(self, modality_indices, device, debug=False):
        super().__init__()
        """
        modality_indices: list of modality indices with true and false values
        Index: True => available modality; False => not available modality
        """
        self.modality_indices = modality_indices
        self.device = device
        self.debug = debug

        self.total_num_modalities = RealWorldHAR_Opt['num_modalities']
        self.available_modalities = self.modality_indices.count(True)
        self.modality_layers = [None] * self.total_num_modalities
        self.modality_type = None
        if self.total_num_modalities == 5: # 5 positions in the case of Realworld-HAR dataset
            self.modality_type = "acc_gyro_6axes"
        else:
            self.modality_type = "acc_or_gyro_3axes"

        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                self.modality_layers[idx] = single_modal_layers(self.modality_type, debug=self.debug).to(self.device)

        if self.debug:
            print(f'Number of available modalities: {self.available_modalities}')
            print(f'Number of modality layers: {len(self.modality_layers)}')

    def forward(self, x):
        output_list = []
        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                modality_layer = self.modality_layers[idx]
                input_data = get_modality_input_realworld(x, idx, self.modality_type)
                if self.debug:
                    print(f"==[{idx}] == // Shape of input data for modality at index {idx}: {input_data.shape}")
                curr_modality_output = modality_layer(input_data)
                output_list.append(curr_modality_output)
        return output_list


def get_modality_input_realworld(input_data, modality_index, modality_type):
    if modality_type == "acc_gyro_6axes":
        acc_part_of_data = input_data[:, 3 * modality_index: 3 * (modality_index+1), :]
        gyro_part_of_data = input_data[:, (3 * modality_index) + 15 : (3 * (modality_index+1)) + 15, :]
        # Concatenate these vertically
        concat_input_data = torch.cat((acc_part_of_data, gyro_part_of_data), dim=1)
        # print(f"Shape of acc_part_of_data: {acc_part_of_data.shape}, gyro_part_of_data: {gyro_part_of_data.shape}, concat_input_data: {concat_input_data.shape}")
        return concat_input_data
    elif modality_type == "acc_or_gyro_3axes":
        return input_data[:, 3 * modality_index: 3 * (modality_index+1), :]
    else:
        raise ValueError(f"Invalid modality type: {modality_type}")


class Dynamic_Model(nn.Module):
    def __init__(self, modality_indices, device, debug=False):
        super().__init__()
        self.modality_indices = modality_indices
        self.device = device
        self.debug = debug
        self.num_available_modalities = self.modality_indices.count(True)
        self.num_classes = RealWorldHAR_Opt['num_classes']
        self.total_num_modalities = RealWorldHAR_Opt['num_modalities']
        self.concat_feature_dim = self.num_available_modalities * FEATURE_FLATTEN_DIM

        self.encoder = Dynamic_Encoder(self.modality_indices, self.device, self.debug).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_feature_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.num_classes)
        ).to(self.device)


    def forward(self, x):
        output_list = self.encoder(x)
        if self.debug:
            print(f"CLASS: {self.__class__.__name__}")
            print(f"Len of output_list: {len(output_list)}")
            print(f"Shape of output_list[0]: {output_list[0].shape}")

        feature = torch.cat(output_list, dim=1) # concatenate all features extracted from encoder
        if self.debug:
            print(f"Shape of feature: {feature.shape}")
        classifier_output = self.classifier(feature) # forward pass through classifier

        if self.debug:
            print(f'Class: {self.__class__.__name__}')
            print(f'Shape of input[0]: {x[0].shape}')
            print(f'Shape of feature: {feature.shape}')
            print(f'Shape of classifier_output: {classifier_output.shape}')
        return classifier_output

