"""
WESAD Models for Harmony
"""
import torch
from torch import nn
from configs import get_dataset_opts

# Dataset-specific meta
WESAD_Opt = get_dataset_opts('wesad')
FEATURE_FLATTEN_DIM = 96


class single_modal_layers(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.input_channel_dim = 1
        self.debug = debug
        self.features = nn.Sequential(
            nn.Conv1d(self.input_channel_dim, 32, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(64, 96, kernel_size=4, stride=1),
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
        modality_indices: list of modality indices
        Index is set to True => available, False => not available
        """
        self.modality_indices = modality_indices
        self.device = device
        self.debug = debug

        self.total_num_modalities = WESAD_Opt['num_modalities']
        self.num_available_modalities = self.modality_indices.count(True)
        self.modality_layers = [None] * self.total_num_modalities
        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                self.modality_layers[idx] = single_modal_layers(self.debug).to(self.device)

        if self.debug:
            print(f'Number of available modalities: {self.num_available_modalities}')
            print(f'Number of modality layers: {len(self.modality_layers)}')

    def forward(self, x):
        output_list = []
        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                modality_layer = self.modality_layers[idx]
                input_data = x[:, idx, :]
                input_data = input_data.unsqueeze(1)
                if self.debug:
                    print(f'==[{idx}] // Shape of input_data for modality at index {idx}: {input_data.shape}')
                output_list.append(modality_layer(input_data))
        return output_list


class Dynamic_Model(nn.Module):
    def __init__(self, modality_indices, device, debug=False):
        super().__init__()
        self.modality_indices = modality_indices
        self.device = device
        self.debug = debug

        self.num_available_modalities = self.modality_indices.count(True)
        self.num_classes = WESAD_Opt['num_classes']
        self.total_num_modalities = WESAD_Opt['num_modalities']
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
            print(f'==Shape of output_list: {len(output_list)}')
            print(f'==Shape of output_list[0]: {output_list[0].shape}')

        feature = torch.cat(output_list, dim=1)
        if self.debug:
            print(f"==Shape of feature: {feature.shape}")
        classifier_output = self.classifier(feature)

        if self.debug:
            print(f'Class: {self.__class__.__name__}')
            print(f'Shape of input[0]: {x[0].shape}')
            print(f'Shape of feature: {feature.shape}')
            print(f'Shape of classifier_output: {classifier_output.shape}')
        return classifier_output
