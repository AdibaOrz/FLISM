"""
WESAD Models for Harmony
"""
import torch
from torch import nn
from configs import get_dataset_opts

# Dataset-specific meta
Simulated_Opt = get_dataset_opts('simulated')
FEATURE_FLATTEN_DIM = 96


class single_modal_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channel_dim = 1
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
        return features_flattened


class Dynamic_Encoder(nn.Module):
    def __init__(self, modality_indices, device, args):
        super().__init__()
        """
        modality_indices: list of modality indices
        Index is set to True => available, False => not available
        """
        self.modality_indices = modality_indices
        self.device = device
        self.args = args

        self.total_num_modalities = self.args.simulated_number_of_modalities
        self.num_available_modalities = self.modality_indices.count(True)
        self.modality_layers = [None] * self.total_num_modalities
        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                self.modality_layers[idx] = single_modal_layers().to(self.device)


    def forward(self, x):
        output_list = []
        for idx, is_available in enumerate(self.modality_indices):
            if is_available:
                modality_layer = self.modality_layers[idx]
                input_data = x[:, idx, :]
                input_data = input_data.unsqueeze(1)
                output_list.append(modality_layer(input_data))
        return output_list


class Dynamic_Model(nn.Module):
    def __init__(self, modality_indices, device, args):
        super().__init__()
        self.modality_indices = modality_indices
        self.device = device
        self.args = args

        self.num_available_modalities = self.modality_indices.count(True)
        self.num_classes = Simulated_Opt['num_classes']
        self.concat_feature_dim = self.num_available_modalities * FEATURE_FLATTEN_DIM

        self.encoder = Dynamic_Encoder(self.modality_indices, self.device, self.args).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_feature_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.num_classes)
        ).to(self.device)


    def forward(self, x):
        output_list = self.encoder(x)

        feature = torch.cat(output_list, dim=1)
        classifier_output = self.classifier(feature)

        return classifier_output
