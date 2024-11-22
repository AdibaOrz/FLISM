# Architecture reference: https://github.com/getalp/PerCom2021-FL/blob/master/FedDist.ipynb

import torch.nn as nn
from configs import get_dataset_opts

# Dataset-specific meta
Simulated_Opt = get_dataset_opts('simulated')
FEATURE_FLATTEN_DIM = 96


class Simulated_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.simulated_number_of_modalities
        self.num_classes = Simulated_Opt['num_classes']

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(64, 96, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.AdaptiveMaxPool1d(1))

        self.flatten_layer = nn.Flatten()
        self.projection_head = nn.Sequential(
            nn.Linear(FEATURE_FLATTEN_DIM, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 50))

        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_FLATTEN_DIM, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.num_classes))


    def forward(self, x, extract_embeds_only):
        out = self.encoder(x)
        out = self.flatten_layer(out)
        if extract_embeds_only:
            return self.projection_head(out)
        else:
            return self.classifier(out)
