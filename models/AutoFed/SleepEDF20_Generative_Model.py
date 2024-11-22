from torch import nn
from configs import get_dataset_opts

from models.AutoFed.generative_utils import generate_model_dict
# Dataset-specific meta
SleepEDF20_Opt = get_dataset_opts('sleepedf20')

class AutoEncoder(nn.Module):
    def __init__(self, input_channel_dim, output_channel_dim, name):
        super(AutoEncoder, self).__init__()
        self.name = name

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channel_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_channel_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def __str__(self):
        model_str = super(AutoEncoder, self).__str__()
        return f"{self.name}\n{model_str}"


class SleepEDF20_Dynamic_Autoencoder(nn.Module):
    def __init__(self):
        super(SleepEDF20_Dynamic_Autoencoder, self).__init__()

        self.dim_per_modalities = SleepEDF20_Opt['dim_per_modalities']
        self.total_num_modalities = SleepEDF20_Opt['num_modalities']
        self.generation_model_mapping_dict = generate_model_dict(self.total_num_modalities)
        self.total_num_autoencoders = int(self.total_num_modalities * (self.total_num_modalities - 1)) # N(N -1)
        self.autoencoder_layers = nn.ModuleList()

        for in_idx in range(self.total_num_modalities):
            for out_idx in range(self.total_num_modalities):
                if in_idx != out_idx:
                    key_for_dict = f'from_{in_idx}_to_{out_idx}'

                    in_dim = self.dim_per_modalities[in_idx]
                    out_dim = self.dim_per_modalities[out_idx]
                    autoencoder = AutoEncoder(in_dim, out_dim, key_for_dict)
                    self.autoencoder_layers.append(autoencoder)


if __name__ == "__main__":
    pass