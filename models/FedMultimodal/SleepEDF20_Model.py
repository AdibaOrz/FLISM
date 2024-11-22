# reter to the code : https://github.com/getalp/PerCom2021-FL/blob/master/FedDist.ipynb

import torch
from torch import nn
from torch import Tensor

from configs import get_dataset_opts

# Dataset-specific meta
SleepEDF20_Opt = get_dataset_opts('sleepedf20')

class SleepEDF20_Dynamic_Model(nn.Module):
    def __init__(self):
        super(SleepEDF20_Dynamic_Model, self).__init__()

        self.num_modalities = SleepEDF20_Opt['num_modalities']
        self.encoder = Dynamic_Encoder(self.num_modalities, SleepEDF20_Opt['dim_lists'])
        self.classifier = Classifier(SleepEDF20_Opt['num_classes'])

    def forward(self, input_list):
        feature_lists = self.encoder(input_list)
        out = self.classifier(feature_lists)

        return out

class Dynamic_Encoder(nn.Module):
    def __init__(self, num_modalities, dim_lists):
        super(Dynamic_Encoder, self).__init__()

        self.num_modalities = num_modalities

        self.encoders = nn.ModuleList([
            Single_Encoder(input_channel_dim=len(dim_lists[idx]))
            for idx in range(num_modalities)
        ])

    def forward(self, input_list):
        feature_lists = []
        for idx in range(self.num_modalities):
            cur_encoder = self.encoders[idx]
            cur_input = input_list[idx]
            cur_features = cur_encoder(cur_input)
            feature_lists.append(cur_features)

        return feature_lists

class Single_Encoder(nn.Module):
    def __init__(self, input_channel_dim):
        super(Single_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=24, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=16, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Conv1d(64, 96, kernel_size=8, stride=1),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, input_data):
        out = self.encoder(input_data)
        flatten_layer = nn.Flatten()
        out = flatten_layer(out)

        return out


class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self,
        d_hid:  int=96,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        # print(x.shape)

        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        # print(x.shape)
        return x


class Classifier(nn.Module):  # Linear classifier
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fusion = FuseBaseSelfAttention()

        feature_flatten_dim = 384
        self.classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, features):
        features_mm = torch.cat(features, dim=1).unsqueeze(2)
        features_mm = features_mm.view(features_mm.shape[0], -1, 96)
        feature_mm_out = self.fusion(features_mm)
        return self.classifier(feature_mm_out)


if __name__ == "__main__":
    pass